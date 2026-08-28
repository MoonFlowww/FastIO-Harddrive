// Per 1M rows (2026-08-19, fixed per-1M formula: float division + measured cycles_per_ns)
// O(1) unordered_map:  329.51 us  (total 19.57 ms; index load 1.12 s)
// avx512 fnv1a:          4.95 ms  (   3 captures in bloom body, GetDirectAccessView hash, fixed size)
// avx512 FSL:           10.49 ms  (3275 captures in bloom body)
// constsize:            11.21 ms
// Additive Binary:     342.78 us  (25 iters, total 20.36 ms)
// Tree Binary Srch:      8.42 ns  (500ns total)

// Raw Iterations Char[5]:    9.04 ms
// Raw Iter Hash<uint32_t>:   4.70 ms
//PS: AVX512-based use mask+ctz, vectorized search doesnt iterate over all of rows
//Reason why fnv1a is faster than iterations w/out logic



// best optimization is compression at 401


/*
== CPU ==
  AMD Ryzen 5 7600X 6-Core Processor
  Online CPUs                                  0-11
  SMT control                                  on
  cpufreq driver                               amd-pstate-epp
  amd_pstate mode                              active
  amd_pstate prefcore                          enabled
  policy0    gov=performance cur=5226390   range=[5457105..5457105] boost=1 epp=performance
  policy1    gov=performance cur=4640971   range=[5457105..5457105] boost=1 epp=performance
  policy10   gov=performance cur=4757589   range=[5457105..5457105] boost=1 epp=performance
  policy11   gov=performance cur=5358419   range=[5457105..5457105] boost=1 epp=performance
  policy2    gov=performance cur=5414149   range=[5457105..5457105] boost=1 epp=performance
  policy3    gov=performance cur=4419692   range=[5457105..5457105] boost=1 epp=performance
  policy4    gov=performance cur=4339627   range=[5457105..5457105] boost=1 epp=performance
  policy5    gov=performance cur=4355093   range=[5457105..5457105] boost=1 epp=performance
  policy6    gov=performance cur=4396619   range=[5457105..5457105] boost=1 epp=performance
  policy7    gov=performance cur=5203270   range=[5457105..5457105] boost=1 epp=performance
  policy8    gov=performance cur=4319628   range=[5457105..5457105] boost=1 epp=performance
  policy9    gov=performance cur=5191852   range=[5457105..5457105] boost=1 epp=performance

== Turbo / boost (global) ==
  cpufreq boost                                1

== Kernel / memory ==
  vm.swappiness                                1
  kernel.numa_balancing                        0
  kernel.randomize_va_space                    0
  kernel.nmi_watchdog                          0
  THP enabled                                  [always] madvise never
  THP defrag                                   always defer defer+madvise madvise [never]

== Tracing sysctls ==
  kernel.perf_event_paranoid                   2
  kernel.kptr_restrict                         0
  kernel.yama.ptrace_scope                     1

== Thermal ==
  hottest sensor                               59C
*/


#include <Compression.h>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <likwid-marker.h>
#include <stdlib.h>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RNTupleRange.hxx>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "latte.hpp"


static constexpr uint64_t N = 100'000'000;
static constexpr int name_len = 5;

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)      __builtin_expect(!!(x), 1)
#define UNLIKELY(x)    __builtin_expect(!!(x), 0) 
#endif


class XorPRNG {
public:
  uint64_t state;
  explicit XorPRNG(uint64_t seed = 0x9e3779b97f4a7c15ULL) {
    state = seed + 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    state = z ^ (z >> 31);
  }

  inline uint64_t next() {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 0x2545F4914F6CDD1DULL;
  }

  inline uint32_t next_bounded(uint32_t limit) {
    // next() is 64-bit; (next()*limit)>>32 would wrap the 96-bit product in
    // uint64_t and return garbage outside [0, limit). Use the top 32 bits of
    // the rng instead (modulo bias here is < 2^-32, irrelevant for data gen).
    return static_cast<uint32_t>(next() >> 32) % limit;
  }
};

static void RNG_String(XorPRNG& rng, char (&name)[name_len]) {
  static constexpr char chars[] = "abcdefghijklmnopqrstuvwxyz"; // 26 + NUL
  constexpr uint32_t char_count = sizeof(chars) - 1;
  for (char& c : name) {
    c = chars[rng.next_bounded(char_count)];
    LATTE_PULSE("0) Rng chars");
  }
}

static auto RNG_int(XorPRNG& rng) -> int {
  // 101 -> uniform[0, 100]
  return static_cast<int>(rng.next_bounded(101));
}


static auto fnv1a(const char* str, std::size_t len) -> uint32_t { 
  uint32_t hash = 0x811C9DC5; // Fowler-Noll-Vo hash magic number
  for (std::size_t i = 0; i < len; ++i) {
    hash ^= static_cast<unsigned char>(str[i]);
    hash *= 0x01000193u; //Fowler-Noll-Vo prime magic number
  }
  return hash;
}

static void saveIndex(const std::unordered_map<uint32_t, std::vector<uint64_t>>& index, const std::string& path) {
  std::ofstream f(path, std::ios::binary);

  uint64_t mapSize = index.size();
  f.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

  for (const auto& [hash, rows] : index) {
    f.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
    uint64_t rowCount = rows.size();
    f.write(reinterpret_cast<const char*>(&rowCount), sizeof(rowCount));
    f.write(reinterpret_cast<const char*>(rows.data()), rowCount * sizeof(uint64_t));
  }
}

static auto loadIndex(const std::string& path) -> std::unordered_map<uint32_t, std::vector<uint64_t>> {
  std::ifstream f(path, std::ios::binary);
  std::unordered_map<uint32_t, std::vector<uint64_t>> index;

  uint64_t mapSize;
  f.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));
  index.reserve(mapSize);

  for (uint64_t i = 0; i < mapSize; ++i) {
    uint32_t hash;
    f.read(reinterpret_cast<char*>(&hash), sizeof(hash));

    uint64_t rowCount;
    f.read(reinterpret_cast<char*>(&rowCount), sizeof(rowCount));

    std::vector<uint64_t> rows(rowCount);
    f.read(reinterpret_cast<char*>(rows.data()), rowCount * sizeof(uint64_t));

    index[hash] = std::move(rows);
  }
  return index;
}

void write(){ Latte::Fast::Start("1) Write");
  Latte::Fast::Start("1.1) Write init");
  auto model = ROOT::RNTupleModel::Create();
  // RNTuple fixed-size array column; a raw char[5] would be normalized to std::array<char,5> anyway.
  auto name = model->MakeField<std::array<char, name_len>>("name");
  auto hash_name = model->MakeField<uint32_t>("hash_name");
  auto age = model->MakeField<int>("age");


  ROOT::RNTupleWriteOptions opts;
  //opts.SetCompression(401);  // LZ4 (default = ZSTD 505)
  auto writer = ROOT::RNTupleWriter::Recreate(
    std::move(model), "Users", "./data/search/users.root", opts
  ); // when writer destructed, it write the footer in file
  //1) Fill() × 50k        ->  fills in-memory column buffers (pages)
  //1.2) page full (~64KB)   ->  page gets compressed (Optional, LZ4 here)
  //3) cluster threshold   ->  all column pages flushed to disk as one cluster (~50MB default)
  //4) destructor          ->  final cluster + footer (schema, cluster index) written


  Latte::Fast::Stop("1.1) Write init");

#if RUN_O1SEARCH
  std::unordered_map<uint32_t, std::vector<uint64_t>> index; //O(1) search if you know what will be searched
#endif  
  XorPRNG rng;
  for(int i = 0; std::cmp_less(i,N); ++i){
    char user[name_len];
    RNG_String(rng, user);
    uint32_t huser = fnv1a(user, name_len);
    std::memcpy(name->data(), user, name_len);
    *hash_name = huser;
    *age = RNG_int(rng);
    writer->Fill(); // Sit in ram, pushed by cluster
#if RUN_O1SEARCH
    index[huser].push_back(i);
#endif
    LATTE_PULSE("1.2) Write Loop");
  }
#if RUN_O1SEARCH
  Latte::Fast::Start("1.3) Write SaveIndex");
  saveIndex(index, "./data/search/users.idx");
  Latte::Fast::Stop("1.3) Write SaveIndex");
#endif  
  Latte::Fast::Stop("1) Write");
}



void sortAndSaveRNTuple() {
  Latte::Fast::Start("Sort RNTuple");

  // Read the RNTuple
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto vName = reader->GetView<std::array<char, name_len>>("name");
  auto vHName = reader->GetView<uint32_t>("hash_name");
  auto vAge = reader->GetView<int>("age");

  struct User {
    uint32_t hash_name;
    std::array<char, name_len> name;
    int age;
  };

  std::vector<User> users;

  // Read all entries
  Latte::Fast::Start("Reading");
  for (auto i : reader->GetEntryRange()) {
    users.push_back({
      vHName(i),
      vName(i),
      vAge(i)
    });
  }
  Latte::Fast::Stop("Reading");

  Latte::Fast::Start("Sorting");
  // Sort by hash_name
  std::sort(users.begin(), users.end(), 
            [](const User& a, const User& b) {
            return a.hash_name < b.hash_name;
            });
  Latte::Fast::Stop("Sorting");

  Latte::Fast::Start("Writing sorted data");
  // Write back to new RNTuple
  auto model = ROOT::RNTupleModel::Create();
  auto fldName = model->MakeField<std::array<char, name_len>>("name");
  auto fldHashName = model->MakeField<uint32_t>("hash_name");
  auto fldAge = model->MakeField<int>("age");

  ROOT::RNTupleWriteOptions opts;
  //opts.SetCompression(401);  // LZ4 (default = ZSTD 505)
  auto writer = ROOT::RNTupleWriter::Recreate(
    std::move(model), "Users", "./data/search/users.root", opts
  );

  for (const auto& user : users) {
    *fldName = user.name;
    *fldHashName = user.hash_name;
    *fldAge = user.age;
    writer->Fill();
  }

  Latte::Fast::Stop("Writing sorted data");
  Latte::Fast::Stop("Sort RNTuple");
}



struct SearchResult{
  std::unique_ptr<ROOT::RNTupleReader> reader;
  std::vector<uint64_t> matches;
  char tName[name_len]; // human readable name

  SearchResult(std::unique_ptr<ROOT::RNTupleReader> r, std::vector<uint64_t> m, const char (&n)[name_len])
  : reader(std::move(r)), matches(std::move(m)) {
    std::memcpy(tName, n, name_len);
  }
  SearchResult() = default;
};

//---------------------------------------------------------------------------------------------
auto SIMD_FSL_Search(const char (&tName)[name_len]) -> SearchResult{
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto vName = reader->GetView<std::array<char, name_len>>("name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

  Latte::Mid::Start("2) Search");
  __m512i tar_firstB = _mm512_set1_epi8(tName[0]);
  __m512i tar_secB   = _mm512_set1_epi8(tName[1]);
  __m512i tar_lastB  = _mm512_set1_epi8(tName[name_len - 1]);

  alignas(64) uint8_t firstB[64], secB[64], lastB[64];

  Latte::Fast::Start("2.1) SIMD body");
  uint64_t v = 0;
  for(; v+64<=nEntries; v+=64){ // database rolling
    for(uint32_t j = 0; j < 64; ++j){ // batch loading v64
      const auto& s = vName(v+j);
      firstB[j] = static_cast<uint8_t>(s.front());
      secB[j]   = static_cast<uint8_t>(s[1]);
      lastB[j]  = static_cast<uint8_t>(s.back());
    }
    //last loading step: convert to vectors
    __m512i vcan_firstB = _mm512_load_si512(firstB);
    __m512i vcan_secB   = _mm512_load_si512(secB);
    __m512i vcan_lastB  = _mm512_load_si512(lastB);

    __mmask64 eq1 = _mm512_cmpeq_epi8_mask(tar_firstB, vcan_firstB);
    __mmask64 eq2 = _mm512_cmpeq_epi8_mask(tar_secB, vcan_secB);
    __mmask64 eq3 = _mm512_cmpeq_epi8_mask(tar_lastB, vcan_lastB);
    uint64_t mask = eq1 & eq2 & eq3;

    while(mask){ // unroll mask to save bloomed
      LATTE_PULSE("2.1.1) unroll bloom");
      uint32_t j = __builtin_ctzll(mask);
      if(std::memcmp(tName, vName(v+j).data(), name_len)==0) matches.push_back(v+j);
      mask &= mask-1;
    }
  }

  Latte::Fast::Stop("2.1) SIMD body");
  Latte::Fast::Start("2.2) SIMD tail");

  for(; v<nEntries;++v){ // v tail (no bloom)
    LATTE_PULSE("2.2.1) Tail");
    if(std::memcmp(tName, vName(v).data(), name_len)==0) matches.push_back(v);
  }

  Latte::Fast::Stop("2.2) SIMD tail");
  Latte::Hard::Stop("2) Search"); 
  return SearchResult(std::move(reader), std::move(matches), tName); // (reader is unique_ptr)
}


//---------------------------------------------------------------------------------------------
auto SIMD_fnv1a_Search(const char (&tName)[name_len]) -> SearchResult{
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto vName = reader->GetView<std::array<char, name_len>>("name");
  auto vHView = reader->GetDirectAccessView<uint32_t>("hash_name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

  Latte::Mid::Start("2) Search");
  __m512i key = _mm512_set1_epi32(static_cast<int>(fnv1a(tName, name_len)));
  alignas(64) uint32_t hashed[16];

  Latte::Fast::Start("2.1) SIMD body");
  uint64_t v = 0;
  for(; v+16<=nEntries; v+=16){
    for(uint32_t j = 0; j < 16; ++j){ // loading v64
      const auto& h = vHView(v+j);
      hashed[j] = static_cast<uint32_t>(h);
    }
    __m512i vcan_hashed = _mm512_load_si512(hashed); // last step of loading

    uint64_t mask = _mm512_cmpeq_epi32_mask(key, vcan_hashed);

    while(mask){ // unroll mask to save bloomed
      LATTE_PULSE("2.1.1) unroll bloom");
      uint32_t j = __builtin_ctzll(mask);
      if(std::memcmp(tName, vName(v+j).data(), name_len)==0) matches.push_back(v+j);
      mask &= mask-1;
    }
  }


  Latte::Fast::Stop("2.1) SIMD body");
  Latte::Fast::Start("2.2) SIMD tail");

  for(; v<nEntries;++v){ // v tail (no bloom)
    LATTE_PULSE("2.2.1) Tail");
    if(std::memcmp(tName, vName(v).data(), name_len)==0) matches.push_back(v);
  }


  Latte::Fast::Stop("2.2) SIMD tail");
  Latte::Hard::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(matches), tName); //move to avoid destruct (reader is unique_ptr)
}


//---------------------------------------------------------------------------------------------
auto O1Search(const char (&tName)[name_len]) -> SearchResult {
  Latte::Fast::Start("LoadIndex");
  auto index = loadIndex("./data/search/users.idx");
  Latte::Fast::Stop("LoadIndex");

  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  Latte::Mid::Start("2) Search");
  uint32_t key = fnv1a(tName, name_len);
  Latte::Fast::Start("2.2) Search find");
  std::vector<uint64_t> matches;
  auto it = index.find(key);
  if (it != index.end()) {
    for (uint64_t row : it->second) {
      if (std::memcmp(tName, vName(row).data(), name_len) == 0) matches.push_back(row);
    }
  }
  Latte::Fast::Stop("2.2) Search find");
  Latte::Hard::Stop("2) Search");

  return SearchResult(std::move(reader), std::move(matches), tName);
}

//---------------------------------------------------------------------------------------------
auto ConstSearch(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::array<char, name_len>>("name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

  char xa[name_len];
  char xb[name_len];

  Latte::Mid::Start("2) Search");

  std::memcpy(xa, tName, name_len);
  for (uint64_t i = 0; i < nEntries; ++i) {
    const auto& iname = vName(i);
    std::memcpy(xb, iname.data(), name_len);
    if (std::memcmp(xa, xb, name_len) == 0) matches.push_back(i);
  }
  Latte::Hard::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(matches), tName);
}

//---------------------------------------------------------------------------------------------




auto BinarySearch(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vHName = reader->GetDirectAccessView<uint32_t>("hash_name");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  const uint64_t nEntries = reader->GetNEntries();

  std::vector<uint64_t> matches;
  matches.reserve(10);


  Latte::Mid::Start("2) Search");

  const uint32_t key = fnv1a(tName, name_len);

  Latte::Mid::Start("2.1) Body");

  uint64_t base = 0;
  uint64_t half = nEntries;
  while (half > 1) { // add-half or nothing, then half/2
    half = half >> 1; 
    const uint32_t value = vHName(base + half - 1);
    base += half & (uint64_t)(-(int64_t)(value < key)); // 0,1 -> 0,-1 -> 0x0,0xFF.. -> base+=0, base+=half
    //LATTE_PULSE("2.1.1) BODY LOOP");
  } base += (uint64_t)(vHName(base) < key);
  /* // add or substract next half to idx
   idx = 0
   next_half = n>>1
   while(next_half)
    idx+=next_half;
    auto cmp = key - vHName(idx);
    next_half &=0x7FFFFFFFFFFFFFFFLL; //reset sign
    next_half = ((cmp>0)-(cmp<0) * (next_half>>1)); //sign from cmp
  */


  Latte::Mid::Stop("2.1) Body");

  Latte::Mid::Start("2.2) tail");
  for (uint64_t i = base; i < nEntries && vHName(i) == key; ++i) matches.push_back(i);
  Latte::Mid::Stop("2.2) tail");

  Latte::Fast::Start("2.3) Deblooming");
  for (size_t i = 0; i < matches.size(); ) {
    if (std::memcmp(tName, vName(matches[i]).data(), name_len) != 0) matches.erase(matches.begin()+i);
    else ++i;
  }
  Latte::Fast::Stop("2.3) Deblooming");
  Latte::Mid::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(matches), tName);
}


static inline auto BinaryTreeDepth(uint64_t N) -> int {
  return std::bit_width(N)-1;
}


void BuildBinaryTree(
  ROOT::RNTupleDirectAccessView<uint32_t>& vHName,
  std::vector<uint32_t>& tree_v,
  std::vector<uint32_t>& tree_i,
  int64_t start, int64_t stop, size_t idx)
{
  if (start > stop) return;

  const int64_t mid = start + (stop - start) / 2;
  tree_v[idx] = vHName(static_cast<uint64_t>(mid));  //DB value
  tree_i[idx] = static_cast<uint32_t>(mid); //DB idx

  BuildBinaryTree(vHName, tree_v, tree_i, start, mid-1, 2 * idx + 1); // left
  BuildBinaryTree(vHName, tree_v, tree_i, mid+1, stop, 2 * idx + 2); //right
}




auto TreeBinarySearch(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vHName = reader->GetDirectAccessView<uint32_t>("hash_name");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  // Perfect tree size = 2^n - 1
  size_t perfect_size = std::bit_ceil(N + 1) - 1;
  std::vector<uint32_t> tree_v(perfect_size, UINT32_MAX);
  std::vector<uint32_t> tree_i(perfect_size, UINT32_MAX);

  Latte::Fast::Start("BuildTree");
  BuildBinaryTree(vHName, tree_v, tree_i, 0, N - 1, 0);
  Latte::Fast::Stop("BuildTree");


  std::vector<uint64_t> matches;
  matches.reserve(10);

  Latte::Fast::Start("2) Search");
  uint32_t key = fnv1a(tName, name_len);
  Latte::Fast::Start("2.1) Search Body");

  size_t idx = 0;
  bool found = false;

  while (idx < tree_v.size()) {
    const uint32_t node = tree_v[idx];
    if (__builtin_expect(node == key, false)) { 
      found = true;
      break;
    }
    //idx = 2*idx + (2 - (key < node));
    idx = key < node ? (2 * idx + 1) : (2 * idx + 2); // faster than bool substraction
    //LATTE_PULSE("2.1.1) BODY LOOP");
  }  
  Latte::Fast::Stop("2.1) Search Body");
  /* 37.71ns, 2.24us
    const uint64_t node = tree_v[idx];
    uint64_t mask = (uint64_t)-int64_t(node == key); // 0x00 or 0xFF
    found = (idx & mask) | (found & ~mask);
    idx = (2 * idx + (2-(key < node))); //slower than (x ? y : z)
  */

  Latte::Fast::Start("2.2) Search Tail");
  if (found) { //vName is IO bottleneck
    uint64_t row = tree_i[idx];
    uint64_t lo = row;
    while (lo > 0 && vHName(lo - 1) == key) --lo;
    uint64_t hi = row;
    while (hi + 1 < N && vHName(hi + 1) == key) ++hi;

    for (uint64_t r = lo; r <= hi; ++r) {
      if (std::memcmp(tName, vName(r).data(), name_len) == 0) matches.push_back(r);
    }
  }
  Latte::Fast::Stop("2.2) Search Tail");
  Latte::Fast::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(matches), tName);
}


void BuildTernaryTree(
  ROOT::RNTupleDirectAccessView<uint32_t>& vHName,
  std::vector<uint32_t>& tree_v,
  std::vector<uint32_t>& tree_i,
  int64_t start, int64_t stop, size_t node_idx)
{
  if (start > stop) return;
  const int64_t low  = start + (stop - start) / 3;
  const int64_t high = start + 2 * (stop - start) / 3;

  const size_t base = 2 * node_idx;          // two slots per node
  tree_v[base]     = vHName(low);
  tree_i[base]     = low;
  tree_v[base + 1] = vHName(high);
  tree_i[base + 1] = high;

  BuildTernaryTree(vHName, tree_v, tree_i, start, low - 1,   3 * node_idx + 1);
  BuildTernaryTree(vHName, tree_v, tree_i, low + 1, high - 1, 3 * node_idx + 2);
  BuildTernaryTree(vHName, tree_v, tree_i, high + 1, stop,   3 * node_idx + 3);
}

static inline auto TernaryTreeDepth(const uint64_t N) -> int {
  if (N <= 0) return -1;
  double val = 2.0 * (double)N - 1.0;
  int h = static_cast<int>(std::floor(std::log(val) / std::log(3.0)));
  // Correct potential floating-point errors
  while (std::pow(3, h+1) <= val) ++h;
  while (std::pow(3, h) > val) --h;
  return h;
}


auto TreeTernarySearch(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vHName = reader->GetDirectAccessView<uint32_t>("hash_name");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  constexpr auto depth = 17; //TernaryTreeDepth(N);
  size_t node_cap = (std::pow(3, depth + 1) - 1) / 2;
  std::vector<uint32_t> tree_v(2 * node_cap, UINT32_MAX);
  std::vector<uint32_t> tree_i(2 * node_cap, UINT32_MAX);

  Latte::Fast::Start("BuildTree");
  BuildTernaryTree(vHName, tree_v, tree_i, 0, N - 1, 0);
  Latte::Fast::Stop("BuildTree");


  std::vector<uint64_t> matches;
  matches.reserve(10);

  Latte::Fast::Start("2) Search");
  uint32_t key = fnv1a(tName, name_len);
  Latte::Fast::Start("2.1) Search Body");

  uint64_t idx = 0;
  uint64_t found = 0;

  #pragma unroll(depth)
  for (int d = 0; d < depth; ++d) {
    const size_t base = 2 * idx;
    const uint32_t low  = tree_v[base];
    const uint32_t high = tree_v[base + 1];

    const int off = 1 + (key > high) - (key < low); // 0 left, 1 mid, 2 right

    const bool mask_low  = (key == low);
    const bool mask_high = (key == high);

    //const bool mask_side = (key==high); //== ((!(key==low)) | (key==high)) -> (1 0) or (0 1) ->! (0 0) or (1 1) ->| 0 or 1
    //found = (idx+mask_high & mask_pass) | (found & ~mask_pass);
    found = (off == 1) & (mask_low | mask_high) ? base + mask_high : found;  //fast path.  off==1 only if off=1+0-0

    idx = 3 * idx + off + 1;
    //LATTE_PULSE("2.1.1) BODY LOOP");
  }

  /*
  1) cmp-chain
      key < low   ; cmpl -> left
      key > high  ; cmph -> high
           -> 3 * idx + (1 + (cmph - cmpl))

  2) asymmetric memcmp
      key-value -> [-x;+x]
                  -> delta>>32-1 : left    ; sign == (-), compiler(x<0) == x>>31
                  -> delta== 0   : middle  ; key == value
                  -> else        : right   ; else

  3) symmetric memcmp (cmp-chain with 3sub instead of 2)
      key-value -> [-x;+x] 
                  -> ((cmp>0)-(cmp<0)) 
                    -> 3 * idx + 1 - r

      in any case i hit (3 * idx + (1 - something))


  1) delta
      cmpright = key - high;
      cmpleft = key - low;
      off = 1 + cmpright>0 - cmpleft>0

      3sub + 2cmp

  2) inverse of delta, cmp first the delta
      off = 1 + key > high - key < low

      mask = uint-int(key==high | key==low) // need 2 mask: mask1= side, mask2=override/keep

  */
  Latte::Fast::Stop("2.1) Search Body");




  Latte::Fast::Start("2.2) Search Tail");
  if (found) { //vName is IO bottleneck
    uint64_t row = tree_i[found];
    uint64_t lo = row;
    while (lo > 0 && vHName(lo - 1) == key) --lo;
    uint64_t hi = row;
    while (hi + 1 < N && vHName(hi + 1) == key) ++hi;

    for (uint64_t r = lo; r <= hi; ++r) {
      if (std::memcmp(tName, vName(r).data(), name_len) == 0) matches.push_back(r);
    }
  }
  Latte::Fast::Stop("2.2) Search Tail");
  Latte::Fast::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(matches), tName);
}


//---------------------------------------------------------------------------------------------

template <class T>
static inline void DoNotOptimize(const T& v) {
  asm volatile("" : : "r,m"(v) : "memory");
}

auto NoSearch(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::array<char, name_len>>("name");
  const auto nEntries = reader->GetNEntries();

  Latte::Mid::Start("2) Search");
  for (uint64_t i = 0; i < nEntries; ++i) {
    const auto& s = vName(i);
    DoNotOptimize(s.front()); // iterations cost 
  }
  Latte::Hard::Stop("2) Search");

  return SearchResult(std::move(reader), std::vector<uint64_t>{}, tName);
}


auto NoSearchHash(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vHView = reader->GetDirectAccessView<uint32_t>("hash_name");
  const auto nEntries = reader->GetNEntries();

  Latte::Mid::Start("2) Search");
  for (uint64_t i = 0; i < nEntries; ++i) {
    const auto& s = vHView(i);
    DoNotOptimize(s); // iterations cost 
  }
  Latte::Hard::Stop("2) Search");

  return SearchResult(std::move(reader), std::vector<uint64_t>{}, tName);
}


//---------------------------------------------------------------------------------------------

void read(SearchResult& Sresult){
  Latte::Mid::Start("3) Read");
  Latte::Fast::Start("3.1) Read Init");
  auto vName = Sresult.reader->GetView<std::array<char, name_len>>("name");
  auto vAge = Sresult.reader->GetDirectAccessView<int>("age");
  Latte::Fast::Stop("3.1) Read Init");
  std::cout 
    << "[Sys] Found " << Sresult.matches.size() 
    << " users named: " << std::string_view(Sresult.tName, name_len)
    << "\n   Within a database made of " << Sresult.reader->GetNEntries() 
    << " users" << '\n';


  for(auto&idx:Sresult.matches){
    const auto& nm = vName(idx);
    std::cout << "[" << idx << "]";
    std::cout << " [User]: ";
    std::cout.write(nm.data(), nm.size());
    std::cout << " [Age]: " << vAge(idx) << '\n';
    LATTE_PULSE("3.2) Read Findings");
  }
  Latte::Hard::Stop("3) Read");
}


auto main() -> int{
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_REGISTER("1_Write");
  LIKWID_MARKER_REGISTER("2_Search");
#if RUN_O1SEARCH
  std::cout << "[" << __TIME__ << "] Use O(1) via unordered_map"  << std::endl;
#elif RUN_SIMDFSLSEARCH
  std::cout << "[" << __TIME__ << "] Iterative AVX512 + FSL" << std::endl;
#elif RUN_SIMDFNV1ASEARCH
  std::cout << "[" << __TIME__ << "] Iterative AVX512 + fnv1a" << std::endl;
#elif RUN_CONSTSEARCH
  std::cout << "[" << __TIME__ << "] Iterative const size"  << std::endl;
#elif RUN_BINARYSEARCH
  std::cout << "[" << __TIME__ << "] Binary search"  << std::endl;
#elif RUN_TREEBINARYSEARCH
  std::cout << "[" << __TIME__ << "] Precomputed Binary Search"  << std::endl;
#elif RUN_TREETERNARYSEARCH
  std::cout << "[" << __TIME__ << "] Precomputed Ternary search"  << std::endl;
#elif RUN_NOSEARCH
  std::cout << "[" << __TIME__ << "] Baseline cost of iterating over RNTuple"  << std::endl;
#elif RUN_NOSEARCHHASH
  std::cout << "[" << __TIME__ << "] Baseline cost of iterating over RNTuple with hash"  << std::endl;
#else 
  std::cout << "No Parameter search function given, Abort()" << '\n';
  abort();
#endif


  char tName[name_len] = {'a', 'l', 'i', 'c', 'e'};
  Latte::Mid::Start("Global");
  LIKWID_MARKER_START("1_Write");
  write();
  LIKWID_MARKER_STOP("1_Write");


  SearchResult Sresult;
  LIKWID_MARKER_START("2_Search");
#if RUN_O1SEARCH
  Sresult = O1Search(tName); // academically fast O(1)
#elif RUN_SIMDFSLSEARCH
  Sresult = SIMD_FSL_Search(tName); // AVX512 with bloom first+second+last name char
#elif RUN_SIMDFNV1ASEARCH
  Sresult = SIMD_fnv1a_Search(tName); // AVX512 with bloom fnv1a hashing
#elif RUN_CONSTSEARCH
  Sresult = ConstSearch(tName); // constant name size
#elif RUN_BINARYSEARCH
  sortAndSaveRNTuple();
  Sresult = BinarySearch(tName); // Log2(N) search, branchless
#elif RUN_TREEBINARYSEARCH
  sortAndSaveRNTuple();
  Sresult = TreeBinarySearch(tName); // Log2(N) search, memcmp style, add/sub
#elif RUN_TREETERNARYSEARCH
  sortAndSaveRNTuple();
  Sresult = TreeTernarySearch(tName); // log3(N) search, branchless + fast-pass
#elif RUN_NOSEARCH
  Sresult = NoSearch(tName); // iterations cost, no search
#elif RUN_NOSEARCHHASH
  Sresult = NoSearchHash(tName); // iterations cost with fixed size uint32_t, no search
#endif
  LIKWID_MARKER_STOP("2_Search");


  read(Sresult);
  Latte::Hard::Stop("Global");


  Latte::DumpToStream(std::cout, Latte::Parameter::Time, Latte::Parameter::Calibrated);
  auto snap_Search = Latte::Snapshot("2) Search")[0];
  double cycles_per_ns;
  LATTE_FREQ(cycles_per_ns);
  std::cout << "Searching took: " << Latte::FormatTime(snap_Search/cycles_per_ns) << '\n';
#if RUN_CONSTSEARCH || RUN_NOSEARCH || RUN_NOSEARCHHASH || RUN_SIMDFSLSEARCH || RUN_SIMDFNV1ASEARCH
  std::cout << "[Expected] RNtuple search took "
    << Latte::FormatTime((static_cast<double>(snap_Search)/static_cast<double>(N)*1'000'000.0)/cycles_per_ns)
    << " per 1M iters " << '\n';
#elif RUN_TREEBINARYSEARCH || RUN_BINARYSEARCH
  std::cout << "[Expected] RNtuple search took "
    << Latte::FormatTime((static_cast<double>(snap_Search)/static_cast<double>(std::bit_width(N)))/cycles_per_ns)
    << " per depth ("<< std::bit_width(N)<< ")" << '\n';

#elif RUN_TREETERNARYSEARCH
  std::cout << "[Expected] RNtuple search took "
    << Latte::FormatTime((static_cast<double>(snap_Search)/static_cast<double>(std::bit_width(N)))/cycles_per_ns)
    << " per depth ("<< TernaryTreeDepth(N) << ")" << '\n';

#endif //O1 doesnt need print

  auto LargeFormat = [](double val) -> std::string {
    const char* units[] = {"", "K", "M", "B", "T"};
    int unit_idx = 0;
    while (val >= 1000.0 && unit_idx < 4) { val /= 1000.0; unit_idx++; }
    std::ostringstream ss;
    if (unit_idx == 0) ss << std::fixed << std::setprecision(0) << val;
    else ss << std::fixed << std::setprecision(2) << val << " " << units[unit_idx];
    return ss.str();
  };

  std::cout << "Total Rows: " << LargeFormat(N);
  std::cout << '\n';

  LIKWID_MARKER_CLOSE;
}

