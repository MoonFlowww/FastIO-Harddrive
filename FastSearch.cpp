// Per 1M rows
// O(1):              20.00 ms 
// avx512 fnv1a:      12.98 ms 
// avx512 FSL:        16.81 ms
// constsize O(1):    20.21 ms 


// Raw iterations:    16.38 ms 
//PS: AVX512-based use mask+ctz, vectorized search doesnt iterate over all of rows
//Reason why fnv1a is faster than iterations w/out logic



// best optimization is compression = 401



//TODO:
//  -slice GPU decoding + unsync multithread for lightweight max speed
//  - use std::array<char, 5> instead of string as name, if fixed size known

//--------------------------------------------- Academically fast through unordered_map O(1) ---------------------
/*
./bin/fastsearch_o1
[19:14:47] Use O(1) via unordered_map
[Sys] Found 4 users named: alice
   Within a database made of 59406880 users
 [User]: alice [Age]: 45
 [User]: alice [Age]: 19
 [User]: alice [Age]: 30
 [User]: alice [Age]: 61

#========================================================================================================================================#
| LATTE TELEMETRY [TIME][RAW]                                                                                                            |
#========================================================================================================================================#
| COMPONENT            |   SAMPLES |        AVG |     MEDIAN |    STD DEV |     SKEW |        MIN |        MAX |      RANGE |    OUTLIER |
|----------------------------------------------------------------------------------------------------------------------------------------|
| 2) Search            |         1 |     1.20 s |     1.20 s |    0.00 ns |     0.00 |     1.20 s |     1.20 s |    0.00 ns |          0 |
| 2.1) Search LoadInde |         1 |     1.20 s |     1.20 s |    0.00 ns |     0.00 |     1.20 s |     1.20 s |    0.00 ns |          0 |
| 2.2) Search find     |         1 |  435.67 us |  435.67 us |    0.00 ns |     0.00 |  435.67 us |  435.67 us |    0.00 ns |          0 |
| 1) Write             |         1 |    27.90 s |    27.90 s |    0.00 ns |     0.00 |    27.90 s |    27.90 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   88.99 ms |   88.99 ms |    0.00 ns |     0.00 |   88.99 ms |   88.99 ms |    0.00 ns |          0 |
| 0) Rng chars         |     65535 |   90.44 ns |    9.79 ns |  174.37 ns |     2.40 |    0.21 ns |    3.87 us |    3.87 us |          1 |
| 1.2) Write Loop      |     65533 |  461.09 ns |  450.00 ns |  142.33 ns |     6.18 |   70.00 ns |    5.14 us |    5.07 us |          3 |
| 1.3) Write SaveIndex |         1 |     1.77 s |     1.77 s |    0.00 ns |     0.00 |     1.77 s |     1.77 s |    0.00 ns |          0 |
| 3) Read              |         1 |   23.64 ms |   23.64 ms |    0.00 ns |     0.00 |   23.64 ms |   23.64 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   22.25 us |   22.25 us |    0.00 ns |     0.00 |   22.25 us |   22.25 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    5.16 ms |    2.75 ms |    3.95 ms |     0.69 |    1.99 ms |   10.73 ms |    8.74 ms |          0 |
| Global               |         1 |    32.24 s |    32.24 s |    0.00 ns |     0.00 |    32.24 s |    32.24 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 1.20 s
[Expected] RNtuple search take 20.00 ms per 1M iters 
Total Rows: 59.41 M
*/

/* -------------------------------------------- Iterative AVX512 + bloom(fnv1a) (single thread)
./bin/fastsearch_fnv1a
[19:13:25] use Iterative AVX512 + fnv1a
[Sys] Found 4 users named: alice
   Within a database made of 59406880 users
 [User]: alice [Age]: 45
 [User]: alice [Age]: 19
 [User]: alice [Age]: 30
 [User]: alice [Age]: 61

#========================================================================================================================================#
| LATTE TELEMETRY [TIME][RAW]                                                                                                            |
#========================================================================================================================================#
| COMPONENT            |   SAMPLES |        AVG |     MEDIAN |    STD DEV |     SKEW |        MIN |        MAX |      RANGE |    OUTLIER |
|----------------------------------------------------------------------------------------------------------------------------------------|
| 2) Search            |         1 |  772.59 ms |  772.59 ms |    0.00 ns |     0.00 |  772.59 ms |  772.59 ms |    0.00 ns |          0 |
| 2.1) SIMD body       |         1 |  771.74 ms |  771.74 ms |    0.00 ns |     0.00 |  771.74 ms |  771.74 ms |    0.00 ns |          0 |
| 2.2) SIMD tail       |         1 |  618.77 us |  618.77 us |    0.00 ns |     0.00 |  618.77 us |  618.77 us |    0.00 ns |          0 |
| 1) Write             |         1 |     3.88 s |     3.88 s |    0.00 ns |     0.00 |     3.88 s |     3.88 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   89.86 ms |   89.86 ms |    0.00 ns |     0.00 |   89.86 ms |   89.86 ms |    0.00 ns |          0 |
| 0) Rng chars         |     65536 |   10.59 ns |    9.79 ns |    9.44 ns |     1.87 |    0.21 ns |  110.00 ns |  109.79 ns |          0 |
| 1.2) Write Loop      |     65525 |   53.03 ns |   50.00 ns |   10.20 ns |     6.50 |   40.00 ns |  220.00 ns |  180.00 ns |         11 |
| 3) Read              |         1 |   22.98 ms |   22.98 ms |    0.00 ns |     0.00 |   22.98 ms |   22.98 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   17.09 us |   17.09 us |    0.00 ns |     0.00 |   17.09 us |   17.09 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    5.13 ms |    2.61 ms |    4.05 ms |     0.69 |    1.94 ms |   10.84 ms |    8.90 ms |          0 |
| Global               |         1 |     4.81 s |     4.81 s |    0.00 ns |     0.00 |     4.81 s |     4.81 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 772.59 ms
[Expected] RNtuple search take 12.98 ms per 1M iters 
Total Rows: 59.41 M
*/




/* ---------------------------------------------------Iterative AVX512 + bloom(First, Second and Last char) (single thread)
./bin/fastsearch_fsl
[19:08:31] use Iterative AVX512 + FSL to search the name
[Sys] Found 4 users named: alice
   Within a database made of 59406880 users
 [User]: alice [Age]: 45
 [User]: alice [Age]: 19
 [User]: alice [Age]: 30
 [User]: alice [Age]: 61

#========================================================================================================================================#
| LATTE TELEMETRY [TIME][RAW]                                                                                                            |
#========================================================================================================================================#
| COMPONENT            |   SAMPLES |        AVG |     MEDIAN |    STD DEV |     SKEW |        MIN |        MAX |      RANGE |    OUTLIER |
|----------------------------------------------------------------------------------------------------------------------------------------|
| 2) Search            |         1 |     1.01 s |     1.01 s |    0.00 ns |     0.00 |     1.01 s |     1.01 s |    0.00 ns |          0 |
| 2.1) SIMD body       |         1 |     1.01 s |     1.01 s |    0.00 ns |     0.00 |     1.01 s |     1.01 s |    0.00 ns |          0 |
| 2.2) SIMD tail       |         1 |  570.00 ns |  570.00 ns |    0.00 ns |     0.00 |  570.00 ns |  570.00 ns |    0.00 ns |          0 |
| 1) Write             |         1 |     3.88 s |     3.88 s |    0.00 ns |     0.00 |     3.88 s |     3.88 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   87.49 ms |   87.49 ms |    0.00 ns |     0.00 |   87.49 ms |   87.49 ms |    0.00 ns |          0 |
| 0) Rng chars         |     65535 |   10.57 ns |    9.79 ns |    9.46 ns |     1.89 |    0.21 ns |  110.00 ns |  109.79 ns |          1 |
| 1.2) Write Loop      |     65527 |   53.14 ns |   50.00 ns |   10.32 ns |     6.69 |   40.00 ns |  270.00 ns |  230.00 ns |          9 |
| 3) Read              |         1 |   22.81 ms |   22.81 ms |    0.00 ns |     0.00 |   22.81 ms |   22.81 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   17.39 us |   17.39 us |    0.00 ns |     0.00 |   17.39 us |   17.39 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    5.00 ms |    2.57 ms |    3.85 ms |     0.70 |    1.99 ms |   10.43 ms |    8.44 ms |          0 |
| Global               |         1 |     5.04 s |     5.04 s |    0.00 ns |     0.00 |     5.04 s |     5.04 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 1.01 s
[Expected] RNtuple search take 16.81 ms per 1M iters 
Total Rows: 59.41 M
*/



/* --------------------------------------- Iterative memcmp for const size (single thread) O(n)
./bin/fastsearch_const
[19:07:02] Use Iterative const size to search the name
[Sys] Found 4 users named: alice
   Within a database made of 59406880 users
 [User]: alice [Age]: 45
 [User]: alice [Age]: 19
 [User]: alice [Age]: 30
 [User]: alice [Age]: 61

#========================================================================================================================================#
| LATTE TELEMETRY [TIME][RAW]                                                                                                            |
#========================================================================================================================================#
| COMPONENT            |   SAMPLES |        AVG |     MEDIAN |    STD DEV |     SKEW |        MIN |        MAX |      RANGE |    OUTLIER |
|----------------------------------------------------------------------------------------------------------------------------------------|
| 2) Search            |         1 |     1.20 s |     1.20 s |    0.00 ns |     0.00 |     1.20 s |     1.20 s |    0.00 ns |          0 |
| 1) Write             |         1 |     3.86 s |     3.86 s |    0.00 ns |     0.00 |     3.86 s |     3.86 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   88.50 ms |   88.50 ms |    0.00 ns |     0.00 |   88.50 ms |   88.50 ms |    0.00 ns |          0 |
| 0) Rng chars         |     65535 |   10.62 ns |    9.79 ns |    9.52 ns |     1.91 |    0.21 ns |  110.00 ns |  109.79 ns |          1 |
| 1.2) Write Loop      |     65519 |   53.24 ns |   50.00 ns |   10.54 ns |     6.10 |   40.00 ns |  210.00 ns |  170.00 ns |         17 |
| 3) Read              |         1 |   23.28 ms |   23.28 ms |    0.00 ns |     0.00 |   23.28 ms |   23.28 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   19.58 us |   19.58 us |    0.00 ns |     0.00 |   19.58 us |   19.58 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    5.12 ms |    2.68 ms |    3.97 ms |     0.69 |    1.95 ms |   10.72 ms |    8.77 ms |          0 |
| Global               |         1 |     5.22 s |     5.22 s |    0.00 ns |     0.00 |     5.22 s |     5.22 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 1.20 s
[Expected] RNtuple search take 20.21 ms per 1M iters 
Total Rows: 59.41 M
*/
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <TDictionary.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mm_malloc.h>
#include <ostream>
#include <random>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <immintrin.h>

#include "latte.hpp"

static constexpr uint64_t N = 26*26*26*26*26*5;
static constexpr int name_len = 5;

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)      __builtin_expect(!!(x), 1)
#define UNLIKELY(x)    __builtin_expect(!!(x), 0) 
#endif


static std::string RNG_String(std::mt19937& rng){
  static constexpr std::string_view chars = "abcdefghijklmnopqrstuvwxyz";
  std::uniform_int_distribution<int> dist(0, chars.size()-1);
  std::string name(name_len, ' ');
  for(auto&c:name){
    c=chars[dist(rng)];
    LATTE_PULSE("0) Rng chars");
  }
  return name;
}
static int RNG_int(std::mt19937& rng){
  std::uniform_int_distribution<int> dist(0, 100);
  return dist(rng);
}

static uint32_t fnv1a(const std::string& str){ // mystical function
  uint32_t hash = 2166136261u; // Fowler-Noll-Vo hash magic number (hex: 0x811C9DC5)
  for(unsigned char c : str){
    hash ^=c;
    hash *=0x01000193u; //Fowler-Noll-Vo prime magic number (hex: 0x01000193)
  }
  return hash;
}

static void saveIndex(const std::unordered_map<uint32_t, std::vector<uint64_t>>& index, std::string path) {
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

static std::unordered_map<uint32_t, std::vector<uint64_t>> loadIndex(std::string path) {
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
  auto name = model->MakeField<std::string>("name");
  auto hash_name = model->MakeField<uint32_t>("hash_name");
  auto age = model->MakeField<int>("age");


  ROOT::RNTupleWriteOptions opts;
  opts.SetCompression(401); // LZ4
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
  std::mt19937 rng(42);
  for(int i = 0; i<N; ++i){
    std::string user = RNG_String(rng);
    uint32_t huser= fnv1a(user);
    *name = user;
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


struct SearchResult{
  std::unique_ptr<ROOT::RNTupleReader> reader;
  std::vector<long unsigned> matches;
  std::string tName; // human readable name
};

//---------------------------------------------------------------------------------------------
SearchResult SIMD_FSL_Search(const std::string& tName){ 
  Latte::Mid::Start("2) Search");
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto vName = reader->GetView<std::string>("name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

  __m512i tar_firstB = _mm512_set1_epi8(tName.front());
  __m512i tar_secB   = _mm512_set1_epi8(tName[1]);
  __m512i tar_lastB  = _mm512_set1_epi8(tName.back());

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
      uint32_t j = __builtin_ctzll(mask);
      if(std::memcmp(tName.data(), vName(v+j).data(), tName.size())==0) matches.push_back(v+j);
      mask &= mask-1;
    }
  }
  
  Latte::Fast::Stop("2.1) SIMD body");
  Latte::Fast::Start("2.2) SIMD tail");
  
  for(; v<nEntries;++v){ // v tail (no bloom)
    if(std::memcmp(tName.data(), vName(v).data(), tName.size())==0) matches.push_back(v);
  }

  Latte::Fast::Stop("2.2) SIMD tail");
  Latte::Hard::Stop("2) Search"); 
  return {std::move(reader), std::move(matches), tName}; // (reader is unique_ptr)
}


//---------------------------------------------------------------------------------------------
SearchResult SIMD_fnv1a_Search(const std::string& tName){
  Latte::Mid::Start("2) Search");
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto vName = reader->GetView<std::string>("name");
  auto vHashName = reader->GetView<uint32_t>("hash_name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

  __m512i tar_hashed = _mm512_set1_epi32((int)fnv1a(tName));
  alignas(64) uint32_t hashed[16];

  Latte::Fast::Start("2.1) SIMD body");
  uint64_t v = 0;
  for(; v+64<=nEntries; v+=16){
    for(uint32_t j = 0; j < 16; ++j){ // loading v64
      const auto& h = vHashName(v+j);
      hashed[j] = static_cast<uint32_t>(h);
    }
    __m512i vcan_hashed = _mm512_load_si512(hashed); // last step of loading

    uint64_t mask = _mm512_cmpeq_epi32_mask(tar_hashed, vcan_hashed);

    while(mask){ // unroll mask to save bloomed
      uint32_t j = __builtin_ctzll(mask);
      if(std::memcmp(tName.data(), vName(v+j).data(), tName.size())==0) matches.push_back(v+j);
      mask &= mask-1;
    }
  }


  Latte::Fast::Stop("2.1) SIMD body");
  Latte::Fast::Start("2.2) SIMD tail");
  
  for(; v<nEntries;++v){ // v tail (no bloom)
    if(std::memcmp(tName.data(), vName(v).data(), tName.size())==0) matches.push_back(v);
  }


  Latte::Fast::Stop("2.2) SIMD tail");
  Latte::Hard::Stop("2) Search");
  return {std::move(reader), std::move(matches), tName}; //move to avoid destruct (reader is unique_ptr)
}


//---------------------------------------------------------------------------------------------
SearchResult O1Search(std::string tName) {
  Latte::Mid::Start("2) Search");
  Latte::Fast::Start("2.1) Search LoadIndex");
  auto index = loadIndex("./data/search/users.idx");
  Latte::Fast::Stop("2.1) Search LoadIndex");

  uint32_t target = fnv1a(tName);
  Latte::Fast::Start("2.2) Search find");
  auto& rows = index[target]; // O(1) hashmap hasname search

  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  Latte::Fast::Stop("2.2) Search find");
  Latte::Hard::Stop("2) Search");
  return {std::move(reader), std::move(rows), tName};
}

//---------------------------------------------------------------------------------------------
SearchResult ConstSearch(const std::string& tName) {
  Latte::Mid::Start("2) Search");
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::string>("name");
  std::vector<uint64_t> candidates;
  candidates.reserve(10);

  char xa[name_len];
  char xb[name_len];
  std::memcpy(&xa, tName.data(), name_len);
  // safe: size()+1 includes the guaranteed null terminator
  for (uint64_t i = 0; i < N; ++i) {
    auto iname = vName(i);
    std::memcpy(&xb, iname.data(), name_len);
    if (std::memcmp(xa, xb, name_len) == 0) candidates.push_back(i);
  }
  Latte::Hard::Stop("2) Search");
  return {std::move(reader), std::move(candidates), tName};
}

//---------------------------------------------------------------------------------------------

template <class T>
static inline void DoNotOptimize(const T& v) {
  asm volatile("" : : "r,m"(v) : "memory");
}

SearchResult NoSearch(const std::string& tName) {
  Latte::Mid::Start("2) Search");
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::string>("name");
  const auto nEntries = reader->GetNEntries();

  for (uint64_t i = 0; i < nEntries; ++i) {
    const auto& s = vName(i);
    DoNotOptimize(s.front()); // iterations cost 
  }

  Latte::Hard::Stop("2) Search");
  return {std::move(reader), std::vector<uint64_t>{}, tName};
}

//---------------------------------------------------------------------------------------------

void read(SearchResult& Sresult){
  Latte::Mid::Start("3) Read");
  Latte::Fast::Start("3.1) Read Init");
  auto vName = Sresult.reader->GetView<std::string>("name");
  auto vAge = Sresult.reader->GetView<int>("age");
  Latte::Fast::Stop("3.1) Read Init");
  std::cout << "[Sys] Found " << Sresult.matches.size() << " users named: " << Sresult.tName << "\n   Within a database made of " << Sresult.reader->GetNEntries() << " users" << std::endl;


  for(auto&idx:Sresult.matches){
    std::cout 
      << " [User]: " << vName(idx)
      << " [Age]: " << vAge(idx) << std::endl;
    LATTE_PULSE("3.2) Read Findings");
  }
  Latte::Hard::Stop("3) Read");
}


int main(){
#if RUN_O1SEARCH
  std::cout << "[" << __TIME__ << "] Use O(1) via unordered_map"  << std::endl;
#elif RUN_SIMDFSLSEARCH
  std::cout << "[" << __TIME__ << "] use Iterative AVX512 + FSL" << std::endl;
#elif RUN_SIMDFNV1ASEARCH
  std::cout << "[" << __TIME__ << "] use Iterative AVX512 + fnv1a" << std::endl;
#elif RUN_CONSTSEACH
  std::cout << "[" << __TIME__ << "] use Iterative const size"  << std::endl;
#elif RUN_NOSEARCH
  std::cout << "[" << __TIME__ << "] Baseline cost of iterating over RNTuple"  << std::endl;
#else 
  std::cout << "No Parameter search function given, Abort()" << std::endl;
  abort();
#endif


  std::string tName = "alice";
  Latte::Mid::Start("Global");
  write();


  SearchResult Sresult;
#if RUN_O1SEARCH
  Sresult = O1Search(tName); // academically fast O(1)
#elif RUN_SIMDFSLSEARCH
  Sresult = SIMD_FSL_Search(tName); // AVX512 with bloom first+second+last name char
#elif RUN_SIMDFNV1ASEARCH
  Sresult = SIMD_fnv1a_Search(tName); // AVX512 with bloom fnv1a hashing
#elif RUN_CONSTSEACH
  Sresult = ConstSearch(tName); // constant name size
#elif RUN_NOSEARCH
  Sresult = NoSearch(tName); // iterations cost, no search
#endif


  read(Sresult);
  Latte::Hard::Stop("Global");



  Latte::DumpToStream(std::cout, Latte::Parameter::Time, Latte::Parameter::Raw);
  auto snap_Search = Latte::Snapshot("2) Search")[0];
  double cycles_per_ns;
  LATTE_FREQ(cycles_per_ns);
  std::cout << "Searching took: " << Latte::FormatTime(snap_Search/cycles_per_ns) << std::endl;
  std::cout << "[Expected] RNtuple search take " << Latte::FormatTime((snap_Search/N*1'000'000)/4.7) << " per 1M iters " << std::endl;
  auto LargeFormat = [](double val) {
    const char* units[] = {"", "K", "M", "B", "T"};
    int unit_idx = 0;
    while (val >= 1000.0 && unit_idx < 4) { val /= 1000.0; unit_idx++; }
    std::ostringstream ss;
    if (unit_idx == 0) ss << std::fixed << std::setprecision(0) << val;
    else ss << std::fixed << std::setprecision(2) << val << " " << units[unit_idx];
    return ss.str();
  };

  std::cout << "Total Rows: " << LargeFormat(N);
  std::cout << std::endl;
}

