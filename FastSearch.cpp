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
#include <immintrin.h>
#include <likwid-marker.h>
#include <stdlib.h>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RNTupleRange.hxx>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

#include "latte.hpp"
#include "include/treeternary.hpp"

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


auto main() -> int {
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

