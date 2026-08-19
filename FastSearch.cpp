// Per 1M rows
// O(1):              load: 1.24 s || search: 18.24ms
// avx512 fnv1a:      4.89 ms  (   3 captures in bloom body, use GetDirectAccessView since hash is fixed size) 
// avx512 FSL:        10.43 ms (3275 captures in bloom body)
// constsize O(1):    11.06 ms 
// Binary Search:    212.77 us (25 iters)


// Raw Iterations Char[5]:    9.15 ms 
// Raw Iter Hash<uint32_t>:   4.47 ms
//PS: AVX512-based use mask+ctz, vectorized search doesnt iterate over all of rows
//Reason why fnv1a is faster than iterations w/out logic



// best optimization is compression at 401



//TODO:
//  -slice GPU decoding + unsync multithread for lightweight max speed
//  - name column stays std::array<char,5> inside RNTuple (fixed-size array column);
//    all search-side arrays are plain char[name_len]

//--------------------------------------------- Academically fast through unordered_map O(1) ---------------------
/*
./bin/fastsearch_o1
[10:39:50] Use O(1) via unordered_map
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
| 2.2) Search find     |         1 |  226.79 us |  226.79 us |    0.00 ns |     0.00 |  226.79 us |  226.79 us |    0.00 ns |          0 |
| 1) Write             |         1 |    28.03 s |    28.03 s |    0.00 ns |     0.00 |    28.03 s |    28.03 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   88.09 ms |   88.09 ms |    0.00 ns |     0.00 |   88.09 ms |   88.09 ms |    0.00 ns |          0 |
| 0) Rng chars         |     65535 |   92.61 ns |    9.79 ns |  176.18 ns |     2.26 |    0.21 ns |    3.82 us |    3.82 us |          1 |
| 1.2) Write Loop      |     65530 |  467.11 ns |  450.00 ns |  139.53 ns |     5.02 |   70.00 ns |    5.07 us |    5.00 us |          6 |
| 1.3) Write SaveIndex |         1 |     1.76 s |     1.76 s |    0.00 ns |     0.00 |     1.76 s |     1.76 s |    0.00 ns |          0 |
| 3) Read              |         1 |   23.76 ms |   23.76 ms |    0.00 ns |     0.00 |   23.76 ms |   23.76 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   20.56 us |   20.56 us |    0.00 ns |     0.00 |   20.56 us |   20.56 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    5.24 ms |    2.58 ms |    4.21 ms |     0.70 |    1.97 ms |   11.19 ms |    9.22 ms |          0 |
| Global               |         1 |    32.35 s |    32.35 s |    0.00 ns |     0.00 |    32.35 s |    32.35 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 1.20 s
Total Rows: 59.41 M

/* -------------------------------------------- Iterative AVX512 + bloom(fnv1a) (single thread)
./bin/fastsearch_fnv1a
[16:55:00] use Iterative AVX512 + fnv1a
Warning in <ROOT_TImplicitMT_DisableImplicitMT>: Implicit multi-threading is already disabled
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
| 2) Search            |         1 |  302.13 ms |  302.13 ms |    0.00 ns |     0.00 |  302.13 ms |  302.13 ms |    0.00 ns |          0 |
| 2.1) SIMD body       |         1 |  296.96 ms |  296.96 ms |    0.00 ns |     0.00 |  296.96 ms |  296.96 ms |    0.00 ns |          0 |
| 2.1.1) unroll bloom  |         3 |   67.19 ms |   72.35 ms |   51.31 ms |    -0.15 |    1.93 ms |  127.29 ms |  125.35 ms |          0 |
| 2.2) SIMD tail       |         1 |    0.21 ns |    0.21 ns |    0.00 ns |     0.00 |    0.21 ns |    0.21 ns |    0.00 ns |          0 |
| 1) Write             |         1 |     4.03 s |     4.03 s |    0.00 ns |     0.00 |     4.03 s |     4.03 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   88.71 ms |   88.71 ms |    0.00 ns |     0.00 |   88.71 ms |   88.71 ms |    0.00 ns |          0 |
| 1.2) Write Loop      |     65525 |   55.01 ns |   50.00 ns |   10.53 ns |     5.70 |   50.00 ns |  220.00 ns |  170.00 ns |         11 |
| 0) Rng chars         |     65535 |   10.98 ns |    9.79 ns |   10.05 ns |     1.75 |    0.21 ns |  120.00 ns |  119.79 ns |          1 |
| 3) Read              |         1 |   23.46 ms |   23.46 ms |    0.00 ns |     0.00 |   23.46 ms |   23.46 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   16.21 us |   16.21 us |    0.00 ns |     0.00 |   16.21 us |   16.21 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    5.29 ms |    2.94 ms |    4.03 ms |     0.68 |    1.96 ms |   10.97 ms |    9.01 ms |          0 |
| Global               |         1 |     4.48 s |     4.48 s |    0.00 ns |     0.00 |     4.48 s |     4.48 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 302.13 ms
[Expected] RNtuple search take 4.89 ms per 1M iters 
Total Rows: 59.41 M
*/




/* ---------------------------------------------------Iterative AVX512 + bloom(First, Second and Last char) (single thread)
./bin/fastsearch_fsl
[17:40:00] use Iterative AVX512 + FSL
Warning in <ROOT_TImplicitMT_DisableImplicitMT>: Implicit multi-threading is already disabled
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
| 2) Search            |         1 |  623.22 ms |  623.22 ms |    0.00 ns |     0.00 |  623.22 ms |  623.22 ms |    0.00 ns |          0 |
| 2.1) SIMD body       |         1 |  618.11 ms |  618.11 ms |    0.00 ns |     0.00 |  618.11 ms |  618.11 ms |    0.00 ns |          0 |
| 2.1.1) unroll bloom  |      3275 |  186.63 us |  130.95 us |  188.29 us |     2.16 |   10.00 ns |    1.80 ms |    1.80 ms |          0 |
| 2.2) SIMD tail       |         1 |   12.49 us |   12.49 us |    0.00 ns |     0.00 |   12.49 us |   12.49 us |    0.00 ns |          0 |
| 2.2.1) Tail          |        31 |   22.26 ns |   10.00 ns |   51.22 ns |     5.13 |    0.21 ns |  300.00 ns |  299.79 ns |          0 |
| 3) Read              |         1 |   25.90 ms |   25.90 ms |    0.00 ns |     0.00 |   25.90 ms |   25.90 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   27.25 us |   27.25 us |    0.00 ns |     0.00 |   27.25 us |   27.25 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    5.49 ms |    5.13 ms |    4.64 ms |     0.12 |    1.72 us |   11.34 ms |   11.34 ms |          0 |
| 1) Write             |         1 |     3.63 s |     3.63 s |    0.00 ns |     0.00 |     3.63 s |     3.63 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   87.18 ms |   87.18 ms |    0.00 ns |     0.00 |   87.18 ms |   87.18 ms |    0.00 ns |          0 |
| 1.2) Write Loop      |     65534 |   51.46 ns |   50.00 ns |   33.83 ns |    54.37 |   40.00 ns |    3.68 us |    3.64 us |          2 |
| 0) Rng chars         |     65522 |   10.22 ns |    9.79 ns |    9.53 ns |     8.89 |    0.21 ns |  710.21 ns |  710.00 ns |         14 |
| Global               |         1 |     4.42 s |     4.42 s |    0.00 ns |     0.00 |     4.42 s |     4.42 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 623.22 ms
[Expected] RNtuple search take 10.43 ms per 1M iters 
Total Rows: 59.41 M
*/



/* --------------------------------------- Iterative memcmp for const size (single thread) O(n)
./bin/fastsearch_const
[17:40:48] use Iterative const size
Warning in <ROOT_TImplicitMT_DisableImplicitMT>: Implicit multi-threading is already disabled
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
| 2) Search            |         1 |  660.78 ms |  660.78 ms |    0.00 ns |     0.00 |  660.78 ms |  660.78 ms |    0.00 ns |          0 |
| 3) Read              |         1 |   25.49 ms |   25.49 ms |    0.00 ns |     0.00 |   25.49 ms |   25.49 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   26.63 us |   26.63 us |    0.00 ns |     0.00 |   26.63 us |   26.63 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    5.31 ms |    4.94 ms |    4.50 ms |     0.12 |    1.79 us |   11.00 ms |   11.00 ms |          0 |
| 1) Write             |         1 |     3.70 s |     3.70 s |    0.00 ns |     0.00 |     3.70 s |     3.70 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   87.39 ms |   87.39 ms |    0.00 ns |     0.00 |   87.39 ms |   87.39 ms |    0.00 ns |          0 |
| 1.2) Write Loop      |     65534 |   51.56 ns |   50.00 ns |   31.64 ns |    50.58 |   40.00 ns |    3.28 us |    3.24 us |          2 |
| 0) Rng chars         |     65522 |   10.24 ns |    9.79 ns |    9.31 ns |     4.13 |    0.21 ns |  410.00 ns |  409.79 ns |         14 |
| Global               |         1 |     4.52 s |     4.52 s |    0.00 ns |     0.00 |     4.52 s |     4.52 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 660.78 ms
[Expected] RNtuple search take 11.06 ms per 1M iters 
Total Rows: 59.41 M
*/

/*
./bin/fastsearch_bin
[14:51:33] Binary Search
Warning in <ROOT_TImplicitMT_DisableImplicitMT>: Implicit multi-threading is already disabled
[Sys] Found 4 users named: alice
   Within a database made of 59406880 users
 [User]: alice [Age]: 45
 [User]: alice [Age]: 30
 [User]: alice [Age]: 61
 [User]: alice [Age]: 19

#========================================================================================================================================#
| LATTE TELEMETRY [TIME][RAW]                                                                                                            |
#========================================================================================================================================#
| COMPONENT            |   SAMPLES |        AVG |     MEDIAN |    STD DEV |     SKEW |        MIN |        MAX |      RANGE |    OUTLIER |
|----------------------------------------------------------------------------------------------------------------------------------------|
| 2) Search            |         1 |   24.44 ms |   24.44 ms |    0.00 ns |     0.00 |   24.44 ms |   24.44 ms |    0.00 ns |          0 |
| 3) Read              |         1 |    4.74 ms |    4.74 ms |    0.00 ns |     0.00 |    4.74 ms |    4.74 ms |    0.00 ns |          0 |
| 3.1) Read Init       |         1 |   16.67 us |   16.67 us |    0.00 ns |     0.00 |   16.67 us |   16.67 us |    0.00 ns |          0 |
| 3.2) Read Findings   |         3 |    1.98 us |    2.13 us |  557.70 ns |    -0.39 |    1.23 us |    2.57 us |    1.34 us |          0 |
| 1) Write             |         1 |     3.71 s |     3.71 s |    0.00 ns |     0.00 |     3.71 s |     3.71 s |    0.00 ns |          0 |
| 1.1) Write init      |         1 |   78.35 ms |   78.35 ms |    0.00 ns |     0.00 |   78.35 ms |   78.35 ms |    0.00 ns |          0 |
| 1.2) Write Loop      |     65531 |   51.75 ns |   50.00 ns |   32.43 ns |    50.95 |   40.00 ns |    3.41 us |    3.37 us |          5 |
| 0) Rng chars         |     65522 |   10.22 ns |    9.79 ns |    9.20 ns |     4.51 |    0.21 ns |  440.21 ns |  440.00 ns |         14 |
| Sort RNTuple         |         1 |     7.03 s |     7.03 s |    0.00 ns |     0.00 |     7.03 s |     7.03 s |    0.00 ns |          0 |
| Reading              |         1 |     2.15 s |     2.15 s |    0.00 ns |     0.00 |     2.15 s |     2.15 s |    0.00 ns |          0 |
| Sorting              |         1 |     3.25 s |     3.25 s |    0.00 ns |     0.00 |     3.25 s |     3.25 s |    0.00 ns |          0 |
| Writing sorted data  |         1 |     1.63 s |     1.63 s |    0.00 ns |     0.00 |     1.63 s |     1.63 s |    0.00 ns |          0 |
| 2.1) Body            |         1 |   18.81 ms |   18.81 ms |    0.00 ns |     0.00 |   18.81 ms |   18.81 ms |    0.00 ns |          0 |
| 2.1.1) BODY LOOP     |        25 |  452.05 us |   10.00 ns |  603.57 us |     0.59 |    0.21 ns |    1.40 ms |    1.40 ms |          0 |
| 2.2) tail            |         1 |   70.00 ns |   70.00 ns |    0.00 ns |     0.00 |   70.00 ns |   70.00 ns |    0.00 ns |          0 |
| 2.3) Deblooming      |         1 |  534.16 us |  534.16 us |    0.00 ns |     0.00 |  534.16 us |  534.16 us |    0.00 ns |          0 |
| Global               |         1 |    10.97 s |    10.97 s |    0.00 ns |     0.00 |    10.97 s |    10.97 s |    0.00 ns |          0 |
#========================================================================================================================================#
Searching took: 24.44 ms
[Expected] RNtuple search take 212.77 us per 1M iters 
Total Rows: 59.41 M
*/


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

== NVIDIA GPU ==
  0, NVIDIA GeForce RTX 4070 Ti, Enabled, 2805 MHz, 285.00 W

== AMD GPU ==
  /sys/class/drm/card0/device/power_dpm_force_performance_level high

*/
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <TDictionary.h>
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
#include <TROOT.h>
#include <ROOT/RDataFrame.hxx>

#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include "latte.hpp"


static constexpr uint64_t N = 26*26*26*26*26*5;
static constexpr int name_len = 5;

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)      __builtin_expect(!!(x), 1)
#define UNLIKELY(x)    __builtin_expect(!!(x), 0) 
#endif


static void RNG_String(std::mt19937& rng, char (&name)[name_len]) {
  static constexpr std::string_view chars = "abcdefghijklmnopqrstuvwxyz";
  std::uniform_int_distribution<std::size_t> dist(0, chars.size() - 1);
  for (char& c : name) {
    c = chars[dist(rng)];
    LATTE_PULSE("0) Rng chars");
  }
}


static auto RNG_int(std::mt19937& rng) -> int{
  std::uniform_int_distribution<int> dist(0, 100);
  return dist(rng);
}

static auto fnv1a(const char* str, std::size_t len) -> uint32_t { // mystical function
  uint32_t hash = 2166136261u; // Fowler-Noll-Vo hash magic number (hex: 0x811C9DC5)
  for (std::size_t i = 0; i < len; ++i) {
    hash ^= static_cast<unsigned char>(str[i]);
    hash *= 0x01000193u; //Fowler-Noll-Vo prime magic number (hex: 0x01000193)
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

  auto viewName = reader->GetView<std::array<char, name_len>>("name");
  auto viewHashName = reader->GetView<uint32_t>("hash_name");
  auto viewAge = reader->GetView<int>("age");

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
      viewHashName(i),
      viewName(i),
      viewAge(i)
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
  opts.SetCompression(401);  // LZ4
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
  Latte::Mid::Start("2) Search");
  ROOT::DisableImplicitMT();
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto vName = reader->GetView<std::array<char, name_len>>("name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

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
  Latte::Mid::Start("2) Search");
  ROOT::DisableImplicitMT();
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto vName = reader->GetView<std::array<char, name_len>>("name");
  auto vhashView = reader->GetDirectAccessView<uint32_t>("hash_name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

  __m512i tar_hashed = _mm512_set1_epi32(static_cast<int>(fnv1a(tName, name_len)));
  alignas(64) uint32_t hashed[16];

  Latte::Fast::Start("2.1) SIMD body");
  uint64_t v = 0;
  for(; v+16<=nEntries; v+=16){
    for(uint32_t j = 0; j < 16; ++j){ // loading v64
      const auto& h = vhashView(v+j);
      hashed[j] = static_cast<uint32_t>(h);
    }
    __m512i vcan_hashed = _mm512_load_si512(hashed); // last step of loading

    uint64_t mask = _mm512_cmpeq_epi32_mask(tar_hashed, vcan_hashed);

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
  Latte::Mid::Start("2) Search");
  Latte::Fast::Start("2.1) Search LoadIndex");
  auto index = loadIndex("./data/search/users.idx");
  Latte::Fast::Stop("2.1) Search LoadIndex");

  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  uint32_t target = fnv1a(tName, name_len);
  Latte::Fast::Start("2.2) Search find");
  std::vector<uint64_t> matches;
  auto it = index.find(target); // O(1) hashmap lookup
  if (it != index.end()) {
    // fnv1a is 32-bit: different names can share a bucket, so confirm with memcmp.
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
  Latte::Mid::Start("2) Search");
  ROOT::DisableImplicitMT();
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::array<char, name_len>>("name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> candidates;
  candidates.reserve(10);

  char xa[name_len];
  char xb[name_len];
  std::memcpy(xa, tName, name_len);
  for (uint64_t i = 0; i < nEntries; ++i) {
    const auto& iname = vName(i);
    std::memcpy(xb, iname.data(), name_len);
    if (std::memcmp(xa, xb, name_len) == 0) candidates.push_back(i);
  }
  Latte::Hard::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(candidates), tName);
}

//---------------------------------------------------------------------------------------------

auto BinarySearch(const char (&tName)[name_len]) -> SearchResult {
  sortAndSaveRNTuple();
  Latte::Mid::Start("2) Search");
  ROOT::DisableImplicitMT();
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vHName = reader->GetDirectAccessView<uint32_t>("hash_name");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  const uint64_t nEntries = reader->GetNEntries();

  std::vector<uint64_t> candidates;
  candidates.reserve(10);
  if (nEntries == 0) return SearchResult(std::move(reader), std::move(candidates), tName);

  const uint32_t key = fnv1a(tName, name_len);

  Latte::Mid::Start("2.1) Body");
  uint64_t base = 0; // base idx
  uint64_t len  = nEntries;
  while (len > 1) {
    const uint64_t half = len >> 1; 
    const uint32_t value = vHName(base + half - 1);
    base += half & (uint64_t)(-(int64_t)(value < key)); // 0,1 -> 0,-1 -> 0,0xFF.. -> base+=0, base+=half
    len -= half;
    //LATTE_PULSE("2.1.1) BODY LOOP");
  } base += (uint64_t)(vHName(base) < key);
  /*
   idx=0
   next= n>>1
   while(next)
    idx += next
    next &= 0x7FFFF (0x7: 0111, it set signbit=0) 
    cmp = memcmp(...)
    next = ((cmp>0)-(cmp<0) * (next>>1)) //sign * next/2
  */
  Latte::Mid::Stop("2.1) Body");

  Latte::Mid::Start("2.2) tail");
  for (uint64_t i = base; i < nEntries && vHName(i) == key; ++i) candidates.push_back(i);
  Latte::Mid::Stop("2.2) tail");

  Latte::Fast::Start("2.3) Deblooming");
  for (size_t i = 0; i < candidates.size(); ) {
    if (std::memcmp(tName, vName(candidates[i]).data(), name_len) != 0) candidates.erase(candidates.begin()+i);
    else ++i;
  }
  Latte::Fast::Stop("2.3) Deblooming");
  Latte::Mid::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(candidates), tName);
}

//---------------------------------------------------------------------------------------------

template <class T>
static inline void DoNotOptimize(const T& v) {
  asm volatile("" : : "r,m"(v) : "memory");
}

auto NoSearch(const char (&tName)[name_len]) -> SearchResult {
  Latte::Mid::Start("2) Search");
  ROOT::DisableImplicitMT();
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::array<char, name_len>>("name");
  const auto nEntries = reader->GetNEntries();

  for (uint64_t i = 0; i < nEntries; ++i) {
    const auto& s = vName(i);
    DoNotOptimize(s.front()); // iterations cost 
  }

  Latte::Hard::Stop("2) Search");
  return SearchResult(std::move(reader), std::vector<uint64_t>{}, tName);
}


auto NoSearchHash(const char (&tName)[name_len]) -> SearchResult {
  Latte::Mid::Start("2) Search");
  ROOT::DisableImplicitMT();
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vhashView = reader->GetDirectAccessView<uint32_t>("hash_name");
  const auto nEntries = reader->GetNEntries();

  for (uint64_t i = 0; i < nEntries; ++i) {
    const auto& s = vhashView(i);
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
    std::cout << " [User]: ";
    std::cout.write(nm.data(), nm.size());
    std::cout << " [Age]: " << vAge(idx) << '\n';
    LATTE_PULSE("3.2) Read Findings");
  }
  Latte::Hard::Stop("3) Read");
}


auto main() -> int{
#if RUN_O1SEARCH
  std::cout << "[" << __TIME__ << "] Use O(1) via unordered_map"  << std::endl;
#elif RUN_SIMDFSLSEARCH
  std::cout << "[" << __TIME__ << "] Iterative AVX512 + FSL" << std::endl;
#elif RUN_SIMDFNV1ASEARCH
  std::cout << "[" << __TIME__ << "] Iterative AVX512 + fnv1a" << std::endl;
#elif RUN_CONSTSEARCH
  std::cout << "[" << __TIME__ << "] Iterative const size"  << std::endl;
#elif RUN_BINARYSEARCH
  std::cout << "[" << __TIME__ << "] Binary Search"  << std::endl;
#elif RUN_NOSEARCH
  std::cout << "[" << __TIME__ << "] Baseline cost of iterating over RNTuple"  << std::endl;
#elif RUN_NOSEARCHHash
  std::cout << "[" << __TIME__ << "] Baseline cost of iterating over RNTuple with hash"  << std::endl;
#else 
  std::cout << "No Parameter search function given, Abort()" << '\n';
  abort();
#endif


  char tName[name_len] = {'a', 'l', 'i', 'c', 'e'};
  Latte::Mid::Start("Global");
  write();


  SearchResult Sresult;
#if RUN_O1SEARCH
  Sresult = O1Search(tName); // academically fast O(1)
#elif RUN_SIMDFSLSEARCH
  Sresult = SIMD_FSL_Search(tName); // AVX512 with bloom first+second+last name char
#elif RUN_SIMDFNV1ASEARCH
  Sresult = SIMD_fnv1a_Search(tName); // AVX512 with bloom fnv1a hashing
#elif RUN_CONSTSEARCH
  Sresult = ConstSearch(tName); // constant name size
#elif RUN_BINARYSEARCH
  Sresult = BinarySearch(tName); // Log2(N) search
#elif RUN_NOSEARCH
  Sresult = NoSearch(tName); // iterations cost, no search
#elif RUN_NOSEARCHHASH
  Sresult = NoSearchHash(tName); // iterations cost with fixed size uint32_t, no search
#endif


  read(Sresult);
  Latte::Hard::Stop("Global");



  Latte::DumpToStream(std::cout, Latte::Parameter::Time, Latte::Parameter::Raw);
  auto snap_Search = Latte::Snapshot("2) Search")[0];
  double cycles_per_ns;
  LATTE_FREQ(cycles_per_ns);
  std::cout << "Searching took: " << Latte::FormatTime(snap_Search/cycles_per_ns) << '\n';
  std::cout << "[Expected] RNtuple search take " << Latte::FormatTime((snap_Search/N*1'000'000)/4.7) << " per 1M iters " << '\n';
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
}

