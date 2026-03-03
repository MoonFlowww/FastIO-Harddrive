#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <TDictionary.h>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <vector>
#include <immintrin.h>

#include "latte.hpp"

static constexpr long long unsigned N = 26*26*26*26*26*5;
static constexpr int name_len = 5;

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)      __builtin_expect(!!(x), 1)
#define UNLIKELY(x)    __builtin_expect(!!(x), 0) // Used in search
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
  opts.SetCompression(101); // LZ4
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


SearchResult Search(std::string tName){ //@param tName target name
  Latte::Mid::Start("2) Search");
  Latte::Mid::Start("2.1) Search Init");
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  const auto nEntries = reader->GetNEntries();
  const int nThreads = std::thread::hardware_concurrency();
  const auto chunkSize = nEntries/nThreads;

  std::vector<std::vector<int>> threadMatches(nThreads); //Local data 
  std::vector<std::thread> threads;


  uint32_t tHashName = fnv1a(tName); // Translate string to uint32_t (prestep vectorization)
  Latte::Hard::Stop("2.1) Search Init");
  Latte::Mid::Start("2.2) Search Loop");
  for(int t = 0; t<nThreads; ++t){
    threads.emplace_back([&, t](){
      auto reader = ROOT::RNTupleReader::Open("Users", "users.root");
      auto vName = reader->GetView<std::string>("name");
      auto vHashName = reader->GetView<uint32_t>("hash_name");
      auto vAge = reader->GetView<int>("age");

      __m512i targetVec = _mm512_set1_epi32(tHashName); // setup HashName Vectorization ZMM


      auto start = t * chunkSize;
      auto end = (t==nThreads-1) ? nEntries : start + chunkSize; // if end to_end else chunkSize
      std::vector<uint32_t> hashes(end-start); // handle last batch close to N

      int n = 0;
      for(uint64_t i = start; i<end; ++i){ // load HashName into vector for vectorization
        hashes[i-start] = vHashName(i);
      }

      std::vector<int> candidates;
      uint64_t i = start; // sync approach
      for(; i+16<=end; i+=16){ //create batch of 15 AVX512
        __m512i chunk = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(&hashes[i-start]) //hashes= list of vHashName
        );

        __mmask16 mask = _mm512_cmpeq_epi32_mask(chunk, targetVec); //chunk(16 vName in hash) == TargetVec(tName)

        while(mask){
          int bit = __builtin_ctz(mask); // number of trailling 0 bits at tail of x
          if(vName(i+bit)==tName) candidates.push_back(i+bit);
          mask&=mask-1; // clear lowest set bit
        }
        LATTE_PULSE("2.4) Search AVX512");
      }
      for(; i<end;++i){ // if tail is (<16 && >0)
        if(hashes[i-start]==tHashName && vName(i) == tName) candidates.push_back(i);
      }
      threadMatches[t] = std::move(candidates); // save candidates to thread space
    });
    LATTE_PULSE("2.3) Search Thread");
  } for(auto& th:threads) th.join();
  std::vector<long unsigned> matches;
  for(auto&v:threadMatches)
  matches.insert(matches.end(), v.begin(), v.end());

  Latte::Hard::Stop("2) Search"); // join matches is a part of the process, keep it in measurements
  return {std::move(reader), std::move(matches), tName}; //move to avoid destruct (reader is unique_ptr)
}

SearchResult FastSearch(std::string tName) {
  Latte::Mid::Start("2) Search");
  Latte::Fast::Start("2.1) Search LoadIndex");
  auto index = loadIndex("./data/search/users.idx");
  Latte::Fast::Stop("2.1) Search LoadIndex");

  uint32_t target = fnv1a(tName);
  Latte::Fast::Start("2.2) Search find");
  auto& rows = index[target]; // O(1) hashmap hasname search

  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::string>("name");
  auto vAge = reader->GetView<int>("age");
  Latte::Fast::Stop("2.2) Search find");
  Latte::Hard::Stop("Search O(1)");
  return {std::move(reader), std::move(rows), tName};
}


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
  std::cout << "[" << __TIME__ << "] Use " << "O(1) search via unordered_map" << " to search the name" << std::endl;
  #else
  std::cout << "[" << __TIME__ << "] Use " << "Iterative method with MaxThread + AVX512" << " to search the name" << std::endl;
  #endif
  std::string tName = "alice";
  Latte::Mid::Start("Global");
  write();
#if RUN_O1SEARCH
  auto Sresult = FastSearch(tName); // use secondary file with unordered_map for O(1) search || Latte Self-record;
#else
  auto Sresult = Search(tName);// MaxThreaded + AVX512
#endif
  read(Sresult);
  Latte::Hard::Stop("Global");



  Latte::DumpToStream(std::cout, Latte::Parameter::Time, Latte::Parameter::Raw);
  auto snap_Search = Latte::Snapshot("2) Search")[0];
  std::cout << "Searching took: " << Latte::FormatTime(snap_Search/4.7) << std::endl; // i use 4.7ghz 
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
  #if RUN_O1SEARCH
  #else
    int nT = std::thread::hardware_concurrency();
  std::cout 
    << "\n  Rows per Thread: " << LargeFormat(N/nT) 
    << "\n    AVX512 per thread: " << LargeFormat((N/nT)/16);
  #endif
  std::cout << std::endl;
}

// Min result over 5 runs (Outliers only exist in +side due to CPU interupts)
// [Raw]             ~30ms per 1m rows single column search
// [Thread]          ~ 7ms per 1m rows single column search
// [Thread & AVX512] ~ 5ms per 1m rows single column search


//TODO:
//  -switch raw names to std::array<char, 4>
//  -slice GPU decoding + unsync multithread for lightweight max speed

//---------------------------------------------Use unordered_map to make it academically fast O(1) (slower in time)---------------------
// 󰣇 codebase/cpp/Aladata ❯ g++ -std=c++20 -O3 -march=native -DRUN_O1SEARCH=1 find.cpp $(root-config --cflags --libs) && ./a.out                                                                                               14:41 
// [14:41:07] Use O(1) search via unordered_map to search the name
// [Sys] Found 4 users named: alice
//    Within a database made of 59406880 users
//  [User]: alice [Age]: 45
//  [User]: alice [Age]: 19
//  [User]: alice [Age]: 30
//  [User]: alice [Age]: 61
//
// #========================================================================================================================================#
// | LATTE TELEMETRY [TIME][RAW]                                                                                                            |
// #========================================================================================================================================#
// | COMPONENT            |   SAMPLES |        AVG |     MEDIAN |    STD DEV |     SKEW |        MIN |        MAX |      RANGE |     BYPASS |
// |----------------------------------------------------------------------------------------------------------------------------------------|
// | 2) Search            |         1 |     1.33 s |     1.33 s |    0.00 ns |     0.00 |     1.33 s |     1.33 s |    0.00 ns |          0 |
// | 2.1) Search LoadInde |         1 |     1.33 s |     1.33 s |    0.00 ns |     0.00 |     1.33 s |     1.33 s |    0.00 ns |          0 |
// | 2.2) Search find     |         1 |  361.41 us |  361.41 us |    0.00 ns |     0.00 |  361.41 us |  361.41 us |    0.00 ns |          0 |
// | 3) Read              |         1 |   29.34 ms |   29.34 ms |    0.00 ns |     0.00 |   29.34 ms |   29.34 ms |    0.00 ns |          0 |
// | 3.1) Read Init       |         1 |   23.10 us |   23.10 us |    0.00 ns |     0.00 |   23.10 us |   23.10 us |    0.00 ns |          0 |
// | 3.2) Read Findings   |         3 |    5.58 ms |    6.25 ms |    1.82 ms |    -0.50 |    3.09 ms |    7.39 ms |    4.30 ms |          0 |
// | 1) Write             |         1 |    40.15 s |    40.15 s |    0.00 ns |     0.00 |    40.15 s |    40.15 s |    0.00 ns |          0 |
// | 1.1) Write init      |         1 |  114.12 ms |  114.12 ms |    0.00 ns |     0.00 |  114.12 ms |  114.12 ms |    0.00 ns |          0 |
// | 0) Rng chars         |     65533 |  105.42 ns |    9.81 ns |  204.66 ns |     2.05 |    0.21 ns |    2.37 us |    2.37 us |          3 |
// | 1.2) Write Loop      |     65532 |  530.47 ns |  510.96 ns |  170.56 ns |     3.83 |   70.13 ns |    5.11 us |    5.04 us |          4 |
// | 1.3) Write SaveIndex |         1 |     2.12 s |     2.12 s |    0.00 ns |     0.00 |     2.12 s |     2.12 s |    0.00 ns |          0 |
// | Global               |         1 |    45.43 s |    45.43 s |    0.00 ns |     0.00 |    45.43 s |    45.43 s |    0.00 ns |          0 |
// #========================================================================================================================================#
// Searching took: 1.33 s
// [Expected] RNtuple search take 22.34 ms per 1M iters 
// Total Rows: 59.41 M


//------------------------------------------Iterative search MaxThread+AVX O(n) (Fast)
// 󰣇 codebase/cpp/Aladata ❯ g++ -std=c++20 -O3 -march=native -DRUN_O1SEARCH=0 find.cpp $(root-config --cflags --libs) && ./a.out                                                                                               14:45 
// [14:45:18] Use Iterative method with MaxThread + AVX512 to search the name
// [Sys] Found 4 users named: alice
//    Within a database made of 59406880 users
//  [User]: alice [Age]: 45
//  [User]: alice [Age]: 19
//  [User]: alice [Age]: 30
//  [User]: alice [Age]: 61
//
// #========================================================================================================================================#
// | LATTE TELEMETRY [TIME][RAW]                                                                                                            |
// #========================================================================================================================================#
// | COMPONENT            |   SAMPLES |        AVG |     MEDIAN |    STD DEV |     SKEW |        MIN |        MAX |      RANGE |     BYPASS |
// |----------------------------------------------------------------------------------------------------------------------------------------|
// | 2) Search            |         1 |  376.86 ms |  376.86 ms |    0.00 ns |     0.00 |  376.86 ms |  376.86 ms |    0.00 ns |          0 |
// | 2.1) Search Init     |         1 |  341.52 us |  341.52 us |    0.00 ns |     0.00 |  341.52 us |  341.52 us |    0.00 ns |          0 |
// | 2.2) Search Loop     |         1 |  347.23 ms |  347.23 ms |    0.00 ns |     0.00 |  347.23 ms |  347.23 ms |    0.00 ns |          0 |
// | 2.3) Search Thread   |        11 |   56.90 us |   48.41 us |   21.48 us |     0.36 |   26.91 us |   96.44 us |   69.53 us |          0 |
// | 3) Read              |         1 |   29.23 ms |   29.23 ms |    0.00 ns |     0.00 |   29.23 ms |   29.23 ms |    0.00 ns |          0 |
// | 3.1) Read Init       |         1 |   19.91 us |   19.91 us |    0.00 ns |     0.00 |   19.91 us |   19.91 us |    0.00 ns |          0 |
// | 3.2) Read Findings   |         3 |    5.71 ms |    6.01 ms |    2.01 ms |    -0.22 |    3.11 ms |    8.00 ms |    4.89 ms |          0 |
// | 2.4) Search AVX512   |    786427 |   10.33 ns |    9.81 ns |   28.84 ns |    12.89 |    0.21 ns |    1.50 us |    1.50 us |          5 |
// | 1) Write             |         1 |    11.34 s |    11.34 s |    0.00 ns |     0.00 |    11.34 s |    11.34 s |    0.00 ns |          0 |
// | 1.1) Write init      |         1 |  113.18 ms |  113.18 ms |    0.00 ns |     0.00 |  113.18 ms |  113.18 ms |    0.00 ns |          0 |
// | 0) Rng chars         |     65526 |   10.47 ns |    9.81 ns |    9.12 ns |     1.27 |    0.21 ns |   70.35 ns |   70.13 ns |         10 |
// | 1.2) Write Loop      |     65520 |   52.19 ns |   50.10 ns |    8.10 ns |     6.63 |   40.08 ns |  200.38 ns |  160.30 ns |         16 |
// #========================================================================================================================================#
// Searching took: 376.15 ms
// [Expected] RNtuple search take 6.17 ms per 1M iters 
// Total Rows: 59.41 M
//   Rows per Thread: 4.95 M
//     AVX512 per thread: 309.41 K
