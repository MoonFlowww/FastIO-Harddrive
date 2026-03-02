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
    LATTE_PULSE("rng chars");
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

void write(){ Latte::Fast::Start("Write");
  auto model = ROOT::RNTupleModel::Create();
  auto name = model->MakeField<std::string>("name");
  auto hash_name = model->MakeField<uint32_t>("hash_name");
  auto age = model->MakeField<int>("age");


  ROOT::RNTupleWriteOptions opts;
  opts.SetCompression(101); // LZ4
  auto writer = ROOT::RNTupleWriter::Recreate(
    std::move(model), "Users", "./data/search/users.root", opts
  );

  //std::unordered_map<uint32_t, std::vector<uint64_t>> index; //O(1) search if you know what will be searched
  std::mt19937 rng(42);
  for(int i = 0; i<N; ++i){
    std::string user = RNG_String(rng);
    uint32_t huser= fnv1a(user);
    *name = user;
    *hash_name = huser;
    *age = RNG_int(rng);
    writer->Fill();
    //index[huser].push_back(i);
    LATTE_PULSE("Write Loop");
  }
  //Latte::Fast::Start("SaveIndex");
  //saveIndex(index, "./data/search/users.idx");
  //Latte::Fast::Stop("SaveIndex");
  Latte::Fast::Stop("Write");
}


struct SearchResult{
  std::unique_ptr<ROOT::RNTupleReader> reader;
  std::vector<long unsigned> matches;
  std::string tName; // human readable name
};


SearchResult Search(std::string tName){ //@param tName target name
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  const auto nEntries = reader->GetNEntries();
  const int nThreads = std::thread::hardware_concurrency();
  const auto chunkSize = nEntries/nThreads;

  std::vector<std::vector<int>> threadMatches(nThreads); //Local data 
  std::vector<std::thread> threads;


  uint32_t tHashName = fnv1a(tName); // Translate string to uint32_t (prestep vectorization)

  Latte::Mid::Start("Search Iteration");
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
      }
      for(; i<end;++i){ // if tail is (<16 && >0)
        if(hashes[i-start]==tHashName && vName(i) == tName) candidates.push_back(i);
      }
      threadMatches[t] = std::move(candidates); // save candidates to thread space
    });
  } for(auto& th:threads) th.join();

  std::vector<long unsigned> matches;
  for(auto&v:threadMatches)
  matches.insert(matches.end(), v.begin(), v.end());

  Latte::Hard::Stop("Search Iteration"); // regroup in matches is a part of the process, keep it in measurements
  return {std::move(reader), std::move(matches), tName}; //move to avoid destruct (reader is unique_ptr)
}

SearchResult FastSearch(std::string tName) {
  Latte::Fast::Start("LoadIndex");
  auto index = loadIndex("./data/search/users.idx");
  Latte::Fast::Stop("LoadIndex");
  Latte::Mid::Start("Search O(1)");
  uint32_t target = fnv1a(tName);

  auto& rows = index[target]; // O(1) hashmap hasname search

  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::string>("name");
  auto vAge = reader->GetView<int>("age");

  Latte::Hard::Stop("Search O(1)");
  return {std::move(reader), std::move(rows), tName};
}


void read(SearchResult& Sresult){
  auto vName = Sresult.reader->GetView<std::string>("name");
  auto vAge = Sresult.reader->GetView<int>("age");
  std::cout << "[Sys] Found " << Sresult.matches.size() << " users named: " << Sresult.tName << "\n   Within a database made of " << Sresult.reader->GetNEntries() << " users" << std::endl;


  for(auto&idx:Sresult.matches){
    std::cout 
      << " [User]: " << vName(idx)
      << " [Age]: " << vAge(idx) << std::endl;
    LATTE_PULSE("Findings");
  }
}


int main(){
  std::string tName = "alice";
  Latte::Mid::Start("Global_full_search");
  write();
  auto Sresult = Search(tName); // MaxThreaded + AVX512
  // auto Sresult = FastSearch(tName); // use secondary file with unordered_map for O(1) search || Latte Self-record;
  read(Sresult);
  Latte::Hard::Stop("Global_full_search");



  Latte::DumpToStream(std::cout, Latte::Parameter::Time, Latte::Parameter::Raw);
  auto snap_Search = Latte::Snapshot("Search Iteration")[0];
  std::cout << "Searching took: " << Latte::FormatTime(snap_Search/4.7) << std::endl; // i use 4.7ghz 
  std::cout << "[Expected] RNtuple take " << Latte::FormatTime((snap_Search/N*1'000'000)/4.7) << " per 1M iters" << std::endl;

  auto LargeFormat = [](double val) {
    const char* units[] = {"", "K", "M", "B", "T"};
    int unit_idx = 0;
    while (val >= 1000.0 && unit_idx < 4) { val /= 1000.0; unit_idx++; }
    std::ostringstream ss;
    if (unit_idx == 0) ss << std::fixed << std::setprecision(0) << val;
    else ss << std::fixed << std::setprecision(2) << val << " " << units[unit_idx];
    return ss.str();
  };
  int nT = std::thread::hardware_concurrency();
  std::cout 
    << "Total rows: " << LargeFormat(N)
    << "\n  Rows per Thread: " << LargeFormat(N/nT) 
    << "\n    AVX512 per thread: " << LargeFormat((N/nT)/16) << std::endl;

}


// [Raw]             ~30ms per 1m rows single column search
// [Thread]          ~ 7ms per 1m rows single column search
// [Thread & AVX512] ~ 5ms per 1m rows single column search


//TODO:
//  -switch raw names to std::array<char, 4>


//󰣇 codebase/cpp/Aladata ❯ g++ -std=c++20 -O3 -march=native find.cpp $(root-config --cflags --libs) && ./a.out                                                                                                                20:48 
//[Sys] Found 4 users named: alice
//   Within a database made of 59406880 users
//    [User]: alice [Age]: 45
//    [User]: alice [Age]: 19
//    [User]: alice [Age]: 30
//    [User]: alice [Age]: 61
//
//       #========================================================================================================================================#
//       | LATTE TELEMETRY [TIME][RAW]                                                                                                            |
//       #========================================================================================================================================#
//       | COMPONENT            |   SAMPLES |        AVG |     MEDIAN |    STD DEV |     SKEW |        MIN |        MAX |      RANGE |     BYPASS |
//       |----------------------------------------------------------------------------------------------------------------------------------------|
//       | Search Iteration     |         1 |  348.14 ms |  348.14 ms |    0.00 ns |     0.00 |  348.14 ms |  348.14 ms |    0.00 ns |          0 |
//       | Findings             |         3 |    6.04 ms |    6.31 ms |    2.18 ms |    -0.18 |    3.26 ms |    8.57 ms |    5.31 ms |          0 |
//       | Write                |         1 |    11.37 s |    11.37 s |    0.00 ns |     0.00 |    11.37 s |    11.37 s |    0.00 ns |          0 |
//       | rng chars            |     65525 |   10.43 ns |    9.81 ns |    9.11 ns |     1.25 |    0.21 ns |   70.13 ns |   69.92 ns |         11 |
//       | Write Loop           |     65522 |   52.01 ns |   50.09 ns |    7.85 ns |     6.60 |   40.08 ns |  150.28 ns |  110.21 ns |         14 |
//       | Global_full_search   |         1 |    11.90 s |    11.90 s |    0.00 ns |     0.00 |    11.90 s |    11.90 s |    0.00 ns |          0 |
//       #========================================================================================================================================#
//       Searching took: 347.48 ms
//       [Expected] RNtuple take 5.74 ms per 1M iters
//       Total rows: 59.41 M
//         Rows per Thread: 4.95 M
//             AVX512 per thread: 309.41 K
//󰣇 codebase/cpp/Aladata ❯ du -sh ./data/search/users.root                                                                                                                                                                    20:49 
//  465M    ./data/search/users.root

