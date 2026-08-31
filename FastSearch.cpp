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

#include <likwid-marker.h>

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dependencies/event_counter.h"
#include "include/baseline.hpp"
#include "include/binary.hpp"
#include "include/common.hpp"
#include "include/const.hpp"
#include "include/fnv1a.hpp"
#include "include/fsl.hpp"
#include "include/o1.hpp"
#include "include/treebinary.hpp"
#include "include/treeternary.hpp"
#include "latte.hpp"

// Every RUN_* switch defaults to 0 so `if constexpr` works even when the
// justfile only defines the active one.
#ifndef RUN_O1SEARCH
  #define RUN_O1SEARCH 0
#endif
#ifndef RUN_SIMDFSLSEARCH
  #define RUN_SIMDFSLSEARCH 0
#endif
#ifndef RUN_SIMDFNV1ASEARCH
  #define RUN_SIMDFNV1ASEARCH 0
#endif
#ifndef RUN_CONSTSEARCH
  #define RUN_CONSTSEARCH 0
#endif
#ifndef RUN_BINARYSEARCH
  #define RUN_BINARYSEARCH 0
#endif
#ifndef RUN_TREEBINARYSEARCH
  #define RUN_TREEBINARYSEARCH 0
#endif
#ifndef RUN_TREETERNARYSEARCH
  #define RUN_TREETERNARYSEARCH 0
#endif
#ifndef RUN_NOSEARCH
  #define RUN_NOSEARCH 0
#endif
#ifndef RUN_NOSEARCHHASH
  #define RUN_NOSEARCHHASH 0
#endif

counters::event_collector collector;

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
    return static_cast<uint32_t>(next() >> 32) % limit;
  }
};


static void RNG_String(XorPRNG& rng, char (&name)[name_len]) {
  static constexpr char chars[] = "abcdefghijklmnopqrstuvwxyz";  // 26 + NUL
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


static void saveIndex(
    const std::unordered_map<uint32_t, std::vector<uint64_t>>& index,
    const std::string& path
) {
  std::ofstream f(path, std::ios::binary);

  uint64_t mapSize = index.size();
  f.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

  for (const auto& [hash, rows] : index) {
    f.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
    uint64_t rowCount = rows.size();
    f.write(reinterpret_cast<const char*>(&rowCount), sizeof(rowCount));
    f.write(
        reinterpret_cast<const char*>(rows.data()), rowCount * sizeof(uint64_t)
    );
  }
}


void write() {
  Latte::Fast::Start("1) Write");
  Latte::Fast::Start("1.1) Write init");
  auto model = ROOT::RNTupleModel::Create();
  // RNTuple fixed-size array column; a raw char[5] would be normalized to std::array<char,5> anyway.
  auto name = model->MakeField<std::array<char, name_len>>("name");
  auto hash_name = model->MakeField<uint32_t>("hash_name");
  auto age = model->MakeField<int>("age");

  ROOT::RNTupleWriteOptions opts;
  //opts.SetCompression(401);  // LZ4 (default = ZSTD 505)
  auto writer = ROOT::RNTupleWriter::Recreate(
      std::move(model),
      "Users",
      "./data/search/users.root",
      opts
  );  // when writer destructed, it write the footer in file
  //1) Fill() × 50k        ->  fills in-memory column buffers (pages)
  //1.2) page full (~64KB)   ->  page gets compressed (Optional, LZ4 here)
  //3) cluster threshold   ->  all column pages flushed to disk as one cluster (~50MB default)
  //4) destructor          ->  final cluster + footer (schema, cluster index) written

  Latte::Fast::Stop("1.1) Write init");

  [[maybe_unused]] std::unordered_map<uint32_t, std::vector<uint64_t>> index;
  XorPRNG rng;
  for (int i = 0; std::cmp_less(i, N); ++i) {
    char user[name_len];
    RNG_String(rng, user);
    uint32_t huser = fnv1a(user, name_len);
    std::memcpy(name->data(), user, name_len);
    *hash_name = huser;
    *age = RNG_int(rng);
    LATTE_FIELD(writer->Fill());  // per-row write latency (span id = "write")
    if constexpr (RUN_O1SEARCH) {
      index[huser].push_back(i);
    }
    LATTE_PULSE("1.2) Write Loop");
  }
  if constexpr (RUN_O1SEARCH) {
    Latte::Fast::Start("1.3) Write SaveIndex");
    saveIndex(index, "./data/search/users.idx");
    Latte::Fast::Stop("1.3) Write SaveIndex");
  }
  Latte::Fast::Stop("1) Write");
}


void read(SearchResult& Sresult) {
  Latte::Fast::Start("read");
  Latte::Mid::Start("3) Read");
  Latte::Fast::Start("3.1) Read Init");
  auto vName = Sresult.reader->GetView<std::array<char, name_len>>("name");
  auto vAge = Sresult.reader->GetDirectAccessView<int>("age");
  Latte::Fast::Stop("3.1) Read Init");
  std::cout << "[Sys] Found " << Sresult.matches.size()
            << " users named: " << std::string_view(Sresult.tName, name_len)
            << "\n   Within a database made of "
            << Sresult.reader->GetNEntries() << " users" << '\n';

  for (auto& idx : Sresult.matches) {
    const auto& nm = vName(idx);
    std::cout << "[" << idx << "]";
    std::cout << " [User]: ";
    std::cout.write(nm.data(), nm.size());
    std::cout << " [Age]: " << vAge(idx) << '\n';
    LATTE_PULSE("3.2) Read Findings");
  }
  Latte::Hard::Stop("3) Read");
  Latte::Fast::Stop("read");
}


template <auto SEARCH>
std::pair<counters::event_aggregate, size_t> bench(
    const char (&tName)[name_len],
    size_t min_repeat = 10,
    size_t min_time_ns = 40'000'000,
    size_t max_repeat = 10'000'000
) {
  size_t n = min_repeat ? min_repeat : 1;
  counters::event_aggregate warm_aggregate{};

  for (size_t i = 0; i < n; ++i) {
    std::atomic_thread_fence(std::memory_order_acquire);
    collector.start();
    std::invoke(SEARCH, tName);
    std::atomic_thread_fence(std::memory_order_release);

    warm_aggregate << collector.end();
    if ((i + 1 == n) && (warm_aggregate.total_elapsed_ns() < min_time_ns) &&
        (n < max_repeat)) {
      n *= 10;
    }
  }

  counters::event_aggregate aggregate{};
  for (size_t i = 0; i < 10; ++i) {
    std::atomic_thread_fence(std::memory_order_acquire);
    collector.start();
    for (size_t j = 0; j < n; ++j) {
      std::invoke(SEARCH, tName);
    }
    std::atomic_thread_fence(std::memory_order_release);

    aggregate << collector.end();
  }
  return {aggregate, n};
}


static std::string format_compact(double v) {
  if (v >= 1e9)
    return std::format("{:.2f} G", v / 1e9);
  else if (v >= 1e6)
    return std::format("{:.2f} M", v / 1e6);
  else if (v >= 1e3)
    return std::format("{:.2f} K", v / 1e3);
  else
    return std::format("{:.2f}", v);
}


void pretty_print_search(
    const std::pair<counters::event_aggregate, size_t>& result,
    double norm_divisor = 1.0,
    const char* norm_unit = "search"
) {
  const auto& agg = result.first;
  const size_t reps = result.second;
  if (collector.has_events() && agg.cycles() > 0.0) {
    const double ipc = agg.instructions() / agg.cycles();
    const std::string cyc = format_compact(agg.cycles() / norm_divisor);
    const std::string ins = format_compact(agg.instructions() / norm_divisor);
    std::println("  {:<10} {:>7} / {}", "cycles", cyc, norm_unit);
    std::println("  {:<10} {:>7} / {}", "ins", ins, norm_unit);
    std::println("  {:<10} {:5.2f}", "i/c", ipc);
    if (agg.branches() > 0.0)
      std::println(
          "  {:<10} {:5.2f}%",
          "br-miss",
          100.0 * agg.branch_misses() / agg.branches()
      );
  }
}


using SearchFn = SearchResult (*)(const char (&)[name_len]);

consteval SearchFn select_search() {
  if constexpr (RUN_O1SEARCH) return &O1Search;
  if constexpr (RUN_SIMDFSLSEARCH) return &SIMD_FSL_Search;
  if constexpr (RUN_SIMDFNV1ASEARCH) return &SIMD_fnv1a_Search;
  if constexpr (RUN_CONSTSEARCH) return &ConstSearch;
  if constexpr (RUN_BINARYSEARCH) return &BinarySearch;
  if constexpr (RUN_TREEBINARYSEARCH) return &TreeBinarySearch;
  if constexpr (RUN_TREETERNARYSEARCH) return &TreeTernarySearch;
  if constexpr (RUN_NOSEARCH) return &NoSearch;
  if constexpr (RUN_NOSEARCHHASH) return &NoSearchHash;
  return nullptr;
}


constexpr SearchFn search_fn = select_search();
static_assert(search_fn != nullptr, "no RUN_* search selected");

struct NormSpec {
  double divisor;
  const char* unit;
};


constexpr NormSpec norm_spec() {
  if constexpr (
      RUN_CONSTSEARCH || RUN_NOSEARCH || RUN_NOSEARCHHASH ||
      RUN_SIMDFSLSEARCH || RUN_SIMDFNV1ASEARCH
  ) {
    return {static_cast<double>(N) / 1'000'000.0, "1M rows"};
  } else if constexpr (RUN_TREEBINARYSEARCH || RUN_BINARYSEARCH) {
    return {static_cast<double>(std::bit_width(N)), "depth"};
  } else if constexpr (RUN_TREETERNARYSEARCH) {
    return {static_cast<double>(TernaryTreeDepth(N)), "depth"};
  } else {
    return {1.0, "search"};
  }
}


auto main() -> int {
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_REGISTER("1_Write");
  LIKWID_MARKER_REGISTER("2_Search");

  std::println("[{}]", __TIME__);

  char tName[name_len] = {'a', 'l', 'i', 'c', 'e'};

  LIKWID_MARKER_START("1_Write");
  LATTE_FIELD(write());
  LIKWID_MARKER_STOP("1_Write");
  if constexpr (
      RUN_BINARYSEARCH || RUN_TREEBINARYSEARCH || RUN_TREETERNARYSEARCH
  ) {
    LATTE_FIELD(sortAndSaveRNTuple());
  }

  LIKWID_MARKER_START("2_Search");
  auto result = bench<search_fn>(tName, 1, 40'000'000, 10'000'000);
  LIKWID_MARKER_STOP("2_Search");

  SearchResult Sresult = search_fn(tName);
  LATTE_FIELD(read(Sresult));

  Latte::DumpToStream(std::cout, Latte::Parameter::Time);


  auto avg_ns = [](const std::vector<double>& ns) -> double {
    return ns.empty() ? 0.0 : std::reduce(ns.begin(), ns.end()) / ns.size();
  };

  constexpr NormSpec norm = norm_spec();

  std::println(
      "{:<10}: {}",
      "Write",
      Latte::FormatTime(avg_ns(Latte::Snapshot("1) Write").to_ns()))
  );
  if constexpr (
      RUN_BINARYSEARCH || RUN_TREEBINARYSEARCH || RUN_TREETERNARYSEARCH
  ) {
    std::println(
        "{:<10}: {}",
        "Sort",
        Latte::FormatTime(avg_ns(Latte::Snapshot("Sort RNTuple").to_ns()))
    );
  }
  std::println(
      "{:<10}: {}",
      "Search",
      Latte::FormatTime(avg_ns(Latte::Snapshot("2) Search").to_ns()))
  );
  pretty_print_search(result, norm.divisor, norm.unit);
  std::println(
      "  Search/{}: {}",
      norm.unit,
      Latte::FormatTime(result.first.elapsed_ns() / norm.divisor)
  );
  std::println(
      "{:<10}: {}",
      "Read",
      Latte::FormatTime(avg_ns(Latte::Snapshot("read").to_ns()))
  );

  auto LargeFormat = [](double val) -> std::string {
    const char* units[] = {"", "K", "M", "B", "T"};
    int unit_idx = 0;
    while (val >= 1000.0 && unit_idx < 4) {
      val /= 1000.0;
      unit_idx++;
    }
    std::ostringstream ss;
    if (unit_idx == 0)
      ss << std::fixed << std::setprecision(0) << val;
    else
      ss << std::fixed << std::setprecision(2) << val << " " << units[unit_idx];
    return ss.str();
  };

  std::println("{:<10}: {}", "Total Rows", LargeFormat(N));

  LIKWID_MARKER_CLOSE;
}
