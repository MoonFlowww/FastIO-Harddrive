#pragma

#include <immintrin.h>
#include <stdint.h>

#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <array>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "../latte.hpp"
#include "common.hpp"

auto SIMD_fnv1a_Search(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto       vName    = reader->GetView<std::array<char, name_len>>("name");
  auto       vHView   = reader->GetDirectAccessView<uint32_t>("hash_name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

  Latte::Mid::Start("2) Search");
  __m512i key = _mm512_set1_epi32(static_cast<int>(fnv1a(tName, name_len)));
  alignas(64) uint32_t hashed[16];

  Latte::Fast::Start("2.1) SIMD body");
  uint64_t v = 0;
  for (; v + 16 <= nEntries; v += 16) {
    for (uint32_t j = 0; j < 16; ++j) {  // loading v64
      const auto& h = vHView(v + j);
      hashed[j]     = static_cast<uint32_t>(h);
    }
    __m512i vcan_hashed = _mm512_load_si512(hashed);  // last step of loading

    uint64_t mask = _mm512_cmpeq_epi32_mask(key, vcan_hashed);

    while (mask) {  // unroll mask to save bloomed
      LATTE_PULSE("2.1.1) unroll bloom");
      uint32_t j = __builtin_ctzll(mask);
      if (std::memcmp(tName, vName(v + j).data(), name_len) == 0)
        matches.push_back(v + j);
      mask &= mask - 1;
    }
  }

  Latte::Fast::Stop("2.1) SIMD body");
  Latte::Fast::Start("2.2) SIMD tail");

  for (; v < nEntries; ++v) {  // v tail (no bloom)
    LATTE_PULSE("2.2.1) Tail");
    if (std::memcmp(tName, vName(v).data(), name_len) == 0)
      matches.push_back(v);
  }

  Latte::Fast::Stop("2.2) SIMD tail");
  Latte::Hard::Stop("2) Search");
  return SearchResult(std::move(reader),
                      std::move(matches),
                      tName);  // move to avoid destruct (reader is unique_ptr)
}
