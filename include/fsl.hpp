#pragma once

#include <immintrin.h>
#include <stdint.h>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <array>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "common.hpp"
#include "../latte.hpp"


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
      secB[j] = static_cast<uint8_t>(s[1]);
      lastB[j] = static_cast<uint8_t>(s.back());
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

