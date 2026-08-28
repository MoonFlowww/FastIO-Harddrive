#pragma once

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
  while (half > 1) {
    half = half >> 1;
    const uint32_t value = vHName(base + half - 1);
    base += half & (uint64_t)(-(int64_t)(value < key));
    // 0,1 -> 0,-1 -> 0x0,0xFF.. -> base+=0, base+=half
    // LATTE_PULSE("2.1.1) BODY LOOP");
  }
  base += (uint64_t)(vHName(base) < key);
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
  for (uint64_t i = base; i < nEntries && vHName(i) == key; ++i)
    matches.push_back(i);
  Latte::Mid::Stop("2.2) tail");

  Latte::Fast::Start("2.3) Deblooming");
  for (size_t i = 0; i < matches.size();) {
    if (std::memcmp(tName, vName(matches[i]).data(), name_len) != 0)
      matches.erase(matches.begin() + i);
    else
      ++i;
  }
  Latte::Fast::Stop("2.3) Deblooming");
  Latte::Mid::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(matches), tName);
}
