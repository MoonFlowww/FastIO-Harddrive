#pragma once 

#include <stdint.h>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "../latte.hpp"
#include "common.hpp"


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


