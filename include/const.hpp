#pragma once

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

auto ConstSearch(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::array<char, name_len>>("name");
  const auto nEntries = reader->GetNEntries();
  std::vector<uint64_t> matches;
  matches.reserve(10);

  char xa[name_len];
  char xb[name_len];

  Latte::Mid::Start("2) Search");

  std::memcpy(xa, tName, name_len);
  for (uint64_t i = 0; i < nEntries; ++i) {
    const auto& iname = vName(i);
    std::memcpy(xb, iname.data(), name_len);
    if (std::memcmp(xa, xb, name_len) == 0) matches.push_back(i);
  }
  Latte::Hard::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(matches), tName);
}


