#pragma once

#include <stdint.h>

#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../latte.hpp"
#include "common.hpp"

auto O1Search(const char (&tName)[name_len]) -> SearchResult {
  Latte::Fast::Start("LoadIndex");
  const auto index = loadIndex("./data/search/users.idx");
  Latte::Fast::Stop("LoadIndex");

  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  Latte::Mid::Start("2) Search");
  uint32_t key = fnv1a(tName, name_len);
  Latte::Fast::Start("2.2) Search find");
  std::vector<uint64_t> matches;
  const auto it = index.find(key);
  if (it != index.end()) {
    for (uint64_t row : it->second) {
      if (std::memcmp(tName, vName(row).data(), name_len) == 0)
        matches.push_back(row);
    }
  }
  Latte::Fast::Stop("2.2) Search find");
  Latte::Hard::Stop("2) Search");

  return SearchResult(std::move(reader), std::move(matches), tName);
}
