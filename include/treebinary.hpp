#pragma once

#include <stdint.h>

#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <array>
#include <bit>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "../latte.hpp"
#include "common.hpp"

void BuildBinaryTree(ROOT::RNTupleDirectAccessView<uint32_t>& vHName,
                     std::vector<uint32_t>& tree_v,
                     std::vector<uint32_t>& tree_i,
                     int64_t start,
                     int64_t stop,
                     size_t idx) {
  if (start > stop) return;

  const int64_t mid = start + (stop - start) / 2;
  tree_v[idx] = vHName(static_cast<uint64_t>(mid));  // DB value
  tree_i[idx] = static_cast<uint32_t>(mid);          // DB idx

  BuildBinaryTree(vHName, tree_v, tree_i, start, mid - 1, 2 * idx + 1);  // left
  BuildBinaryTree(vHName, tree_v, tree_i, mid + 1, stop, 2 * idx + 2);  // right
}


auto TreeBinarySearch(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vHName = reader->GetDirectAccessView<uint32_t>("hash_name");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  // Perfect tree size = 2^n - 1
  size_t perfect_size = std::bit_ceil(N + 1) - 1;
  std::vector<uint32_t> tree_v(perfect_size, UINT32_MAX);
  std::vector<uint32_t> tree_i(perfect_size, UINT32_MAX);

  Latte::Fast::Start("BuildTree");
  BuildBinaryTree(vHName, tree_v, tree_i, 0, N - 1, 0);
  Latte::Fast::Stop("BuildTree");

  std::vector<uint64_t> matches;
  matches.reserve(10);

  Latte::Fast::Start("2) Search");
  uint32_t key = fnv1a(tName, name_len);
  Latte::Fast::Start("2.1) Search Body");

  size_t idx = 0;
  bool found = false;

  while (idx < tree_v.size()) {
    const uint32_t node = tree_v[idx];
    if (__builtin_expect(node == key, false)) {
      found = true;
      break;
    }
    // idx = 2*idx + (2 - (key < node));
    idx = key < node ? (2 * idx + 1)
                     : (2 * idx + 2);  // faster than bool substraction
    // LATTE_PULSE("2.1.1) BODY LOOP");
  }
  Latte::Fast::Stop("2.1) Search Body");
  /* 37.71ns, 2.24us
    const uint64_t node = tree_v[idx];
    uint64_t mask = (uint64_t)-int64_t(node == key); // 0x00 or 0xFF
    found = (idx & mask) | (found & ~mask);
    idx = (2 * idx + (2-(key < node))); //slower than (x ? y : z)
  */

  Latte::Fast::Start("2.2) Search Tail");
  if (found) {  // vName is IO bottleneck
    uint64_t row = tree_i[idx];
    uint64_t lo = row;
    while (lo > 0 && vHName(lo - 1) == key) --lo;
    uint64_t hi = row;
    while (hi + 1 < N && vHName(hi + 1) == key) ++hi;

    for (uint64_t r = lo; r <= hi; ++r) {
      if (std::memcmp(tName, vName(r).data(), name_len) == 0)
        matches.push_back(r);
    }
  }
  Latte::Fast::Stop("2.2) Search Tail");
  Latte::Fast::Stop("2) Search");
  return SearchResult(std::move(reader), std::move(matches), tName);
}
