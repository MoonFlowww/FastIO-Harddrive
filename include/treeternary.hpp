#pragma once

#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "../latte.hpp"
#include "common.hpp"

inline void BuildTernaryTree(ROOT::RNTupleDirectAccessView<uint32_t>& vHName,
                             std::vector<uint32_t>& tree_v,
                             std::vector<uint32_t>& tree_i,
                             int64_t start,
                             int64_t stop,
                             size_t node_idx) {
  if (start > stop) return;
  const int64_t low = start + (stop - start) / 3;
  const int64_t high = start + 2 * (stop - start) / 3;

  const size_t base = 2 * node_idx;
  tree_v[base] = vHName(static_cast<uint64_t>(low));
  tree_i[base] = static_cast<uint32_t>(low);
  tree_v[base + 1] = vHName(static_cast<uint64_t>(high));
  tree_i[base + 1] = static_cast<uint32_t>(high);

  BuildTernaryTree(vHName, tree_v, tree_i, start, low - 1, 3 * node_idx + 1);
  BuildTernaryTree(vHName, tree_v, tree_i, low + 1, high - 1, 3 * node_idx + 2);
  BuildTernaryTree(vHName, tree_v, tree_i, high + 1, stop, 3 * node_idx + 3);
}

static constexpr auto TernaryTreeDepth(const uint64_t n) -> int {
  if (n == 0) return -1;
  uint64_t val = 2 * n - 1;
  uint64_t cap = 1;
  int h = 0;
  while (cap * 3 <= val) {
    cap *= 3;
    ++h;
  }
  return h;
}

static constexpr auto Pow3(const int e) -> uint64_t {
  uint64_t p = 1;
  for (int i = 0; i < e; ++i) p *= 3;
  return p;
}

inline auto TreeTernarySearch(const char (&tName)[name_len]) -> SearchResult {
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");
  auto vHName = reader->GetDirectAccessView<uint32_t>("hash_name");
  auto vName = reader->GetView<std::array<char, name_len>>("name");

  constexpr auto depth = TernaryTreeDepth(N);
  constexpr size_t node_cap = (Pow3(depth + 1) - 1) / 2;
  std::vector<uint32_t> tree_v(2 * node_cap, UINT32_MAX);
  std::vector<uint32_t> tree_i(2 * node_cap, UINT32_MAX);

  Latte::Fast::Start("BuildTree");
  BuildTernaryTree(vHName, tree_v, tree_i, 0, static_cast<int64_t>(N) - 1, 0);
  Latte::Fast::Stop("BuildTree");

  std::vector<uint64_t> matches;
  matches.reserve(10);

  Latte::Fast::Start("2) Search");
  uint32_t key = fnv1a(tName, name_len);
  Latte::Fast::Start("2.1) Search Body");

  uint64_t idx = 0;
  uint64_t found = 0;

#pragma unroll(depth)
  for (int d = 0; d < depth; ++d) {
    const size_t base = 2 * idx;
    const uint32_t low = tree_v[base];
    const uint32_t high = tree_v[base + 1];

    const int off = 1 + (key > high) - (key < low);  // 0 left, 1 mid, 2 right

    const bool mask_low = (key == low);
    const bool mask_high = (key == high);

    // const bool mask_side = (key==high); //== ((!(key==low)) | (key==high)) ->
    // (1 0) or (0 1) ->! (0 0) or (1 1) ->| 0 or 1 found = (idx+mask_high &
    // mask_pass) | (found & ~mask_pass);
    found = (off == 1) & (mask_low | mask_high)
                ? base + mask_high
                : found;  // fast path.  off==1 only if off=1+0-0

    idx = 3 * idx + off + 1;
    // LATTE_PULSE("2.1.1) BODY LOOP");
  }

  /*
  1) cmp-chain
      key < low   ; cmpl -> left
      key > high  ; cmph -> high
           -> 3 * idx + (1 + (cmph - cmpl))

  2) asymmetric memcmp
      key-value -> [-x;+x]
                  -> delta>>32-1 : left    ; sign == (-), compiler(x<0) == x>>31
                  -> delta== 0   : middle  ; key == value
                  -> else        : right   ; else

  3) symmetric memcmp (cmp-chain with 3sub instead of 2)
      key-value -> [-x;+x]
                  -> ((cmp>0)-(cmp<0))
                    -> 3 * idx + 1 - r

      in any case i hit (3 * idx + (1 - something))


  1) delta
      cmpright = key - high;
      cmpleft = key - low;
      off = 1 + cmpright>0 - cmpleft>0

      3sub + 2cmp

  2) inverse of delta, cmp first the delta
      off = 1 + key > high - key < low

      mask = uint-int(key==high | key==low) // need 2 mask: mask1= side,
  mask2=override/keep

  */
  Latte::Fast::Stop("2.1) Search Body");

  Latte::Fast::Start("2.2) Search Tail");
  if (found) {  // vName is IO bottleneck
    uint64_t row = tree_i[found];
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
