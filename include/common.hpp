#pragma once

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleRange.hxx>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>
#include <algorithm>
#include <bit>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "../latte.hpp"


inline constexpr uint64_t N = 100'000'000ULL;
inline constexpr int       name_len = 5;

struct SearchResult {
  std::unique_ptr<ROOT::RNTupleReader> reader;
  std::vector<uint64_t> matches;
  char tName[name_len]; // human readable name

  SearchResult(std::unique_ptr<ROOT::RNTupleReader> r, std::vector<uint64_t> m, const char (&n)[name_len])
    : reader(std::move(r)), matches(std::move(m)) {
    std::memcpy(tName, n, name_len);
  }
  SearchResult() = default;
};

inline auto fnv1a(const char* str, std::size_t len) -> uint32_t {
  uint32_t hash = 0x811C9DC5; // FNV offset basis
  for (std::size_t i = 0; i < len; ++i) {
    hash ^= static_cast<unsigned char>(str[i]);
    hash *= 0x01000193u; // FNV prime
  }
  return hash;
}


static inline auto BinaryTreeDepth(uint64_t N) -> int {
  return std::bit_width(N)-1;
}


void sortAndSaveRNTuple() {
  Latte::Fast::Start("Sort RNTuple");

  // Read the RNTuple
  auto reader = ROOT::RNTupleReader::Open("Users", "./data/search/users.root");

  auto vName = reader->GetView<std::array<char, name_len>>("name");
  auto vHName = reader->GetView<uint32_t>("hash_name");
  auto vAge = reader->GetView<int>("age");

  struct User {
    uint32_t hash_name;
    std::array<char, name_len> name;
    int age;
  };

  std::vector<User> users;

  // Read all entries
  Latte::Fast::Start("Reading");
  for (auto i : reader->GetEntryRange()) {
    users.push_back({
      vHName(i),
      vName(i),
      vAge(i)
    });
  }
  Latte::Fast::Stop("Reading");

  Latte::Fast::Start("Sorting");
  // Sort by hash_name
  std::sort(users.begin(), users.end(), 
            [](const User& a, const User& b) {
            return a.hash_name < b.hash_name;
            });
  Latte::Fast::Stop("Sorting");

  Latte::Fast::Start("Writing sorted data");
  // Write back to new RNTuple
  auto model = ROOT::RNTupleModel::Create();
  auto fldName = model->MakeField<std::array<char, name_len>>("name");
  auto fldHashName = model->MakeField<uint32_t>("hash_name");
  auto fldAge = model->MakeField<int>("age");

  ROOT::RNTupleWriteOptions opts;
  //opts.SetCompression(401);  // LZ4 (default = ZSTD 505)
  auto writer = ROOT::RNTupleWriter::Recreate(
    std::move(model), "Users", "./data/search/users.root", opts
  );

  for (const auto& user : users) {
    *fldName = user.name;
    *fldHashName = user.hash_name;
    *fldAge = user.age;
    writer->Fill();
  }

  Latte::Fast::Stop("Writing sorted data");
  Latte::Fast::Stop("Sort RNTuple");
}


static auto loadIndex(const std::string& path) -> std::unordered_map<uint32_t, std::vector<uint64_t>> {
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

