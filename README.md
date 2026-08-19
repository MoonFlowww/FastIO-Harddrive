# FastSearch Bench over RNTuple

Query `Search "alice" in name`, 59.4M rows, 4 matches, single thread, no shard.

## Results

Per 1M rows, current build (`GetDirectAccessView` + fixed-size `char[5]` names).

| method | ms / 1M | Total (~60m rows) |
|---|---|---|
| `NoSearchHash` (raw `uint32_t` iteration) | 4.70 | 278.99 ms |
| `NoSearch` (raw `char[5]` iteration) | 9.04 | 537.12 ms |
| --- | --- | --- |
| `SIMD_fnv1a_Search` | 4.95 | 294.23 ms |
| `SIMD_FSL_Search` | 10.49 | 623.01 ms |
| `ConstSearch` | 11.21 | 666.17 ms |
| `BinarySearch` (new, additive) | 0.343 | 20.36 ms |
| `BinarySearch` (old, add/sub) | 0.323 | 19.17 ms |
| `O1Search` | O(1) | load: 1.12 s // O1find: 19.56 ms |

Notes:

- Fixed-size columns use `GetDirectAccessView` (`hash_name` `uint32_t`, `age` `int`). Direct pointer into the page buffer, fixed stride, no per-element `shared_ptr` refcount traffic.
- `name` is a fixed-size `std::array<char,5>` in RNTuple. Search-side arrays are plain `char[name_len]`. Confirmation is a 5-byte `memcmp`.
- AVX512 methods bloom with mask+ctz. The vectorized search does not iterate over all rows on the confirm path.
- `BinarySearch` needs `hash_name` sorted ascending. One-time sort+rewrite cost: 7.02 s (measured this run), not counted in search time.
- `O1Search` loads a persisted `unordered_map` index. The load dominates (find = 19–20 ms).

>  if we can keep umap in ram, it is the fastest, unfortunately it scale poorly ...

## Environment

```
CPU        AMD Ryzen 5 7600X, 6C/12T (0-11), SMT on
freq       amd-pstate-epp active, prefcore on, gov=performance, epp=performance, boost=1
           range pinned [5457105..5457105] kHz, observed cur 4.23-5.41 GHz
thermal    60C, not throttling
THP        enabled=always, defrag=never
kernel     swappiness=1, numa_balancing=0, randomize_va_space=0, nmi_watchdog=0
perf       perf_event_paranoid=2, kptr_restrict=0, yama.ptrace_scope=1
```

Run-to-run variance: deltas under ~5% are noise, not a real difference.

## Access path

```
SIMD_fnv1a_Search, hot path (per row):
  vhashView(v+j)                   GetDirectAccessView<uint32_t> "hash_name"
                                   direct pointer into page buffer, fixed stride
                                   u32 load, no shared_ptr, no materialization
  memcmp(tName, vName(v+j).data()) only on bloom hits; GetView<std::array<char,5>>
                                   "name" materializes 5 bytes (rare, 4 hits)
```

`SIMD_FSL_Search` reads the name column every row via `GetView<std::array<char,5>>`.
It gathers `front()`, `[1]`, `back()` into 3 scratch arrays. That is why it runs ~2x the
fnv1a path.

## Methods

```
SIMD_fnv1a_Search                          4.95 ms/1M   3 bloom captures
  GetDirectAccessView<uint32_t> hash_name   (direct page-buffer access, fixed size)
  for v in steps of 16:
      load hash_name[v..v+16] -> __m512i
      mask = cmpeq_epi32(target_hash, vec)
      while mask: j=ctz(mask); memcmp confirm; mask &= mask-1
  tail: scalar memcmp

SIMD_FSL_Search                            10.49 ms/1M   3275 bloom captures
  for v in steps of 64:
      gather byte[0], byte[1], byte[len-1] of 64 names -> 3 scratch arrays
      mask = cmpeq8(f) & cmpeq8(s) & cmpeq8(l)
      while mask: j=ctz(mask); memcmp confirm; mask &= mask-1
  tail: scalar memcmp

BinarySearch (new)                         342.78 us/1M   O(log n), branchless
  requires hash_name sorted ascending (one-time sort+rewrite: 7.02 s)
  base=0; len=N
  while len > 1:
      half = len >> 1
      base += half if hash_name[base+half-1] < key else 0   # branchless
      len -= half
  tail: scan while hash_name[i] == key -> candidates
  memcmp confirm each candidate

BinarySearch (old)                         322.73 us/1M   O(log n), add/sub stepping
  same probe pattern as (new); only the index arithmetic differs (noise-level delta)

NoSearchHash                               4.70 ms/1M
  for v in 0..N: touch(hash_name[v])        # GetDirectAccessView<uint32_t>

NoSearch                                   9.04 ms/1M
  for v in 0..N: touch(name[v].front())     # GetView<std::array<char,5>>

ConstSearch                                11.21 ms/1M
  for v in 0..N: memcmp(target, name[v], name_len)

O1Search                                   load 1.12 s || find 19.56 ms
  index = loadIndex("users.idx")
  rows  = index[fnv1a(target)]     # find = 19.56 ms
```


TODO:
- consumer ID
- gpu slice decoding
- unsync multithread for lightweight max speed
- sharding: `SHARD_MAP(hash % BIG_N) -> shard id`. `BIG_N > shard count`, so `% BIG_N` leaves room to scale. To add a shard: split one existing shard into subs, update the lookup table.
