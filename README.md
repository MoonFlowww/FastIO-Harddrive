# FastSearch Bench over RNTuple

Query `Search "alice" in name`, 59.4M rows, 4 matches, single thread, no shard.

## Results

Per 1M rows, current build (`GetDirectAccessView` + fixed-size `char[5]` names):

| method | ms / 1M | Total (~60m rows) |
|---|---|---|
| `NoSearchHash` (raw `uint32_t` iteration) | 4.47 | 277.36 ms |
| `NoSearch` (raw `char[5]` iteration) | 9.15 | 538.14 ms |
| --- | --- | --- |
| `SIMD_fnv1a_Search` | 4.89 | 298.64 ms |
| `SIMD_FSL_Search` | 10.43 | 625.84 ms |
| `ConstSearch` | 11.06 | 670.86 ms |
| `O1Search` | O(1) | load: 1.24 s // O1find: 18.24 ms |

Notes:

- Fixed-size columns are read with `GetDirectAccessView` (`hash_name` `uint32_t`, `age` `int`): direct pointer into the page buffer, fixed stride, no per-element `shared_ptr` refcount traffic.
- The `name` column is a fixed-size `std::array<char,5>` inside RNTuple; all search-side arrays are plain `char[name_len]`, and confirmation is a 5-byte `memcmp`.
- AVX512 methods bloom with mask+ctz: the vectorized search does not iterate over all rows on the confirm path.
- `O1Search` loads a persisted `unordered_map` index; the load dominates (find = 18–19 ms).

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
runs       perf under `taskset -c 3`
```

Run-to-run: instructions ±0.06%, L1 loads ±0.3%, cycles ±4%, task-clock ±2.7%.
Deltas under ~5% are not separable.

## perf stat, whole program

Whole-program runs include the ~4 s write phase; search-only differences are in Results.

| | nosearch | nosearchhash | fnv1a | fsl | const | o1 |
|---|---|---|---|---|---|---|
| task-clock (ms) | 4636.15 | 4362.66 | 4410.18 | 4744.79 | 4760.50 | 31783.19 |
| cycles:u | 22.118 G | 20.771 G | 20.729 G | 22.541 G | 22.686 G | 166.279 G |
| instructions:u | 60.959 G | 55.734 G | 55.980 G | 60.965 G | 61.594 G | 93.950 G |
| IPC | 2.756 | 2.683 | 2.701 | 2.705 | 2.715 | 0.565 |
| L1-dcache-loads | 19.358 G | 17.633 G | 17.687 G | 19.269 G | 19.769 G | 35.324 G |
| L1-dcache-load-misses | 289.5 M | 290.7 M | 292.5 M | 289.8 M | 289.2 M | 1056.5 M |
| page-faults | 168 842 | 168 836 | 170 486 | 170 485 | 169 253 | 433 097 |

Delta vs `nosearch` (same build):

| | Δ task-clock | Δ cycles | Δ insn | insn/row | L1 loads/row |
|---|---|---|---|---|---|
| fnv1a | -226.0 ms | -1.389 G | -4.979 G | -83.8 | -28.1 |
| fsl | +108.6 ms | +0.423 G | +0.006 G | +0.1 | -1.5 |
| const | +124.4 ms | +0.567 G | +0.635 G | +10.7 | +6.9 |

## Access path

```
SIMD_fnv1a_Search, hot path (per row):
  vhashView(v+j)                   GetDirectAccessView<uint32_t> "hash_name"
                                   direct pointer into page buffer, fixed stride
                                   u32 load — no shared_ptr, no materialization
  memcmp(tName, vName(v+j).data()) only on bloom hits; GetView<std::array<char,5>>
                                   "name" materializes 5 bytes (rare, 4 hits)
```

`SIMD_FSL_Search` reads the name column every row via `GetView<std::array<char,5>>`
(gathers `front()`, `[1]`, `back()` into 3 scratch arrays), which is why it is ~2× the
fnv1a path.

## Methods

```
SIMD_fnv1a_Search                          4.89 ms/1M   3 bloom captures
  GetDirectAccessView<uint32_t> hash_name   (direct page-buffer access, fixed size)
  for v in steps of 16:
      load hash_name[v..v+16] -> __m512i
      mask = cmpeq_epi32(target_hash, vec)
      while mask: j=ctz(mask); memcmp confirm; mask &= mask-1
  tail: scalar memcmp

SIMD_FSL_Search                            10.43 ms/1M   3275 bloom captures
  for v in steps of 64:
      gather byte[0], byte[1], byte[len-1] of 64 names -> 3 scratch arrays
      mask = cmpeq8(f) & cmpeq8(s) & cmpeq8(l)
      while mask: j=ctz(mask); memcmp confirm; mask &= mask-1
  tail: scalar memcmp

NoSearchHash                               4.47 ms/1M
  for v in 0..N: touch(hash_name[v])        # GetDirectAccessView<uint32_t>

NoSearch                                   9.15 ms/1M
  for v in 0..N: touch(name[v].front())     # GetView<std::array<char,5>>

ConstSearch                                11.06 ms/1M
  for v in 0..N: memcmp(target, name[v], name_len)

O1Search                                   load 1.24 s || find 18.24 ms
  index = loadIndex("users.idx")
  rows  = index[fnv1a(target)]     # find = 18.24 ms
```


TODO:
- consumer ID
- gpu slice decoding
- unsync multithread for lightweight max speed
- sharding: `SHARD_MAP(hash % BIG_N) -> shard id` (`_ % BIG_N` is here to scale; if new shard necessary, we can just cut one into subs, and update the lookup table; BIG_N > Shards)
