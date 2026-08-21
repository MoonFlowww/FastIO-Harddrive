# FastSearch Bench over RNTuple

Query `Search "alice" in name`, 59.4M rows, 4 matches, single thread, no shard.

## Results

Per 1M rows, current build (`GetDirectAccessView` + fixed-size `char[5]` names).

| method | ms / 1M | Total (~60m rows) |
|---|---|---|
| `NoSearchHash` (raw `uint32_t` iteration) | 4.70 | 278.99 ms |
| `NoSearch` (raw `char[5]` iteration) | 9.04 | 537.12 ms |
| --- | --- | --- |
| `SIMD_fnv1a_Search` | 4.95ms | 294.23 ms |
| `SIMD_FSL_Search` | 10.49ms | 623.01 ms |
| `ConstSearch` | 11.21ms | 666.17 ms |
| `BinarySearch` (new, additive) | 343us | 20.36 ms |
| `BinarySearch` (old, add/sub) | 323us | 19.17 ms |
| `TreeBinarySearch` | 9.7ns | ~530ns |
| `O1Search` | O(1) | load: 1.12 s // O1find: 19.56 ms |

Notes:

- Fixed-size columns use `GetDirectAccessView` (`hash_name` `uint32_t`, `age` `int`). Direct pointer into the page buffer, fixed stride, no per-element `shared_ptr` refcount traffic.
- `name` is a fixed-size `std::array<char,5>` in RNTuple. Search-side arrays are plain `char[name_len]`. Confirmation is a 5-byte `memcmp`.
- AVX512 methods bloom with mask+ctz. The vectorized search does not iterate over all rows on the confirm path.
- `BinarySearch` needs `hash_name` sorted ascending. One-time sort+rewrite cost: 7.02 s (measured this run), not counted in search time.
- `TreeBinarySearch` needs `hash_name` sorted ascending, same as `BinarySearch`. Tree nodes store `uint32_t` (hash width) — a wider node type halves keys per cache line and doubles the tree's memory footprint.
- `O1Search` loads a persisted `unordered_map` index. The load dominates (find = 19–20 ms).

>  if we can keep umap in ram, it is the fastest, unfortunately it scale poorly ...

## Per-region hardware counters (likwid markers)

`1_Write` and `2_Search` are wrapped with the LIKWID marker API
(`-DLIKWID_PERFMON`, `-llikwid`). Run:

```bash
/usr/bin/likwid-lua ~/.local/opt/likwid-5.5.2/likwid-perfctr -m -C 0 -g BRANCH ./bin/fastsearch_fnv1a   # branch counters + CPI
/usr/bin/likwid-lua ~/.local/opt/likwid-5.5.2/likwid-perfctr -m -C 0 -g CACHE  ./bin/fastsearch_fnv1a   # cache miss rates
/usr/bin/likwid-lua ~/.local/opt/likwid-5.5.2/likwid-perfctr -m -C 0 -g MEM    ./bin/fastsearch_fnv1a   # DRAM bandwidth
```

Requires the fixed likwid 5.5.2 reader (system 5.5.1 ships a marker-file
parsing bug on glibc >= 2.40: `sscanf "%d:%139c"` fails on short region
names). The fixed build lives in `~/.local/opt/likwid-5.5.2` with a patched
module at `~/.local/share/lua/likwid.lua`; perf groups are symlinked from
`/usr/share/likwid/perfgroups`. `perf_event_paranoid` must be <= 1.
`~/.local/bin/likwid-perfctr` is a wrapper to the fixed 5.5.2 reader, so a
plain `likwid-perfctr -m ...` on PATH also works.
Plain `./bin/...` runs are unaffected (markers become no-ops, Latte output
unchanged).

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

TreeBinarySearch                           ~8.42 ns/1M   O(log n), precomputed tree
  requires hash_name sorted ascending (one-time sort+rewrite: 7.02 s)
  BuildTree: recursive, writes hash into tree_v[idx], row into tree_i[idx]
  tree_v, tree_i: uint32_t, size = bit_ceil(N+1)-1 (perfect tree, next pow2)
  idx=0
  while idx < tree_v.size():
      node = tree_v[idx]
      if node == key: found; break
      idx = 2*idx+1 if key < node else 2*idx+2 (faster than bool substraction)
  tail: scan while hash_name[i] == key -> candidates
  memcmp confirm each candidate
  node type must match key width (uint32_t) -- wider nodes halve keys/cache-line

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
- gpu slice decoding (low)
- unsync multithread for lightweight max speed
- sharding: `SHARD_MAP(hash % BIG_N) -> shard id`. `BIG_N > shard count`, so `% BIG_N` leaves room to scale. To add a shard: split one existing shard into subs, update the lookup table. (high)
- mixing ternary and binary search. Ternary introduce more cache load (with flat-tree sitting in a vec, we can avoid cache load only on early depths) and one additional cmp plus its corresponding logic (jmp or branchless tricks; if trick, will be expensive). Furthermore skipping 66% instead of 50% is highly interesting if x it large (either early depths).
- For scaling, instead of rebuilding the tree-roots every new lines in DB, we can trim the last one/two level (deepest) for an AVX512 O(n) search; if DB wide enough, new rows will always land between range-bounds, making tree-update evitable. Might not be search-speed efficient, even with "sub-range prefetch" (when reaching near end of tree we can start guessing what sub portion of the rows will be loaded; when roots are removed. Idiomatic: prefetch 50%-end of left and 50%-start of right, cache update with cache line to overwrite the half we don't need ? Need more light). Then if we have an update we can simply use a offset
