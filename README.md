# FastSearch Bench over RNTuple

Query `Search "alice"`, 59.4M rows, 4 matches, single thread, no shard.

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

## Timing

| method | ms / 1M | telemetry search |
|---|---|---|
| `SIMD_fnv1a_Search` | 12.77 | 769.18 ms |
| `NoSearch` | 16.38 | — |
| `SIMD_FSL_Search` | 16.81 | 1.01 s |
| `ConstSearch` | 20.21 | 1.19 s |
| `O1Search` | 1200 | 1.20 s (LoadIndex; find = 226.79 us) |


>  if we can keep umap in ram, it is the fastest, unfortunately it scale poorly ...

## perf stat, whole program

| | nosearch | fnv1a | fsl | const | o1 |
|---|---|---|---|---|---|
| task-clock (ms) | 5397.87 | 5220.96 | 5705.49 | 5656.66 | 32793.72 |
| cycles:u | 26.465 G | 25.494 G | 27.870 G | 27.686 G | 171.271 G |
| instructions:u | 71.092 G | 61.598 G | 71.197 G | 73.614 G | 95.382 G |
| IPC | 2.686 | 2.416 | 2.555 | 2.659 | 0.557 |
| L1-dcache-loads | 22.884 G | 19.491 G | 23.008 G | 23.782 G | 35.590 G |
| L1-dcache-load-misses | 383.3 M | 351.1 M | 388.3 M | 387.1 M | 1108.3 M |
| page-faults | 97 205 | 97 213 | 97 217 | 97 214 | 419 888 |

Delta vs `nosearch`:

| | Δ task-clock | Δ cycles | Δ insn | insn/row | L1 loads/row |
|---|---|---|---|---|---|
| fnv1a | -176.9 ms | -0.970 G | -9.494 G | -159.8 | -57.1 |
| fsl | +307.6 ms | +1.405 G | +0.106 G | +1.8 | +2.1 |
| const | +258.8 ms | +1.222 G | +2.522 G | +42.5 | +15.1 |

## Profile

`SIMD_fnv1a_Search` inner loop:

```
      cmpb  $0x0,__libc_single_threaded   ; fast path check, FAILS
      je    660
660:  lock xadd %edx,(%r9)                ; 40.15%
551:  mov   0x8(%r12),%rax                ; 33.06%  dep on locked store
      movabs $0x100000001,%r8
      cmp   %r8,%rax
6c0:  lock xadd %r11d,(%r9)               ; 19.29%
597:  mov   0x0(%r13),%r13d               ;  0.10%  u32 load
      mov   %r13d,-0x4(%r14)              ;  0.24%  store hashed[j]
      cmp   %r14,%r12                     ;  0.78%
```

`SIMD_FSL_Search` gather:

```
      movzbl (%r11),%r15d                 ;  0.10%
      movzbl 0x1(%r11),%ecx               ;  0.05%
      movzbl -0x1(%r11,%rcx,1),%r12d      ;  0.05%
360:  lock xadd %edi,(%r9)                ; 46.85%
1dd:  cmp/movabs $0x100000001             ; 47.33% + 3.37%
```

Per-row path:

```
vHashName(v+j)
  bounds-check vs page range
  shared_ptr copy      -> lock xadd +1
  memcpy 4 bytes
  shared_ptr destroy   -> lock xadd -1, unique check, maybe _M_release_last_use_cold
```

~63 cycles/row (fnv1a), ~84 (fsl), for 4 resident bytes.

## Methods

```
SIMD_fnv1a_Search                          12.77 ms/1M   3 captures
  for v in steps of 16:
      load hash_name[v..v+16] -> __m512i
      mask = cmpeq_epi32(target_hash, vec)
      while mask: j=ctz(mask); memcmp confirm; mask &= mask-1
  tail: scalar memcmp

SIMD_FSL_Search                            16.81 ms/1M   3275 captures
  for v in steps of 64:
      gather byte[0], byte[1], byte[len-1] of 64 names -> 3 scratch arrays
      mask = cmpeq8(f) & cmpeq8(s) & cmpeq8(l)
      while mask: j=ctz(mask); memcmp confirm; mask &= mask-1
  tail: scalar memcmp

NoSearch                                   16.38 ms/1M
  for v in 0..N: touch(name[v].front())

ConstSearch                                20.21 ms/1M
  for v in 0..N: memcmp(target, name[v], name_len)

O1Search                                   1.20 s
  index = loadIndex("users.idx")
  rows  = index[fnv1a(target)]     # 226.79 us
```


TODO:
- std::array<char, 5> instead of string as name
- consumer ID
- gpu slice deconding
- sharding: `SHARD_MAP(hash % BIG_N) -> shard id` (`_ % BIG_N` is here to scale; if new shard necessary, we can just cut one into subs, and update the lookup table; BIG_N > Shards)
