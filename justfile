set shell := ["bash", "-uc"]

name       := "fastsearch"
src        := "FastSearch.cpp"
data       := "data/search"
bindir     := "bin"

# Strip ROOT's own -std flag so the project standard (-std=c++23) wins.
root_flags := `root-config --cflags | sed 's/ -std=[^ ]*//'` + " " + `root-config --libs` + " -llikwid"

o_base := "-O3 -march=native -DLIKWID_PERFMON"
o_math := " -ffast-math"
o_loop := " -funroll-loops -fprefetch-loop-arrays -falign-functions=32 -falign-loops=32 -falign-jumps=32"
o_code := " -fno-plt -fno-semantic-interposition -fno-stack-protector -fvect-cost-model=unlimited"
o_link := " -flto -Wl,-O3 -Wl,--as-needed -pipe"

# DWARF debug info: "0"/"off"/"none" disables; otherwise adds -g -gdwarf-<N> (default 4)
dwarf := env_var_or_default("DWARF", "4")

dwarf_flags := if dwarf == "0" {
    ""
} else if dwarf == "off" {
    ""
} else if dwarf == "none" {
    ""
} else {
    " -g -gdwarf-" + dwarf
}

hpc := o_base + o_math + o_loop + o_code + o_link + dwarf_flags


default:
    @just --list

info:
    @echo "compiler : $$(g++ --version | head -1)"
    @echo "root     : $$(root-config --version)"
    @lscpu | grep -E "Model name|^Socket|^NUMA" || true

build: build-o1 build-fsl build-fnv1a build-bin build-treeter build-treebin build-const build-nosearch build-nosearchhash
    @echo "DONE all variants built into {{bindir}}/"
    @echo ""

build-o1:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_O1SEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_o1
    @echo "DONE {{bindir}}/{{name}}_o1    [O(1) unordered_map hash-index search]"
    @echo ""

build-fsl:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_SIMDFSLSEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_fsl
    @echo "DONE {{bindir}}/{{name}}_simd  [SIMD FSL]"
    @echo ""

build-fnv1a:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_SIMDFNV1ASEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_fnv1a
    @echo "DONE {{bindir}}/{{name}}_simd  [AVX512 + fnv1a]"
    @echo ""

build-bin:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_BINARYSEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_bin
    @echo "DONE {{bindir}}/{{name}}_bin [binary search]"
    @echo ""

build-treeter:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_TREETERNARYSEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_ter
    @echo "DONE {{bindir}}/{{name}}_treeter [tree ternary search]"
    @echo ""


build-treebin:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_TREEBINARYSEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_treebin
    @echo "DONE {{bindir}}/{{name}}_treebin [precomputed tree binary search]"
    @echo ""

build-const:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_CONSTSEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_const
    @echo "DONE {{bindir}}/{{name}}_static [const name size memcmp search]"
    @echo ""

build-nosearch:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_NOSEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_nosearch
    @echo "DONE {{bindir}}/{{name}}_nosearch [no search char[5], cost baseline]"
    @echo ""

build-nosearchhash:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++23 {{hpc}} -DRUN_NOSEARCHHASH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_nosearchhash
    @echo "DONE {{bindir}}/{{name}}_nosearchhash [no search uint32_t, cost baseline]"
    @echo ""


run-o1: build-o1
    @echo "running O(1) search"
    ./{{bindir}}/{{name}}_o1
    @echo ""

run-fsl: build-fsl
    @echo "running AVX512 + FSL search"
    ./{{bindir}}/{{name}}_fsl
    @echo ""


run-fnv1a: build-fnv1a
    @echo "running AVX512 + fnv1a search"
    ./{{bindir}}/{{name}}_fnv1a
    @echo ""
run-bin: build-bin
    @echo "running binary search"
    ./{{bindir}}/{{name}}_bin
    @echo ""



run-treebin: build-treebin
    @echo "running tree binary search"
    ./{{bindir}}/{{name}}_treebin
    @echo ""
run-treeter: build-treeter
    @echo "running tree ternary search"
    ./{{bindir}}/{{name}}_ter
    @echo ""


run-const: build-const
    @echo "running const-len memcmp search"
    ./{{bindir}}/{{name}}_const
    @echo ""

run-nosearch: build-nosearch
    @echo "running baseline cost of iteration over char[5]"
    ./{{bindir}}/{{name}}_nosearch
    @echo ""

run-nosearchhash: build-nosearchhash
    @echo "running baseline cost of iteration over uint32_t"
    ./{{bindir}}/{{name}}_nosearchhash
    @echo ""


run: run-o1 run-fsl run-fnv1a run-bin run-treeter run-treebin run-const run-nosearch run-nosearchhash

clean:
    rm -rf {{bindir}} {{data}}
    @echo "cleaned {{bindir}}/ and {{data}}/"
    @echo ""
