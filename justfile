set shell := ["bash", "-uc"]

name       := "fastsearch"
src        := "FastSearch.cpp"
data       := "data/search"
bindir     := "bin"

root_flags := `root-config --cflags --libs`

o_base := "-O3 -march=native -mtune=native -mprefer-vector-width=512"
o_math := " -ffast-math -fno-math-errno -fomit-frame-pointer"
o_loop := " -funroll-loops -fprefetch-loop-arrays -falign-functions=32 -falign-loops=32 -falign-jumps=32"
o_code := " -fno-plt -fno-semantic-interposition -fno-stack-protector -fvect-cost-model=unlimited"
o_link := " -flto -Wl,-O3 -Wl,--as-needed -pipe"

hpc := o_base + o_math + o_loop + o_code + o_link


default:
    @just --list

info:
    @echo "compiler : $$(g++ --version | head -1)"
    @echo "root     : $$(root-config --version)"
    @lscpu | grep -E "Model name|^Socket|^NUMA" || true

build: build-o1 build-fsl build-fnv1a build-const build-nosearch
    @echo "DONE all variants built into {{bindir}}/"
    @echo ""

build-o1:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++20 {{hpc}} -DRUN_O1SEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_o1
    @echo "DONE {{bindir}}/{{name}}_o1    [O(1) unordered_map hash-index search]"
    @echo ""

build-fsl:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++20 {{hpc}} -DRUN_SIMDFSLSEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_fsl
    @echo "DONE {{bindir}}/{{name}}_simd  [SIMD FSL]"
    @echo ""

build-fnv1a:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++20 {{hpc}} -DRUN_SIMDFNV1ASEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_fnv1a
    @echo "DONE {{bindir}}/{{name}}_simd  [AVX512 + fnv1a]"
    @echo ""

build-const:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++20 {{hpc}} -DRUN_CONSTSEACH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_const
    @echo "DONE {{bindir}}/{{name}}_static [const name size memcmp search]"
    @echo ""

build-nosearch:
    mkdir -p {{bindir}} {{data}}
    g++ -std=c++20 {{hpc}} -DRUN_NOSEARCH=1 {{src}} {{root_flags}} -o {{bindir}}/{{name}}_nosearch
    @echo "DONE {{bindir}}/{{name}}_nosearch [no search, cost baseline]"
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


run-const: build-const
    @echo "running const-len memcmp search"
    ./{{bindir}}/{{name}}_const
    @echo ""

run-nosearch: build-nosearch
    @echo "running baseline cost of iteration"
    ./{{bindir}}/{{name}}_nosearch
    @echo ""


run: run-o1 run-fsl run-fnv1a run-const run-nosearch

clean:
    rm -rf {{bindir}} {{data}}
    @echo "cleaned {{bindir}}/ and {{data}}/"
    @echo ""
