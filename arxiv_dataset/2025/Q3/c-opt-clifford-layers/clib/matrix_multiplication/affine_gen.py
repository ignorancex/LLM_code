from typing import Callable
import os

VHEADER = os.environ.get("VHEADER", "#include <immintrin.h>")
VTYPE = os.environ.get("VTYPE", "__m256")
VLEN = int(os.environ.get("VLEN", "8"))
VLOAD = os.environ.get("VLOAD", "_mm256_loadu_ps")
VFMA = os.environ.get("VFMA", "_mm256_fmadd_ps")
VFMS = os.environ.get("VFMS", "_mm256_fnmadd_ps")
VSET = os.environ.get("VSET", "AVX")

def mult_blades(x: int, y: int, dim: int, g: tuple[int, ...]) -> tuple[int, int]:
    l = list(range(2**dim))
    l = sorted(l, key=lambda i: (i.bit_count(), i))
    x = l.index(x)
    y = l.index(y)
    z = x ^ y
    f = 1
    for k in range(dim):
        x0 = x & 1
        y0 = y & 1
        x >>= 1
        y >>= 1
        if y0:
            f *= (-1)**(x.bit_count())
            if x0:
                f *= g[k]
    return l[z], f

def gen(dim: int, g: tuple[int, ...]) -> str:
    to_bin = {
        -1: "11",
        0: "00",
        1: "01",
    }
    s = f"void affine_forward_opt2_{dim}d"
    for d in range(dim):
        s += f"_{to_bin[g[d]]}"
    s += "(int in_channels, int out_channels, float* weight, float* bias, float* x, int batches, float* output){"
    n_blades = 2**dim
    s += "\nfor(int i=0;i<batches;++i){"
    for bitset_out in range(n_blades):
        s += "\n  for(int j=0;j<out_channels;++j){"
        s += f"\n    output[i*{n_blades}*out_channels + {n_blades}*j + {bitset_out}] = bias[j + {bitset_out}*out_channels];"
        s += "\n    for(int k=0;k<in_channels;++k){"
        for bitset_in in range(n_blades):
            for bitset_weight in range(n_blades):
                z, f = mult_blades(bitset_in, bitset_weight, dim, g)
                if z == bitset_out and f != 0:
                    op = "+=" if f == 1 else "-="
                    s += f"\n      output[i*{n_blades}*out_channels + {n_blades}*j + {bitset_out}] {op} x[i*{n_blades}*in_channels + {n_blades}*k + {bitset_in}] * weight[{bitset_weight}*in_channels*out_channels + j*in_channels + k];"
        s += "\n    }"
        s += "\n  }"
    s += "\n}\n}"
    return s

def genmm(dim: int, g: tuple[int, ...]) -> str:
    to_bin = {
        -1: "11",
        0: "00",
        1: "01",
    }
    s = f"void affine_forward_opt3_{dim}d"
    for d in range(dim):
        s += f"_{to_bin[g[d]]}"
    s += "(int in_channels, int out_channels, float* weight, float* bias, float* x, int batches, float* output){"
    n_blades = 2**dim
    s += "\nfor(int i=0;i<batches;++i){"
    for bitset_out in range(n_blades):
        s += "\n  for(int j=0;j<out_channels;++j){"
        s += f"\n    {VTYPE} acc;"
        for k in range(VLEN):
            s += f"((float*)(&acc))[{k}] = "
        s += f"0; ((float*)(&acc))[0] = bias[j + {bitset_out}*out_channels];"
        s += "\n    int k;"
        for bitset_in in range(n_blades):
            s += f"\n    for(k=0;k<=in_channels - {VLEN};k+={VLEN}){{"
            for bitset_weight in range(n_blades):
                z, f = mult_blades(bitset_in, bitset_weight, dim, g)
                if z == bitset_out and f != 0:
                    s += f"\n      {VTYPE} vx = {VLOAD}(x + i*{n_blades}*in_channels + {bitset_in}*in_channels + k);"
                    s += f"\n      {VTYPE} vw = {VLOAD}(weight + {bitset_weight}*in_channels*out_channels + j*in_channels + k);"
                    if VSET == "NEON":
                        if f == 1:
                            s += f"\n      acc = {VFMA}(acc, vx, vw);"
                        else:
                            s += f"\n      acc = {VFMS}(acc, vx, vw);"
                    elif VSET == "AVX":
                        if f == 1:
                            s += f"\n      acc = {VFMA}(vx, vw, acc);"
                        else:
                            s += f"\n      acc = {VFMS}(vx, vw, acc);"
            s += "\n    }"
            s += "\n    for(;k<in_channels;++k){"
            for bitset_weight in range(n_blades):
                z, f = mult_blades(bitset_in, bitset_weight, dim, g)
                if z == bitset_out and f != 0:
                    op = "+=" if f == 1 else "-="
                    s += f"\n      ((float*)(&acc))[0] {op} x[i*{n_blades}*in_channels + {bitset_in}*in_channels + k] * weight[{bitset_weight}*in_channels*out_channels + j*in_channels + k];"
            s += "\n    }"
        s += f"\n    output[i*{n_blades}*out_channels + {bitset_out}*out_channels + j] = ((float*)(&acc))[0]"
        for k in range(1, VLEN):
            s += f" + ((float*)(&acc))[{k}]"
        s += ";"
        s += "\n  }"
    s += "\n}\n}"
    return s

def gen_write(fun: Callable, file_name: str):
    s = ""
    s += VHEADER + "\n\n"
    for dim in range(1, 4):
        for g in range(3**dim):
            g_list = []
            for i in range(dim):
                g_list.append((g // (3**i)) % 3 - 1)
            if not any(g_list):
                continue
            s += fun(dim, tuple(g_list)) + "\n\n"
    with open(file_name, "w") as f:
        f.write(s)

def main():
    gen_write(gen, "affine_forward_opt2.h")
    gen_write(genmm, "affine_forward_opt3.h")

def test():
    print(mult_blades(3, 3, 3, (-1, -1, -1)))

if __name__ == "__main__":
    # test()
    main()