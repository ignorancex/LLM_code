import argparse
import os

VHEADER = os.environ.get("VHEADER", "#include <immintrin.h>")
VTYPE = os.environ.get("VTYPE", "__m256")
VLEN = int(os.environ.get("VLEN", "8"))
VLOAD = os.environ.get("VLOAD", "_mm256_loadu_ps")
VSTORE = os.environ.get("VSTORE", "_mm256_storeu_ps")
VFMA = os.environ.get("VFMA", "_mm256_fmadd_ps")
VFMS = os.environ.get("VFMS", "_mm256_fnmadd_ps")
VDUP = os.environ.get("VDUP", "_mm256_set1_ps")
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

def gen(dim: int, g: tuple[int, ...], unroll: int) -> str:
    B = VLEN * unroll
    n_blades = 2**dim
    to_bin = {
        -1: "11",
        0: "00",
        1: "01",
    }
    s = f"void conv_opt2_{dim}d"
    for d in range(dim):
        s += f"_{to_bin[g[d]]}"
    s += f"(int n_batches, int in_channels, int d1{', int d2' if dim >= 2 else ''}{', int d3' if dim >= 3 else ''}, int out_channels, int filter_size, float* weight, float* bias, float* input, float* output) {{"
    s += "\nint filter_mem = filter_size" + "*filter_size" * (dim - 1) + ";"
    s += f"\nfloat* x = (float*) malloc(n_batches * in_channels * d1 {'* d2' if dim >= 2 else ''}{'* d3' if dim >= 3 else ''} * {n_blades} * sizeof(float));"
    s += f"""
for(int batch_b=0;batch_b<n_batches/{B};++batch_b) {{
for(int b=0;b<{B};++b) {{
for(int in_channel=0;in_channel<in_channels;++in_channel) {{"""
    for k in range(1, dim+1):
        s += f"\nfor(int id{k}=0;id{k}<d{k};++id{k}) {{"
    for k in range(n_blades):
        if dim == 1:
            s += f"\nx[in_channel*d1*n_batches*2 + id1*n_batches*2 + batch_b*{B}*2 + b + {k}*{B}] = "
            s += f"input[batch_b*{B}*in_channels*d1*2 + b*in_channels*d1*2 + in_channel*d1*2 + id1*2 + {k}];"
        elif dim == 2:
            s += f"\nx[in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*{B}*4 + b + {k}*{B}] = "
            s += f"input[batch_b*{B}*in_channels*d1*d2*4 + b*in_channels*d1*d2*4 + in_channel*d1*d2*4 + id1*d2*4 + id2*4 + {k}];"
        elif dim == 3:
            s += f"\nx[in_channel*d1*d2*d3*n_batches*8 + id1*d2*d3*n_batches*8 + id2*d3*n_batches*8 + id3*n_batches*8 + batch_b*{B}*8 + b + {k}*{B}] = "
            s += f"input[batch_b*{B}*in_channels*d1*d2*d3*8 + b*in_channels*d1*d2*d3*8 + in_channel*d1*d2*d3*8 + id1*d2*d3*8 + id2*d3*8 + id3*8 + {k}];"
    s += "\n" + r"}" * (dim+3)

    s += f"\nfloat* kernel=(float*) malloc({n_blades} * out_channels * in_channels * filter_mem * sizeof(float));"
    s += """
for(int out_channel=0;out_channel<out_channels;++out_channel) {
for(int in_channel=0;in_channel<in_channels;++in_channel) {"""
    for k in range(1, dim+1):
        s += f"\nfor(int u{k}=0;u{k}<filter_size;++u{k}) {{"
    for k in range(n_blades):
        if dim == 1:
            s += f"\nkernel[out_channel*in_channels*filter_mem*2 + in_channel*filter_mem*2 + u1*2 + {k}] = "
            s += f"weight[{k} * out_channels * in_channels * filter_mem + out_channel * in_channels * filter_mem + in_channel * filter_mem + u1];"
        elif dim == 2:
            s += f"\nkernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u1*filter_size*4 + u2*4 + {k}] = "
            s += f"weight[{k} * out_channels * in_channels * filter_mem + out_channel * in_channels * filter_mem + in_channel * filter_mem + u1 * filter_size + u2];"
        elif dim == 3:
            s += f"\nkernel[out_channel*in_channels*filter_mem*8 + in_channel*filter_mem*8 + u1*filter_size*filter_size*8 + u2*filter_size*8 + u3*8 + {k}] = "
            s += f"weight[{k} * out_channels * in_channels * filter_mem + out_channel * in_channels * filter_mem + in_channel * filter_mem + u1 * filter_size * filter_size + u2 * filter_size + u3];"
    s += "\n" + r"}" * (dim+2)

    s += "\nint out_d1 = d1 - filter_size + 1;"
    if dim >= 2:
        s += "\nint out_d2 = d2 - filter_size + 1;"
    if dim >= 3:
        s += "\nint out_d3 = d3 - filter_size + 1;"

    s += f"\nfloat* y = (float*) malloc(out_channels*out_d1{'*out_d2' if dim>=2 else ''}{'*out_d3' if dim >= 3 else ''}*n_batches*{n_blades}*sizeof(float));"
    s += """
for(int out_channel=0;out_channel<out_channels;++out_channel) {"""
    for k in range(1, dim+1):
        s += f"\nfor(int id{k}=0;id{k}<out_d{k};++id{k}) {{"
    s += f"\nfor(int batch_b=0;batch_b<n_batches/{B};++batch_b) {{"
    s += f"\nfor(int b=0;b<{B};++b) {{"
    for k in range(n_blades):
        if dim == 1:
            s += f"\ny[out_channel*out_d1*n_batches*2 + id1*n_batches*2 + batch_b*{B}*2 + b + {k}*{B}] = "
        elif dim == 2:
            s += f"\ny[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*{B}*4 + b + {k}*{B}] = "
        elif dim == 3:
            s += f"\ny[out_channel*out_d1*out_d2*out_d3*n_batches*8 + id1*out_d2*out_d3*n_batches*8 + id2*out_d3*n_batches*8 + id3*n_batches*8 + batch_b*{B}*8 + b + {k}*{B}] = "
        s += f"bias[{k}*out_channels + out_channel];"
    s += "\n" + r"}" * (dim+3)

    s += """
for(int out_channel=0;out_channel<out_channels;++out_channel) {
for(int in_channel=0;in_channel<in_channels;++in_channel) {"""
    for k in range(1, dim+1):
        s += f"\nfor(int id{k}=0;id{k}<d{k};++id{k}) {{"
    s += f"\nfor(int batch_b=0;batch_b<n_batches/{B};++batch_b) {{"
    for k in range(n_blades):
        for i in range(unroll):
            if dim == 1:
                s += f"\n{VTYPE} vec{k}_{i} = {VLOAD}(x + in_channel*d1*n_batches*2 + id1*n_batches*2 + batch_b*{B}*2 + {k}*{B} + {i}*{VLEN});"
            elif dim == 2:
                s += f"\n{VTYPE} vec{k}_{i} = {VLOAD}(x + in_channel*d1*d2*n_batches*4 + id1*d2*n_batches*4 + id2*n_batches*4 + batch_b*{B}*4 + {k}*{B} + {i}*{VLEN});"
            elif dim == 3:
                s += f"\n{VTYPE} vec{k}_{i} = {VLOAD}(x + in_channel*d1*d2*d3*n_batches*8 + id1*d2*d3*n_batches*8 + id2*d3*n_batches*8 + id3*n_batches*8 + batch_b*{B}*8 + {k}*{B} + {i}*{VLEN});"

    for k in range(1, dim+1):
        s += f"\nfor(int u{k}=id{k}+filter_size<=d{k}?0:id{k}-d{k}+filter_size;u{k}<filter_size && u{k}<=id{k};++u{k}) {{"
    
    for k in range(1, dim+1):
        s += f"\nint od{k} = id{k} - u{k};"
    for k in range(n_blades):
        if dim == 1:
            s += f"\n{VTYPE} k{k} = {VDUP}(kernel[out_channel*in_channels*filter_mem*2 + in_channel*filter_mem*2 + u1*2 + {k}]);"
        elif dim == 2:
            s += f"\n{VTYPE} k{k} = {VDUP}(kernel[out_channel*in_channels*filter_mem*4 + in_channel*filter_mem*4 + u1*filter_size*4 + u2*4 + {k}]);"
        elif dim == 3:
            s += f"\n{VTYPE} k{k} = {VDUP}(kernel[out_channel*in_channels*filter_mem*8 + in_channel*filter_mem*8 + u1*filter_size*filter_size*8 + u2*filter_size*8 + u3*8 + {k}]);"
    for k in range(n_blades):
        for i in range(unroll):
            if dim == 1:
                s += f"\n{VTYPE} out{k}_{i} = {VLOAD}(y + out_channel*out_d1*n_batches*2 + od1*n_batches*2 + batch_b*{B}*2 + {k}*{B} + {i}*{VLEN});"
            elif dim == 2:
                s += f"\n{VTYPE} out{k}_{i} = {VLOAD}(y + out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*{B}*4 + {k}*{B} + {i}*{VLEN});"
            elif dim == 3:
                s += f"\n{VTYPE} out{k}_{i} = {VLOAD}(y + out_channel*out_d1*out_d2*out_d3*n_batches*8 + od1*out_d2*out_d3*n_batches*8 + od2*out_d3*n_batches*8 + od3*n_batches*8 + batch_b*{B}*8 + {k}*{B} + {i}*{VLEN});"
    for z in range(n_blades):
        for i in range(unroll):
            for x in range(n_blades):
                for k in range(n_blades):
                    _z, f = mult_blades(x, k, dim, g)
                    if _z != z or f == 0:
                        continue
                    if VSET == "NEON":
                        if f == -1:
                            s += f"\nout{z}_{i} = {VFMS}(out{z}_{i}, vec{x}_{i}, k{k});"
                        elif f == 1:
                            s += f"\nout{z}_{i} = {VFMA}(out{z}_{i}, vec{x}_{i}, k{k});"
                    elif VSET == "AVX":
                        if f == -1:
                            s += f"\nout{z}_{i} = {VFMS}(vec{x}_{i}, k{k}, out{z}_{i});"
                        elif f == 1:                                     
                            s += f"\nout{z}_{i} = {VFMA}(vec{x}_{i}, k{k}, out{z}_{i});"
    for k in range(n_blades):
        for i in range(unroll):
            if dim == 1:
                s += f"\n{VSTORE}(y + out_channel*out_d1*n_batches*2 + od1*n_batches*2 + batch_b*{B}*2 + {k}*{B} + {i}*{VLEN}, out{k}_{i});"
            elif dim == 2:
                s += f"\n{VSTORE}(y + out_channel*out_d1*out_d2*n_batches*4 + od1*out_d2*n_batches*4 + od2*n_batches*4 + batch_b*{B}*4 + {k}*{B} + {i}*{VLEN}, out{k}_{i});"
            elif dim == 3:
                s += f"\n{VSTORE}(y + out_channel*out_d1*out_d2*out_d3*n_batches*8 + od1*out_d2*out_d3*n_batches*8 + od2*out_d3*n_batches*8 + od3*n_batches*8 + batch_b*{B}*8 + {k}*{B} + {i}*{VLEN}, out{k}_{i});"
    s += "\n" + r"}" * dim + r"}" * (dim+3)

    s += f"""
for(int batch_b=0;batch_b<n_batches/{B};++batch_b) {{
for(int b=0;b<{B};++b) {{
for(int out_channel=0;out_channel<out_channels;++out_channel) {{"""
    for k in range(1, dim+1):
        s += f"\nfor(int id{k}=0;id{k}<out_d{k};++id{k}) {{"
    for k in range(n_blades):
        if dim == 1:
            s += f"\noutput[batch_b*{B}*out_channels*out_d1*2 + b*out_channels*out_d1*2 + out_channel*out_d1*2 + id1*2 + {k}] = "
            s += f"y[out_channel*out_d1*n_batches*2 + id1*n_batches*2 + batch_b*{B}*2 + b + {k}*{B}];"
        elif dim == 2:
            s += f"\noutput[batch_b*{B}*out_channels*out_d1*out_d2*4 + b*out_channels*out_d1*out_d2*4 + out_channel*out_d1*out_d2*4 + id1*out_d2*4 + id2*4 + {k}] = "
            s += f"y[out_channel*out_d1*out_d2*n_batches*4 + id1*out_d2*n_batches*4 + id2*n_batches*4 + batch_b*{B}*4 + b + {k}*{B}];"
        elif dim == 3:
            s += f"\noutput[batch_b*{B}*out_channels*out_d1*out_d2*out_d3*8 + b*out_channels*out_d1*out_d2*out_d3*8 + out_channel*out_d1*out_d2*out_d3*8 + id1*out_d2*out_d3*8 + id2*out_d3*8 + id3*8 + {k}] = "
            s += f"y[out_channel*out_d1*out_d2*out_d3*n_batches*8 + id1*out_d2*out_d3*n_batches*8 + id2*out_d3*n_batches*8 + id3*n_batches*8 + batch_b*{B}*8 + b + {k}*{B}];"
    s += "\n" + r"}" * (dim+3)

    s += "\nfree(x);"
    s += "\nfree(y);"
    s += "\nfree(kernel);"
    s += "\n}"
    return s


def main(args):
    _unrolls = str(args.unroll).split(",")
    unrolls = {
        1: int(_unrolls[0]),
        2: int(_unrolls[1]),
        3: int(_unrolls[2])
    }
    s = VHEADER
    s += "\n#include <stdlib.h>"
    for dim in range(1, 4):
        for g in range(3**dim):
            g_list = []
            for i in range(dim):
                g_list.append((g // (3**i)) % 3 - 1)
            if not any(g_list):
                continue
            s += "\n" + gen(dim, tuple(g_list), unrolls[dim]) + "\n"
    with open("conv_opt2.h", "w") as f:
        f.write(s)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate convolutional layer code.")
    parser.add_argument(
        "--unroll",
        type=str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)