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

def main():
    s = "double estimate_flops(int dim, float* g, int batches, int n_blades, int in_channels, int out_channels, int d1, int d2, int d3, int filter_size) {"
    for dim in range(1, 4):
        s += "\n    if (dim == " + str(dim) + ") {"
        flag = False
        for g in range(3**dim):
            g_list = []
            for i in range(dim):
                g_list.append((g // (3**i)) % 3 - 1)
            if flag == False:
                s += "\n        if ("
                flag = True
            else:
                s += "\n        else if ("
            for i in range(dim):
                if i > 0:
                    s += " && "
                if g_list[i] == 1:
                    s += "g[" + str(i) + "] == 1."
                elif g_list[i] == -1:
                    s += "g[" + str(i) + "] == -1."
                else:
                    s += "g[" + str(i) + "] == 0."
            s += ")"
            cnt = 0
            for z in range(2**dim):
                for x in range(2**dim):
                    for y in range(2**dim):
                        dz, f = mult_blades(x, y, dim, g_list)
                        if dz == z and f != 0:
                            cnt += 1
            s += "\n            return 2.*" + str(cnt) + f"* out_channels * in_channels * (d1-filter_size+1) {'* (d2-filter_size+1)' if dim >= 2 else ''} {'* (d3-filter_size+1)' if dim >= 3 else ''} * batches * filter_size {'* filter_size' if dim >= 2 else ''}{'* filter_size' if dim >= 3 else ''};"
        s += "\n    }"
    s += "\n" + r'    fprintf(stderr, "Unsupported dimension: %d\n", dim);'
    s += "\n" + r'    exit(EXIT_FAILURE);'
    s += "\n}"
    print(s)
            

if __name__ == "__main__":
    main()