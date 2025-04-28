import taichi as ti
import numpy as np
import taichi.math as tm
import math
import time

@ti.func
def sub(a, b):
    ans = 0.0
    if tm.isinf(a) and tm.isinf(b):
        ans = -math.inf
    else:
        maxim = tm.max(a, b)
        minim = tm.min(a, b)
        ans = maxim + tm.log(1.0 - tm.exp(minim - maxim))
    return ans

def add_py(a, b):
    if math.isinf(a) and math.isinf(b):
        return -math.inf
    maxim = max(a, b)
    minim = min(a, b)
    return maxim + math.log(1.0 + math.exp(minim - maxim))

@ti.func
def add(a, b):
    ans = 0.0
    if tm.isinf(a) and tm.isinf(b):
        ans = -math.inf
    else:
        maxim = tm.max(a, b)
        minim = tm.min(a, b)
        ans = maxim + tm.log(1.0 + tm.exp(minim - maxim))
    #if tm.isnan(ans):
    #    print(f"Error with {a} + {b}")
    return ans

@ti.func
def n(log_p, n):
    return (sub(0.0, log_p)) * n

@ti.func
def f(log_p, n):
    return sub(0.0, sub(0.0, log_p) * n)

m_min = 2
m_max = 8

m_table = 2
table_size = 5

log_multiple = 2.5

restrict_support = False

#ti.init(arch=ti.gpu, default_fp=ti.f64)
ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.kernel
def init(log_p: float, n2: ti.template()):
    n2[0, 1] = log_p
    n2[1, 1] = log_p + n(log_p, 1)
    n2[2, 1] = log_p + n(log_p, 2)
    n2[3, 1] = log_p + n(log_p, 3)

@ti.kernel
def modified(p: float, s: int,
             o: ti.template(),
             p1: ti.template(),
             p2: ti.template(),
             p3: ti.template(),
             supp_lo: int,
             supp_hi: int):
    zero = -math.inf
    one = 0.0
    q = sub(one, log_p)
    for aa in range(supp_lo, supp_hi):
        b = s - aa
        #print(f"{p}, {s}: p2[1, {aa} - 1]={p2[1, aa - 1]}, p2[5, {aa} - 1]={p2[5, aa - 1]},p2[6, {aa} - 1]={p2[6, aa - 1]},f(p, {b})={f(p, b)},p1[0, {aa} - 1]={p1[0, aa - 1]}")
        o[0, aa] = add(p + add(add(p2[1, aa - 1], p2[5, aa - 1]), p2[6, aa - 1]),
                  f(p, b) + p1[0, aa - 1])
        o[1, aa] = add(add(q + f(p, aa) + p1[1, aa],
                  n(p, b) + o[0, aa]),
                  p + p2[2, aa - 1])
        o[5, aa] = add(q + f(p, aa) + p1[5, aa], p + p2[4, aa - 1])
        if aa < 3:
            o[6, aa] = zero
        else:
            o[6, aa] = add(q + f(p, b) + p1[6, aa - 1], p + p + p3[3, aa - 2])
        o[2, aa] = add(add(add(q + f(p, b) + p1[2, aa - 1],
                  p + q + p2[3, aa - 1]),
                  q + n(p, aa) + o[1, aa]),
                  q + n(p, b) + o[6, aa])
        o[4, aa] = add(add(q + f(p, b) + p1[4, aa - 1],
                  p + q + p2[3, aa - 1]),
                  q + n(p, aa) + o[5, aa])
        o[3, aa] = add(add(q + q + f(p, aa) + p1[3, aa],
                  q + n(p, b) + o[2, aa]),
                  q + n(p, b) + o[4, aa])
        #print(f"s = {s}, o[0, {aa}] = {o[0, aa]}")

def fill_diagonal(s, current, table):
    for k in range(0, 7):
        for j in range(1, s):
            if j < table_size:
                if s - j < table_size:
                    table[k][j][s - j] = math.exp(current[k, j])

def clip(x, lo, hi):
    return min(max(x, lo), hi)

def update_supp(old_lo, old_hi, s, current):
    if not restrict_support:
        return 1, s
    res_lo = 1
    res_hi = s
    for lo in range(old_lo - 20, old_lo + 20):
        if lo >=0:
            if lo < s:
                if not math.isinf(current[0, lo]):
                    res_lo = lo
                    break
    for hi in range(old_hi + 20, old_lo - 20, -1):
        if hi >=0:
            if hi < s:
                if not math.isinf(current[0, hi]):
                    res_hi = hi
                    break
    return clip(res_lo - 20, 1, s), clip(res_hi + 20, 1, s)

for m in range(m_min, m_max + 1):
    start_time = time.time()
    p_float = 2**(-m)
    a = math.floor(log_multiple * math.log(1.0 / p_float) / p_float)

    #print(f"a = {a}")

    log_p =  tm.log(p_float)

    n0 = ti.field(ti.f64)
    n1 = ti.field(ti.f64)
    n2 = ti.field(ti.f64)
    n3 = ti.field(ti.f64)

    table = np.zeros((7, table_size, table_size))

    ti.root.dense(ti.i, 7).dense(ti.j, a).place(n0)
    ti.root.dense(ti.i, 7).dense(ti.j, a).place(n1)
    ti.root.dense(ti.i, 7).dense(ti.j, a).place(n2)
    ti.root.dense(ti.i, 7).dense(ti.j, a).place(n3)

    n0.fill(-math.inf)
    n1.fill(-math.inf)
    n2.fill(-math.inf)
    n3.fill(-math.inf)

    init(log_p, n2)
    if m == m_table:
        fill_diagonal(2, n2, table)

    supp_lo = 1
    supp_hi = 3
    for s in range(3, a):
        if s % 4 == 0:
            supp_lo, supp_hi = update_supp(supp_lo, supp_hi, s, n0)
            modified(log_p, s, n0, n3, n2, n1, supp_lo, supp_hi)
            if m == m_table:
                fill_diagonal(s, n0, table)
        if s % 4 == 1:
            supp_lo, supp_hi = update_supp(supp_lo, supp_hi, s, n1)
            modified(log_p, s, n1, n0, n3, n2, supp_lo, supp_hi)
            if m == m_table:
                fill_diagonal(s, n1, table)
        if s % 4 == 2:
            supp_lo, supp_hi = update_supp(supp_lo, supp_hi, s, n2)
            modified(log_p, s, n2, n1, n0, n3, supp_lo, supp_hi)
            if m == m_table:
                fill_diagonal(s, n2, table)
        if s % 4 == 3:
            supp_lo, supp_hi = update_supp(supp_lo, supp_hi, s, n3)
            modified(log_p, s, n3, n2, n1, n0, supp_lo, supp_hi)
            if m == m_table:
                fill_diagonal(s, n3, table)

    acc = -math.inf
    for l in range(a):
        # for k in range(7):
            k = 0
            if s % 4 == 0:
                acc = add_py(acc, n0[k, l])
            if s % 4 == 1:
                acc = add_py(acc, n1[k, l])
            if s % 4 == 2:
                acc = add_py(acc, n2[k, l])
            if s % 4 == 3:
                acc = add_py(acc, n3[k, l])

    print(f"p = {p_float}, size = {a}, -p log(s) = {-p_float * acc}, m = {m}, "
          + f"t = {time.time() - start_time}")
    file = open(f'result_{m:02d}.txt', 'w')
    file.write(f"p = {p_float}, size = {a}, -p log(s) = {-p_float * acc}, m = {m}\n")
    file.close()

#print(table)
for k in range(0, 7):
    np.savetxt(f"table_{k}.csv", table[k][:][:], delimiter=",")
