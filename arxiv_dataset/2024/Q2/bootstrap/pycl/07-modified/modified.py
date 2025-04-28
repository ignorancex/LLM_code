import pyopencl as cl
import numpy as np
import math
import time

from dotenv import load_dotenv, dotenv_values
# loading variables from .env file
load_dotenv()

def sub(a, b):
    ans = 0.0
    if math.isinf(a) and math.isinf(b):
        ans = -math.inf
    else:
        maxim = max(a, b)
        minim = min(a, b)
        ans = maxim + math.log(1.0 - math.exp(minim - maxim))
    return ans

def add(a, b):
    if math.isinf(a) and math.isinf(b):
        return -math.inf
    maxim = max(a, b)
    minim = min(a, b)
    return maxim + math.log(1.0 + math.exp(minim - maxim))

def n(log_p, n):
    return (sub(0.0, log_p)) * n

m_min = 2
m_max = 8

m_table = 2
table_size = 5

log_multiple = 2.5

restrict_support = False

def fill_diagonal(s, current, table):
    for k in range(0, 7):
        for j in range(1, s):
            if j < table_size:
                if s - j < table_size:
                    table[k][j][s - j] = math.exp(current[k, j])

def clip(x, lo, hi):
    return min(max(x, lo), hi)

# def update_supp(old_lo, old_hi, s, current):
#     if not restrict_support:
#         return 1, s
#     res_lo = 1
#     res_hi = s
#     for lo in range(old_lo - 20, old_lo + 20):
#         if lo >=0:
#             if lo < s:
#                 if not math.isinf(current[0, lo]):
#                     res_lo = lo
#                     break
#     for hi in range(old_hi + 20, old_lo - 20, -1):
#         if hi >=0:
#             if hi < s:
#                 if not math.isinf(current[0, hi]):
#                     res_hi = hi
#                     break
#     return clip(res_lo - 20, 1, s), clip(res_hi + 20, 1, s)

# Read kernel code from external file
with open("modified.cl", "r") as file:
    kernel_code = file.read()

context = cl.create_some_context()
queue = cl.CommandQueue(context)
# Load and compile the kernel
program = cl.Program(context, kernel_code).build()


# Kernel code for applying modified rule
def modified(log_p, a, s, o, p1, p2, p3):
    program.modified(queue, (1, a), None, o, p1, p2, p3,
                     np.double(log_p), np.int32(a), np.int32(s))


for m in range(m_min, m_max + 1):
    start_time = time.time()
    p_float = 2**(-m)
    a = math.floor(log_multiple * math.log(1.0 / p_float) / p_float)

    log_p =  math.log(p_float)

    dimensions = (7, a)

    # Initialize input buffer
    n0 = np.full(dimensions, -np.inf, dtype=np.double)
    n0_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE
                          | cl.mem_flags.COPY_HOST_PTR, hostbuf=n0)
    n1 = np.full(dimensions, -np.inf, dtype=np.double)
    n1_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE
                          | cl.mem_flags.COPY_HOST_PTR, hostbuf=n1)
    n2 = np.full(dimensions, -np.inf, dtype=np.double)
    n2[0][1] = log_p
    n2[1][1] = log_p + n(log_p, 1)
    n2[2][1] = log_p + n(log_p, 2)
    n2[3][1] = log_p + n(log_p, 3)
    n2_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE
                          | cl.mem_flags.COPY_HOST_PTR, hostbuf=n2)
    n3 = np.full(dimensions, -np.inf, dtype=np.double)
    n3_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE
                          | cl.mem_flags.COPY_HOST_PTR, hostbuf=n3)

    table = np.zeros((7, table_size, table_size))

    if m == m_table:
        fill_diagonal(2, n2, table)

    # TODO: Fill up table
    for s in range(3, a):
        if s % 4 == 0:
            # supp_lo, supp_hi = update_supp(supp_lo, supp_hi, s, n0)
            modified(log_p, a, s, n0_buffer, n3_buffer, n2_buffer, n1_buffer) #, supp_lo, supp_hi)
            #if m == m_table:
            #fill_diagonal(s, n0_buffer, table)
        if s % 4 == 1:
            # supp_lo, supp_hi = update_supp(supp_lo, supp_hi, s, n1_buffer)
            modified(log_p, a, s, n1_buffer, n0_buffer, n3_buffer, n2_buffer) #, supp_lo, supp_hi)
            #if m == m_table:
            #fill_diagonal(s, n1_buffer, table)
        if s % 4 == 2:
            # supp_lo, supp_hi = update_supp(supp_lo, supp_hi, s, n2_buffer)
            modified(log_p, a, s, n2_buffer, n1_buffer, n0_buffer, n3_buffer) #, supp_lo, supp_hi)
            #if m == m_table:
            #fill_diagonal(s, n2_buffer, table)
        if s % 4 == 3:
            # supp_lo, supp_hi = update_supp(supp_lo, supp_hi, s, n3_buffer)
            modified(log_p, a, s, n3_buffer, n2_buffer, n1_buffer, n0_buffer) #, supp_lo, supp_hi)
            #if m == m_table:
            #fill_diagonal(s, n3_buffer, table)

    queue.finish()
    cl.enqueue_copy(queue, n0, n0_buffer).wait()
    cl.enqueue_copy(queue, n1, n1_buffer).wait()
    cl.enqueue_copy(queue, n2, n2_buffer).wait()
    cl.enqueue_copy(queue, n3, n3_buffer).wait()
    acc = -math.inf
    for l in range(a):
        # for k in range(7):
            k = 0
            if s % 4 == 0:
                acc = add(acc, n0[k][l])
            if s % 4 == 1:
                acc = add(acc, n1[k][l])
            if s % 4 == 2:
                acc = add(acc, n2[k][l])
            if s % 4 == 3:
                acc = add(acc, n3[k][l])

    print(f"p = {p_float}, size = {a}, -p log(s) = {-p_float * acc}, m = {m}, "
          + f"t = {time.time() - start_time}")
    file = open(f'result_{m:02d}.txt', 'w')
    file.write(f"p = {p_float}, size = {a}, -p log(s) = {-p_float * acc}, m = {m}\n")
    file.close()

# print(table)
# for k in range(0, 7):
#     np.savetxt(f"table_{k}.csv", table[k][:][:], delimiter=",")

# if __name__ == "__main__":
#     main()
