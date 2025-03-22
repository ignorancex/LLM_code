#!/bin/python
# generate include/header files with different parameters
import os
import sys
import math
import argparse
import subprocess
import time
import multiprocessing
from multiprocessing import Process, Queue
from subprocess import Popen, PIPE, STDOUT

run_file = "bench_b63_bench_times"


single_treader = True
all_delta = False

max_threads: int = 250
cur_threads = 0

RUNS=1
ITERS=20

def HH(i: float):
    if i == 1.0 or i == 0.0:
        return 0.0

    if i > 1.0 or i < 0.0:
        print("error: ", i)
        raise ValueError

    return -(i * math.log2(i) + (1 - i) * math.log2(1 - i))


def H1(value: float):
    if value == 1.0:
        return 0.5

    # approximate inverse binary entropy function
    steps = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.0000000001, 0.0000000000001, 0.000000000000001]
    r = 0.000000000000000000000000000000001
    if value < 0.:
        value = 0.

    for step in steps:
        i = r
        while (i + step < 1.0) and (HH(i) < value):
            i += step
            
        if step > i:
            break
        r = i - step

    return r


def testH1():
    x = 0.00001
    y = HH(x)
    x_ = H1(y)

    print(x, x_, y)


def calc_q_(n: int, w_: float, d_: float):
    wm1 = 1 - w_
    # because H(0.5) = 1 and from (1/2)^{-n} comes a -1 we can skip them.
    return wm1 * HH((d_ - (w_ / 2)) / wm1)


def calc_q(n: int, w: int, d: int):
    """
    assumes that omega and `d` is absolute
    :return:
    """
    w_ = float(w)/float(n)
    d_ = float(d)/float(n)
    return calc_q_(n, w_, d_)


def calc_q2(n: int, d: int):
    d_ = float(d)/float(n)
    return HH(d_) - 1.0


def write_config(outfile, n: int, N: int, r: int, w: int, d: int, list_size: int, epsilon=0, THRESHHOLD=10, RUNS=2, ITERS=10, ratio=0.5, gamma=0.1):
    with open(outfile, "w") as f:
        f.write("""#ifndef NN_CODE_OPTIONS_H
#define NN_CODE_OPTIONS_H\n""")

        f.write("#define TEST_BASE_LIST_SIZE_LOG " + str(int(math.log(list_size, 2))) + "\n")
        f.write("#define TEST_BASE_LIST_SIZE " + str(list_size) + "\n")
        f.write("#define G_n " + str(n) + "\n")
        f.write("constexpr uint64_t w=" + str(w) + ";\n")
        f.write("constexpr uint64_t N=" + str(N) + ";\n")
        f.write("constexpr uint64_t r=" + str(r) + ";\n")
        f.write("constexpr uint64_t d=" + str(d) + ";\n")
        f.write("constexpr uint64_t epsilon=" + str(epsilon) + ";\n")
        f.write("constexpr uint64_t THRESHHOLD=" + str(THRESHHOLD) + ";\n")
        f.write("constexpr double ratio=" + str(ratio) + ";\n")
        f.write("constexpr double gam=" + str(gamma) + "*G_n;\n")
        f.write("#define ITERS  " + str(ITERS) + "\n")
        f.write("#define RUNS " + str(RUNS) + "\n")
        if all_delta:
            f.write("#define ALL_DELTA\n")
        f.write("#endif //NN_CODE_OPTIONS_H")


def rebuild():
    p = Popen(["make", run_file, "-j1"], stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd= "../cmake-build-release")
    _ = p.stdout.read()
    if p != 0:
        print("Error")
    #print(t)


def bench(benchfile):
    print("running", run_file)
    p = Popen("./" + run_file, shell=True,  stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd= "../cmake-build-release/bench")
    output = p.stdout.read()
    out_str = output.decode("utf-8")

    with open(benchfile, "a") as f:
        f.write(out_str)

    global cur_threads
    cur_threads -= 1


def do(outfile, benchfile, n: int, N: int, r: int, w: int, d: int, list_size: int, epsilon=0, THRESHHOLD=10, RUNS=2, ITERS=10):
    write_config(outfile, n, N, r, w, d, list_size, epsilon, THRESHHOLD, RUNS, ITERS)
    time.sleep(1)
    rebuild()
    time.sleep(1)
    bench(benchfile)


def worker(q):
    while True:
        item = q.get(True)
        print (os.getpid(), "got", *item)
        do(*item)
        q.task_done()


def bench_clever(outfile, benchfile):
    """
    this functions tries to find the optimal parameters for different n, lam and omega within an range. The difference to
    "bench_clever" ist, that this function does not choose the theoretically optimum for each parameter.
    :param outputfile:
    :param benchfile:
    :return:
    """

    if not single_treader:
        the_queue = multiprocessing.JoinableQueue()
        the_pool = multiprocessing.Pool(max_threads, worker,(the_queue,))

    n_dict = {
        "32": {
            "lam": [10],
            "w": [i for i in range(0, 17, 4)],
            "r": [2],
            "d": [i for i in range(8, 17, 2)],
            "N": [10, 50, 100, 200, 500],
            "epsilon": [1],
            "t": [20, 100],
        },
        "64": {
            "lam": [15],
            "w": [i for i in range(0, 1, 4)],
            "r": [2],
            "d": [i for i in range(20, 27, 2)],
            "N": [10, 50, 100, 500],
            "epsilon": [0],
            "t": [20, 100],
        },

        "128": {
            "lam": [10],
            "w": [i for i in range(0, 53, 4)],
            "r": [2],
            "d": [i for i in range(46, 52, 2)],
            "N": [10, 50, 100, 1000],
            "epsilon": [0],
            "t": [50],
        },

        "256": {
            "lam": [15],
            "w":  [i for i in range(108, 129, 4)],
            "r": [2],
            "d": [i for i in range(110, 125, 2)],
            "N": [50, 100, 500, 1000],
            "epsilon": [0],
            "t": [50],
        }
    }

    for n in ["256"]:   #n_dict.keys():
        for lam in n_dict[n]['lam']:
            list_size = 1<<lam
            for epsilon in n_dict[n]['epsilon']:
                for w in n_dict[n]['w']:
                    for THRESHHOLD in n_dict[n]['t']:
                        for r in n_dict[n]['r']:
                            for d in n_dict[n]['d']:
                                for N in n_dict[n]['N']:
                                    if single_treader:
                                        print("n:", n, "lam:", lam, "w:", w, "r:", r, "N:", N, "d:", d, epsilon, THRESHHOLD)
                                        write_config(outfile, n, N, r, w, d, list_size, epsilon, THRESHHOLD, RUNS, ITERS)
                                        time.sleep(1)
                                        rebuild()
                                        time.sleep(1)
                                        bench(benchfile)
                                    else:
                                        the_queue.put((outfile, benchfile, n, N, r, w, d, 1<<lam, epsilon, THRESHHOLD, RUNS, ITERS, ))
                                        time.sleep(7)

    if not single_treader:
        the_queue.close()
        the_queue.join_thread()

        # prevent adding anything more to the process pool and wait for all processes to finish
        the_pool.close()
        the_pool.join()


def runtime_estimator(d):
    """
    :param d:
    :return:
    """
    fields = ["lam", "w", "r", "d", "N", "epsilon", "t"]

    key = list(d.keys())[0] # = n
    key_ = int(key)
    lam_ = d[key]['lam'][0]/key_
    w_ = d[key]['w'][0]/key_

    Hi2 = H1(1. - (lam_))
    w_star =2*Hi2*(1-Hi2)
    n_datasets = sum([len(d[key][s]) for s in fields])
    
    # TODO over all input values
    #n, lam, w, r, N, d, q, w_star = optimal_parameter_set(int(key), lam_, w_)
    n, lam, w, r, N, d, q, w_star = key_, lam_, w_, \
        d[key]["r"][0], d[key]["N"][0], d[key]["d"][0], 0, 0
    delta_star = H1(1. - lam)

    if w <= w_star:
        theta = (1.0-w)*(1.0 - HH((delta_star - w/2.)/(1-w)))
    else:
        theta = 2*lam + HH(w) - 1.

    print("log(time)", theta, "time", 2**theta, "total:", n_datasets*2**theta, "nr datasets", n_datasets)


def bench_dist(outfile, benchfile):
    """
    this function finds the optimal parameter set for different distributions. In this case for the distribution,
    that every element has the same weight gamma*n in the two lists.
    It then bruteforces the parameter set in a given range.
    :param outfile:
    :param benchfile:
    :return:
    """

    print("dist")

    n_dict = {
        "32": {
            "lam": [15],
            "gam": [0.1, 0.2, 0.3, 0.4, 0.5],
            "w": [17],
            "r": [2],
            "d": [60],
            "N": [10, 50, 100, 500],
            "epsilon": [1],
            "t": [50],
        },
        "64": {
            "lam": [10],
            "gam": [0.1, 0.2, 0.3, 0.4,0.5],
            "w": [i for i in range(0, 1, 4)],
            "r": [2],
            "d": [i for i in range(18, 33, 2)],
            "N": [50, 100, 500],
            "epsilon": [0],
            "t": [50],
        },
        "128": {
            "lam": [15],
            "gam": [0.3, 0.4, 0.5],
            "w": [i for i in range(0, 65, 4)],
            "r": [2],
            "d": [i for i in range(50, 59, 2)],
            "N": [10, 50, 100, 500, 1000],
            "epsilon": [0],
            "t": [50],
        },
        "256": {
            "lam": [15],
            "gam": [0.1, 0.2, 0.3, 0.4, 0.5],
            "w": [i for i in range(4, 129, 4)],
            "r": [2],
            "d": [i for i in range(100, 125, 2)],
            "N": [10, 100, 500],
            "epsilon": [1],
            "t": [50],
        },
    }

    epsilon = 1
    for n in ["128"]:   #n_dict.keys():
        for lam in n_dict[n]['lam']:
            list_size =  1<<lam
            for gamma in n_dict[n]['gam']:
                for THRESHHOLD in n_dict[n]['t']:
                    for w in n_dict[n]['w']:
                        for r in n_dict[n]['r']:
                            for d in n_dict[n]['d']:
                                for N in n_dict[n]['N']:
                                    print(n, N, r, w, d, list_size, epsilon, THRESHHOLD, RUNS, ITERS, gamma)
                                    write_config(outfile, n, N, r, w, d, list_size, epsilon, THRESHHOLD, RUNS, ITERS, 0.5, gamma)

                                    p = Popen(["make", "bench_b63_bench_dist_times", "-j1"], stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd= "../cmake-build-release")
                                    _ = p.stdout.read()

                                    print("finished build")
                                    p = Popen("./bench_b63_bench_dist_times", stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd= "../cmake-build-release/bench")
                                    output = p.stdout.read()
                                    out_str = output.decode("utf-8")

                                    with open(benchfile, "a") as f:
                                        f.write(out_str)


def optimal_parameter_blockwise_set(n: int, lam: float, w: float):
    """
    :param n: 100
    :param lam: 2**(lam n)
    :param w: w*n
    :return:
    """
    r = lam*n/(math.log2(n))
    Hi= H1(1. - ((r-1)*lam)/r)
    w_star =2*Hi*(1-Hi)
    print("w", w_star, w)
    if w > w_star:
        d = 1/2 * (1 - math.sqrt(1-2*w))
    else:
        d = Hi

    q = calc_q_(n, w, d)
    N = int(n/q)

    print("Hi", Hi )
    print("n:", n, "lam:", lam, "w:", w, "r:", r, "N:", N, "d:", d, "q:", q)
    print("n:", n, "size:", lam*n, "w:", w*n, "r:", int(r), "N:", N, "d:", d*n, "q:", q)
    return n, lam, w, r, N, d, q, w_star


def optimal_parameter_set(n: int, lam: float, w: float):
    """
    :param n: 100
    :param lam: 2**(lam n)
    :param w: w*n
    :return:
    """

    assert(w < 1)
    r = lam*n/(math.log2(n))
    Hi= H1(1. - (lam))
    Hi2= H1(1. - (lam))

    w_star =2*Hi2*(1-Hi2)
    print("w", w_star, w)
    if w > w_star:
        d = 1/2. * (1 - math.sqrt(1-2*w))
    else:
        d = Hi

    q = calc_q_(n, w, d)
    N = int(n/q)

    print("Hi", Hi )
    print("n:", n, "lam:", lam, "w:", w, "r:", r, "N:", N, "d:", d, "q:", q)
    print("n:", n, "size:", lam*n, "w:", w*n, "r:", int(r), "N:", N, "d:", d*n/r, "q:", q)
    return n, lam, w, r, N, d, q, w_star


def optimal_w(lam: float, n=1):
    """
    calcs an optimal w for a given list size to ensure that only one element tuple valid
    :param lam:
    :return:
    """
    w = H1(1-2*lam)*n
    optimal_parameter_set(n, lam, w)


def optimal_w2(n: int, lam: int):
    """
    :param lam:
    :return:
    """
    lam_ = lam/n
    w = H1(1-2*lam_)
    optimal_parameter_set(n, lam_, w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skip some pages.')
    parser.add_argument('-o', '--output', help='outputfile', required=False)
    parser.add_argument('-b', '--benchfile', help='benchfile', required=False)
    parser.add_argument('-p', '--params', nargs='+', help='output the optimal parameter set for given (n, \lambda, w) e.g. (100, 0.15, 0.2)', required=False)
    parser.add_argument('-w', '--omega', nargs='+', help='outputs an optimal w for a given list size, to ensure that only on element is valid. e.g: call with 0.15 for a relative weight. Or call it with 100 0.15 to get a weiht scalled on a given n', required=False)
    parser.add_argument('-a', '--all', nargs='+', help='Given (n, lam) = 100, 15', required=False)
    parser.add_argument('-r', '--runtime', help='runtime estimator', required=False, action='store_true')
    parser.add_argument('-m', '--multithreaded', help='run multithreaded', required=False, action='store_true')
    parser.add_argument('-d', '--dist', help='run distribution benchmarks', required=False, action='store_true')
    parser.add_argument('-c', '--clever', help='run clever benchmarks', required=False, action='store_true')
    parser.add_argument('-u', '--run_file', help='reset the file to execute', required=False)
    parser.add_argument('-e', '--all_delta', help='delta * k in each bucket', required=False,  action='store_true')

    args = parser.parse_args()

    if args.multithreaded:
        single_treader = False

    if args.run_file:
        run_file = args.run_file

    if args.all_delta:
        all_delta = args.all_delta

    if args.all:
        optimal_w2(int(args.all[0]), int(args.all[1]))
        exit()

    if args.omega:
        if len(args.omega) == 2:
            optimal_w(float(args.omega[1]), int(args.omega[0]))
        else:
            optimal_w(float(args.omega[0]))
        exit()

    if args.params:
        if len(args.params) != 3:
            print("please pass 'n, \lmbda, w'")
            exit(1)

        optimal_parameter_set(int(args.params[0]), float(args.params[1]), float(args.params[2]))
        exit()

    if args.runtime:
        n_dict = {
            "256": {
                "lam": [25],
                "w": [17],
                "r": [2],
                "d": [i for i in range(20, 33, 2)],
                "N": [50, 100, 500],
                "epsilon": [0, 1],
                "t": [10, 100, 1000],
            }
        }

        runtime_estimator(n_dict)

    if args.dist:
        bench_dist(args.output, args.benchfile)

    if args.clever:
        bench_clever(args.output, args.benchfile)
