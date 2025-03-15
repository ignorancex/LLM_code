#   1  2     3     4 5 6 7 8 9 10 11      12
# name,n,list_size,w,r,N,d,t,e,g,time,numer_of_solutions\n
import csv
import matplotlib.pyplot as plt
import numpy as np


def read_linear(csv_file, n, lam):
    dataset = {}
    w_index = 1

    possible_n = [64, 128, 256]
    possible_lam = [10,15,20]

    if n not in possible_n or lam not in possible_lam:
        return None

    n_index = possible_n.index(n)
    lam_index = possible_lam.index(lam)

    with open(csv_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)    # skip the header

        for i in range(n_index*3 + lam_index):
            next(reader)

        data = next(reader)[w_index]
        return data

    return None


def read(csv_files, plot_names, time_accesors, linear_baseline=None):
    """
    :param csv_files:
    :param plot_names:
    :param time_accesors:
    :return:
    """
    w_index = 3

    data = []
    for i,csv_file in enumerate(csv_files):
        time_index = time_accesors[i]

        dataset = {}
        with open(csv_file, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if len(row) == 0:
                    continue
                if row[w_index] not in dataset:
                    dataset[row[w_index]] = []

                dataset[row[w_index]].append((float(row[time_index]), row[5:11]))

        # find optimum
        for key, items in dataset.items():
            dataset[key] = min(i for i, j in  items)

        data.append(dataset)

    X = np.array(list(data[0].keys()))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for i, d in enumerate(data):
        X = np.array(list(d.keys()))
        Y = np.array(list(d.values()))

        ax1.scatter(X,Y, label=plot_names[i])
        ax1.plot(X,Y)

     # plot quadratic search as a baseline
    if linear_baseline is not None:
        ldata = read_linear("good/our_linear.csv", linear_baseline[0], linear_baseline[1])
        qd = [ldata for _ in range(len(X))]
        ax1.scatter(X,qd, label="qadratic")
        ax1.plot(X,qd)

    plt.close(fig)
    return fig, data


def minimize(csv_files_e0, csv_files_e1, csv_files_alldelta, time_accesors):
    _, data_e0 = read([csv_files_e0], ["e0"], [time_accesors])
    _, data_e1 = read([csv_files_e1], ["e1"], [time_accesors])
    _, data_ad = read([csv_files_alldelta], ["ad"], [time_accesors])

    # now minimize over these values
    out =[{}]
    for key, item in data_e0[0].items():
        out[0][key] = min(data_e0[0][key], data_e1[0][key], data_ad[0][key])
    return out


def generate_tex(outfile, data):
    """
    take the highest key value and set n = 2*this_value.
    :param data: [{'0': 0.0, '4': 0.0, '8': 0.0, '12': 0.0, '16': 0.0, '20': 0.0, '24': 0.0, '28': 0.0, '32': 0.0}]
    :return:
    """
    if type(data) != list or len(data) != 1:
        return

    n = 2*int(list(data[0].keys())[-1])

    with open(outfile, "w") as f:
        f.write("W T\n")

        for keys, items in data[0].items():
            weight = float(keys)/float(n)
            f.write(str(weight) +  " " + str(items) + "\n")


def plot(name):
    plt.yscale('log')
    plt.legend()
    plt.title(name)
    plt.show()

#read(["good/gamma0.1.csv", "good/gamma0.2.csv", "good/gamma0.3.csv", "good/gamma0.4.csv", "good/gamma0.5.csv", "good/bench_n64_lam10_e0_allw.csv"],
#     ["g0.1", "g0.2", "g0.3", "g0.4", "g0.5", "normal"],
#     [10, 10, 10, 10, 10, 9],
#     "n=64, lam=10, all_delta, w=[0,...,32] + baseline")

#fig = read_linear("good/our_linear.csv", 255, 20)

#fig = read(["good/bench_n64_lam10_e0_allw.csv", "good/bench_n64_lam10_e1_allw.csv"],
#     ["epsilon=0", "epsilon=1"],
#     [10, 10], [64, 10])
#plot("n=64, lam=10, w=[0,...,32] + quadratic baseline")


# create latex files
#fig, data = read(["good/bench_n32_lam10_e0_allw.csv"], ["epsilon=0"], [9])
#generate_tex("good/plot_n32_lam10_e0", data)
#fig, data = read(["good/bench_n32_lam10_e1_allw.csv"], ["epsilon=1"], [9])
#generate_tex("good/plot_n32_lam10_e1", data)
#fig, data = read(["good/bench_n32_lam15_e0_allw.csv"], ["epsilon=0"], [9])
#generate_tex("good/plot_n32_lam15_e0", data)
#fig, data = read(["good/bench_n32_lam15_e1_allw.csv"], ["epsilon=1"], [9])
#generate_tex("good/plot_n32_lam15_e1", data)


# fig, data = read(["good/bench_n64_lam10_e0_allw.csv"], ["epsilon=0"], [9])
# generate_tex("good/plot_n64_lam10_e0", data)
# fig, data = read(["good/bench_n64_lam10_e1_allw.csv"], ["epsilon=1"], [9])
# generate_tex("good/plot_n64_lam10_e1", data)
# fig, data = read(["good/bench_n64_lam15_e0_allw.csv"], ["epsilon=0"], [9])
# generate_tex("good/plot_n64_lam15_e0", data)
# fig, data = read(["good/bench_n64_lam15_e1_allw.csv"], ["epsilon=1"], [9])
# generate_tex("good/plot_n64_lam15_e1", data)
#
# fig, data = read(["good/bench_n128_lam10_e0_allw.csv"], ["epsilon=0"], [9])
# generate_tex("good/plot_n128_lam10_e0", data)
# fig, data = read(["good/bench_n128_lam10_e1_allw.csv"], ["epsilon=1"], [9])
# generate_tex("good/plot_n128_lam10_e1", data)
# fig, data = read(["good/bench_n128_lam15_e0_allw_v2.csv"], ["epsilon=0"], [9])
# generate_tex("good/plot_n128_lam15_e0", data)
# fig, data = read(["good/bench_n128_lam15_e1_allw_v2.csv"], ["epsilon=1"], [9])
# generate_tex("good/plot_n128_lam15_e1", data)
#
# fig, data = read(["good/bench_n256_lam10_e0_allw.csv"], ["epsilon=0"], [9])
# generate_tex("good/plot_n256_lam10_e0", data)
# fig, data = read(["good/bench_n256_lam10_e1_allw.csv"], ["epsilon=1"], [9])
# generate_tex("good/plot_n256_lam10_e1", data)
#fig, data = read(["good/bench_n256_lam15_e0_allw.csv"], ["epsilon=0"], [9])
#generate_tex("good/plot_n256_lam15_e0", data)
#fig, data = read(["good/bench_n256_lam15_e1_allw.csv"], ["epsilon=1"], [9])
#generate_tex("good/plot_n256_lam15_e1", data)


#generate text code for the alldelta tests
#fig, data = read(["good/bench_n32_lam10_allw_alldelta.csv"], ["n=32,lam=10,alldelta"], [9])
#generate_tex("good/plot_n32_lam10_allw_alldelta", data)
#fig, data = read(["good/bench_n32_lam15_allw_alldelta.csv"], ["n=32,lam=15,alldelta"], [9])
#generate_tex("good/plot_n32_lam15_allw_alldelta", data)
# fig, data = read(["good/bench_n32_lam20_allw_alldelta.csv"], ["n=32,lam=20,alldelta"], [9])
# generate_tex("good/plot_n32_lam20_allw_alldelta", data)
#
# fig, data = read(["good/bench_n64_lam10_allw_alldelta.csv"], ["n=64,lam=10,alldelta"], [9])
# generate_tex("good/plot_n64_lam10_allw_alldelta", data)
# fig, data = read(["good/bench_n64_lam15_allw_alldelta.csv"], ["n=64,lam=15,alldelta"], [9])
# generate_tex("good/plot_n64_lam15_allw_alldelta", data)
# fig, data = read(["good/bench_n64_lam20_allw_alldelta.csv"], ["n=64,lam=20,alldelta"], [9])
# generate_tex("good/plot_n64_lam20_allw_alldelta", data)
#
# fig, data = read(["good/bench_n128_lam10_allw_alldelta.csv"], ["n=128,lam=10,alldelta"], [9])
# generate_tex("good/plot_n128_lam10_allw_alldelta", data)
# fig, data = read(["good/bench_n128_lam15_allw_alldelta.csv"], ["n=128,lam=15,alldelta"], [9])
# generate_tex("good/plot_n128_lam15_allw_alldelta", data)
# # does not exist: fig, data = read(["good/bench_n128_lam20_allw_alldelta.csv"], ["n=128,lam=20,alldelta"], [9])
# #                 generate_tex("good/plot_n128_lam20_allw_alldelta", data)
#
# fig, data = read(["good/bench_n256_lam10_allw_alldelta.csv"], ["n=256,lam=10,alldelta"], [9])
# generate_tex("good/plot_n256_lam10_allw_alldelta", data)
# fig, data = read(["good/bench_n256_lam15_allw_alldelta.csv"], ["n=256,lam=15,alldelta"], [9])
# generate_tex("good/plot_n256_lam15_allw_alldelta", data)
# # does not exist: fig, data = read(["good/bench_n256_lam20_allw_alldelta.csv"], ["n=256,lam=20,alldelta"], [9])
# #                 generate_tex("good/plot_n256_lam20_allw_alldelta", data)


# OLD generate code for gamma plots
# fig, data = read(["good/gamma0.1.csv"], ["n=64,lam=10,g=0.1"], [10])
# generate_tex("good/plot_n64_lam10_gamma0.1", data)
# fig, data = read(["good/gamma0.2.csv"], ["n=64,lam=10,g=0.2"], [10])
# generate_tex("good/plot_n64_lam10_gamma0.2", data)
# fig, data = read(["good/gamma0.3.csv"], ["n=64,lam=10,g=0.3"], [10])
# generate_tex("good/plot_n64_lam10_gamma0.3", data)
# fig, data = read(["good/gamma0.4.csv"], ["n=64,lam=10,g=0.4"], [10])
# generate_tex("good/plot_n64_lam10_gamma0.4", data)
# fig, data = read(["good/gamma0.5.csv"], ["n=64,lam=10,g=0.5"], [10])
# generate_tex("good/plot_n64_lam10_gamma0.5", data)

# data = minimize("good/bench_n32_lam10_e0_allw.csv",  "good/bench_n32_lam10_e1_allw.csv",  "good/bench_n32_lam10_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n32_lam10", data)
# data = minimize("good/bench_n32_lam15_e0_allw.csv",  "good/bench_n32_lam15_e1_allw.csv",  "good/bench_n32_lam15_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n32_lam15", data)
# data = minimize("good/bench_n64_lam10_e0_allw.csv",  "good/bench_n64_lam10_e1_allw.csv",  "good/bench_n64_lam10_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n64_lam10", data)
# data = minimize("good/bench_n64_lam15_e0_allw.csv",  "good/bench_n64_lam15_e1_allw.csv",  "good/bench_n64_lam15_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n64_lam15", data)
# data = minimize("good/bench_n128_lam10_e0_allw.csv", "good/bench_n128_lam10_e1_allw.csv", "good/bench_n128_lam10_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n128_lam10", data)
# data = minimize("good/bench_n128_lam15_e0_allw.csv", "good/bench_n128_lam15_e1_allw.csv", "good/bench_n128_lam15_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n128_lam15", data)
# data = minimize("good/bench_n256_lam10_e0_allw.csv", "good/bench_n256_lam10_e1_allw.csv", "good/bench_n256_lam10_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n256_lam10", data)
# data = minimize("good/bench_n256_lam15_e0_allw.csv", "good/bench_n256_lam15_e1_allw.csv", "good/bench_n256_lam15_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n256_lam15", data)

# v2 tests
# data = minimize("good/v2/bench_n128_lam10_e0_allw.csv", "good/v2/bench_n128_lam10_e1_allw.csv", "good/v2/bench_n128_lam10_allw_alldelta.csv", 9)
# generate_tex("good/plot_opt_n128_lam10_v2", data)
# data = minimize("good/v2/bench_n256_lam15_e0_allw.csv", "good/v2/bench_n256_lam15_e1_allw.csv", "good/v2/bench_n256_lam15_allw_alldelta_v2.csv", 9)
# generate_tex("good/plot_opt_n256_lam15_v2", data)
fig, data = read(["good/bench_n256_lam15_allw_alldelta_v2.csv"], ["n=256,lam=15,alldelta"], [9])
generate_tex("good/plot_n256_lam15_allw_alldelta_v2", data)

exit(1)

for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
	def t(path, y, alldelta, e_str, g_str):
		return path + "y" + str(y) + e_str + alldelta + g_str + ".csv"
	y = 10
	n = 64
	g_str =  "_g" + str(g).replace(".", "")
	path = "good/g/" + str(n) +"/v2/"
	data = minimize(t(path, y, "", "", g_str), t(path, y, "", "_e1", g_str), t(path, y, "_alldelta", "", g_str), 10)
	generate_tex("good/g/64/v2/.out/plot_opt_n"+str(64)+"_lam"+str(10)+"_gamma"+str(g)+"_v2", data)

exit(0)

# generate code for gamma plots
for n in [32, 64, 128, 256]:
    y = 10
    e = 0
    e_str = ""
    if e > 0:
        e_str = "_e"+str(e)

    alldelta = "_alldelta"
    path = "good/g/" + str(n) +"/"

    def t(path, y, alldelta, e_str, g_str):
        return path + "y" + str(y) + e_str + alldelta + g_str + ".csv"

    for g in [0.1, 0.2, 0.3, 0.4, 0.5]:
        g_str =  "_g" + str(g).replace(".", "")
        #fig, data = read([t(path, y, alldelta, e_str, g_str)], ["n="+str(n)+",lam="+str(y)+",g="+str(g)], [10])
        #generate_tex(path+".out/plot_n"+str(n)+"_lam"+str(y)+e_str+alldelta+"_gamma"+str(g), data)

        data = minimize(t(path, y, "", "", g_str), t(path, y, "", "_e1", g_str), t(path, y, "_alldelta", "", g_str), 10)
        generate_tex(path+".out/plot_opt_n"+str(n)+"_lam"+str(y)+"_gamma"+str(g), data)
