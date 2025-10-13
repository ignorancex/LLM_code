# import psr
# import time
# import numpy as np
# times_qiskit = []
# times_pennylane = []
# times_qulacs = []
# for k in range(0, 5):
#     print("Times: ", k)
#     time_qiskit = []
#     time_pennylane = []
#     time_qulacs = []
#     for j in range(2, 4):
#         for package in [psr.psr_qiskit, psr.psr_pennylane, psr.psr_qulacs]:
#             start = time.time()
#             package(j)
#             end = time.time()
#             exe_time = end-start
#             if package == psr.psr_qiskit:
#                 time_qiskit.append(exe_time)
#             elif package == psr.psr_pennylane:
#                 time_pennylane.append(exe_time)
#             else:
#                 time_qulacs.append(exe_time)
#     times_qiskit.append(time_qiskit)
#     times_pennylane.append(time_pennylane)
#     times_qulacs.append(time_qulacs)
# np.savetxt("./times/qiskit.txt", np.mean(times_qiskit, axis = 0))
# np.savetxt("./times/pennylane.txt", np.mean(times_pennylane, axis = 0))
# np.savetxt("./times/qulacs.txt", np.mean(times_qulacs, axis = 0))
# np.savetxt("./times/qiskit_std.txt", np.std(times_qiskit, axis = 0))
# np.savetxt("./times/pennylane_std.txt", np.std(times_pennylane, axis = 0))
# np.savetxt("./times/qulacs_std.txt", np.std(times_qulacs, axis = 0))