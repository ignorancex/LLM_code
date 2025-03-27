
def run(qc):
    qc.mct([1, 3], 6)  # M1

    qc.barrier([0, 1, 2, 3, 4, 5, 6])

    qc.h([7, 8, 9, 10, 11, 12, 13])
    for i in range(7):
        qc.cz(i, i+7)

    qc.barrier([0, 1, 2, 3, 4, 5, 6])

    qc.h([7, 8, 9, 10, 11, 12, 13])
    qc.measure([7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6])



