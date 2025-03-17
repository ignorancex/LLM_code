import math

def run(qc):
    qc.x(0)
    qc.h(2)

    qc.barrier([0,1,2])

    qc.h(4)
    qc.p(math.pi / 4, 4)
    qc.barrier([3,4,5,6,7,8,9,10,11,12,13,14])

    mch(qc, [5, 8], 4)  # M1

    # a-=3
    qc.x(1)
    qc.cx(1, 2)
    qc.x(0)
    qc.cx(0, 1)
    qc.mct([0, 1], 2)
    qc.barrier([0,1,2])

    # if a<0, b++
    for i in range(12):
        control = []
        control.append(2)
        if i < 11:
            for j in range(11 - i):
                control.append(j+3)
        qc.mct(control, 14 - i)

    qc.barrier([0,1,2])

    # a+=3
    qc.mct([0, 1], 2)
    qc.cx(0, 1)
    qc.x(0)
    qc.cx(1, 2)
    qc.x(1)
    qc.barrier([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    qc.measure([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0,1,2,3,4,5,6,7,8,9,10,11])

def mch(qc, c_q, t_q):
    qc.ry(math.pi/4,t_q)
    qc.mct(c_q,t_q)
    qc.ry(-math.pi/4,t_q)
    return qc




