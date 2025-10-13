import math


#calculate the binary entropy of a given number between 0 and 1
def calc_binary_entropy(num):
    if num == 0:
        return 0
    elif num == 1:
        return 1
    else:
        return -(num*math.log2(num) + (1-num)*math.log2(1-num))


print((1-calc_binary_entropy(0.016))*0.138)