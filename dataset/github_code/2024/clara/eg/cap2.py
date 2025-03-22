def cap(s):
    x = 1
    y = 2
    d = x + y
    print (d)

    def add(a, b):
        return a + b

    for z in range(1):
        d += z
    d += add(x, y)
    return d

def blue():
    return "you blue"