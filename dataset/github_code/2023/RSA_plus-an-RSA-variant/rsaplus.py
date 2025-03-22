from math import gcd
from math import log
from random import *
import time
import cypari2

pari = cypari2.Pari()
pari.allocatemem()
pari.allocatemem()
pari.allocatemem()



# 1. Auxiliary functions: 

def sqrt_threemodfour(y, p):
    return pow(y, (p + 1) // 4, p)


def sqrt_fivemodeight(y, p):
    ret = pow(y, (p + 3) // 8, p)
    if pow(y, (p - 1) // 4, p) != 1:
        ret *= pow(2, (p - 1) // 4, p)
    return ret



---------------------------------------------------------------------------------------------------------

# 2. Key generation: 
def generate_rsa(bits):
    bound = 2 ** bits
    e = 65537
    temp = 1
    while temp % 8 ==1 or temp-1 % e == 0:
        temp = pari.randomprime([bound, bound*2])
    p = temp

    bound *= 4
    temp = 1
    while temp % 8 ==1 or temp-1 % e == 0:
        temp = pari.randomprime([bound, bound*2])
    q = temp

    d = int(pow(e, -1, (p-1)*(q-1)))

    return e, d, p, q



---------------------------------------------------------------------------------------------------------

# 3. Encryption and decryption: 

def rsap_encrypt(m, n, bits):
    baseprime = pari.randomprime([2**150, 2**190])
    expo = pari.random([pari.truncate(log(2)*(bits+1)/log(baseprime)), pari.truncate(log(2)*(3/2*bits)/log(baseprime))])
    x = baseprime**expo

    c = pow(m, x, n)
    y = pow(x, 2, n)

    return c, y



def rsap_decrypt(p, q, c, y):
    n = p * q
    phin = (p - 1) * (q - 1)

    if p % 4 == 3:
        t1 = sqrt_threemodfour(y, p)
    else:
        t1 = sqrt_fivemodeight(y, p)
    if q % 4 == 3:
        t2 = sqrt_threemodfour(y, q)
    else:
        t2 = sqrt_fivemodeight(y, q)
    x1 = pari.lift(pari.chinese(pari.Mod(t1, p), pari.Mod(t2, q)))
    x2 = pari.lift(pari.chinese(pari.Mod(t1, p), pari.Mod(q - t2, q)))

    if gcd(x1, phin) != 1:
        x1 = n - x1

    if gcd(x1, phin) == 1:
        x1 = int(pow(x1, -1, phin))
        m11 = pow(c,x1 % (p-1),p)
        m12 = pow(c,x1 % (q-1),q)
        m1 = pari.lift(pari.chinese(pari.Mod(m11,p), pari.Mod(m12,q)))
    else:
        m1 = 0

    if gcd(x2, phin) != 1:
            x2 = n - x2

    if gcd(x2, phin) == 1:
        x2 = int(pow(x2, -1, phin))
        m21 = pow(c,x2 % (p-1),p)
        m22 = pow(c,x2 % (q-1),q)
        m2 = pari.lift(pari.chinese(pari.Mod(m21,p), pari.Mod(m22,q)))
    else:
        m2 = 0


    return m1, m2



def rsa_encrypt(m,e,n):
    c = pow(m,e,n)

    return c


def rsa_decrypt(c,d,p,q):
    m1 = pow(c,d,p)
    m2 = pow(c,d,q)
    m = pari.lift(pari.chinese(pari.Mod(m1,p), pari.Mod(m2,q)))

    return m


def rabin_encrypt(m,n):
    return pow(m,2,n)


def rabin_decrypt(c,p,q):
    n = p*q
    if p % 4 == 3:
        t1 = sqrt_threemodfour(c, p)
    else:
        t1 = sqrt_fivemodeight(c, p)
    if q % 4 == 3:
        t2 = sqrt_threemodfour(c, q)
    else:
        t2 = sqrt_fivemodeight(c, q)
    x1 = pari.lift(pari.chinese(pari.Mod(t1, p), pari.Mod(t2, q)))
    x2 = pari.lift(pari.chinese(pari.Mod(t1, p), pari.Mod(q - t2, q)))
    x3 = n - x1
    x4 = n - x2

    return x1, x2, x3, x4



---------------------------------------------------------------------------------------------------------

# 4. Runtime test: 

def runtime_test(bits, inst, i):
    l = []
    control = 0
    t0 = 0
    t = time.time()
    for j in range(inst):
        e, d, p, q = generate_rsa(bits)
        l.insert(j, [e, d, p, q])
    t0 = ((time.time() - t)*1000)/inst
    print("Key generation complete")
    t = time.time()
    t1 = t2 = t3 = 0
    for j in l:
        p = j[2]
        q = j[3]
        n = p*q
        for s in range(i):
            m = randint(0, n)
            c, y = rsap_encrypt(m, n, bits)
            test = rsap_decrypt(p, q, c, y)
            if m != test[0] and m != test[1]:
                control = -1
                break
    t1 = ((time.time() - t)*1000)/(inst*i)
    print("RSA+ done")

    t = time.time()
    for j in l:
        e = j[0]
        d = j[1]
        p = j[2]
        q = j[3]
        n = p*q
        for s in range(i):
            m = randint(0, n)
            c = rsa_encrypt(m, e, n)
            test = rsa_decrypt(c, d, p, q)
            if m != test:
                control = -2
                break
    t2 = ((time.time() - t)*1000)/(inst*i)
    print("RSA done")

    t = time.time()
    for j in l:
        p = j[2]
        q = j[3]
        n = p*q
        for s in range(i):
            m = randint(0, n)
            c = rabin_encrypt(m, n)
            test = rabin_decrypt(c, p, q)
            if m != test[0] and m != test[1] and m != test[2] and m != test[3]:
                control = -3
                break
    t3 = ((time.time() - t)*1000)/(inst*i)

    return control, t0, t1, t2, t3

