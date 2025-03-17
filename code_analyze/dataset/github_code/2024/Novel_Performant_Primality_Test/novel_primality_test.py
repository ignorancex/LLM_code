'''
This Python file was executed using Python 3.8.10 and Sympy 1.12.
Last updated in October 26 2024 by Luca Di Domenico
'''

# These are exactly all the primes less than 1000.
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

def find_r(n):
    '''This function finds the smallest possible non-cube r in Z/nZ, where the input n is a positive integer bigger than 3 and congruent to 1 modulo 3. Any prime p congruent to 1 modulo 3 has (p-1)/3 cubes. If the function returns "0", then n is surely composite. In most cases, r=2 is returned.'''
    exponent = n // 3
    for r in SMALL_PRIMES:
        if (pow(r, exponent, n) != 1): return r
    for r in range(998, n):
        if (pow(r, exponent, n) != 1): return r
    return 0

def square_and_multiply_1mod3(n, r):
    '''This function uses the square-and-multiply algorithm to compute efficiently [1,1,0]^{n} on the projectivisation of the Pell cubic with parameter r and modulo n, where n is an odd integer bigger than 3 and congruent to 1 mod 3.'''
    mask = 1 << (n.bit_length() - 2)
    x, y, z = 1,1,0
    double_r = (r << 1) % n
    while(mask > 0):
        # Square operation over the projectiviastion
        x,y,z=(x*x+(double_r*y*z))%n,((x*y << 1)+ r*z*z)%n,((x*z << 1)+y*y)% n
        # Multiply operation over the projectiviastion
        if(n & mask): x, y, z = (x + r*z) % n, (y + x) % n, (z + y) % n
        mask >>= 1
    return x, y, z

def square_and_multiply_2mod3(n):
    '''This function uses the square-and-multiply algorithm to compute efficiently [1,1,0]^{n} on the projectivisation of the Pell cubic with parameter r = 2 and modulo n, where n is an odd integer bigger than 3 and congruent to 2 mod 3.'''
    mask = 1 << (n.bit_length() - 2)
    x, y, z = 1,1,0
    while(mask > 0):
        # Square operation over the projectiviastion
        x,y,z=(x*x +((y*z) << 2))%n,((x*y + z*z) << 1)%n,((x*z << 1)+y*y) % n
        # Multiply operation over the projectiviastion
        if(n & mask): x,y,z=(x+(z << 1))% n, (y + x) % n, (z + y) % n
        mask >>= 1
    return x, y, z

def check_primality_conditions(n, isCongruentTo2mod3):
    '''Given an odd integer "n" not divisible by 3 and a boolean "isCongruentTo2mod3", this function classifies "n" as either prime or composite using the structure of the n-th power of an element of the projectivisation of the Pell's cubic. If the function returns "False", then "n" is surely composite. If the function returns "True", then "n" is likely a prime.'''
    quotient = n // 3
    # Case n congruent to 2 modulo 3
    if (isCongruentTo2mod3):
        x,y,z = square_and_multiply_2mod3(n)
        if ((x == 1) and (y == 0) and (z == pow(2, quotient, n))): return True
        return False
    # Case n congruent to 1 modulo 3
    r = find_r(n)
    if (r):
        x,y,z = square_and_multiply_1mod3(n, r)
        if ((x == 1) and (z == 0)):
            power = pow(r, quotient, n)
            if ((y == power) and (((power + pow(power, 2, n)) % n) == (n-1))): return True
    return False

def is_prime_basic(n):
    '''Given an integer "n" bigger than 1, this function classifies it as prime or composite using the nth-power of an element of the projectivisation of the Pell's cubic. If the function returns "False", then "n" is surely composite. If the function returns "True", then "n" is likely a prime.'''
    # Super basic checks
    assert n > 1, "Please insert an integer bigger than 1."
    if ((n == 2) or (n == 3)): return True
    if ((n & 1) == 0): return False
    remainder = n % 3
    # We exclude multiples of 3
    if (remainder < 1): return False
    # We deal with the non-trivial cases.
    return check_primality_conditions(n, (remainder > 1))

def is_prime_strong(n):
    '''Given an integer "n" bigger than 1, this function classifies it as prime or composite using some trivial checks before exploiting the characterisation of the nth-power of an element of the projectivisation of the Pell's cubic. If the function returns "False", then "n" is surely composite. If the function returns "True", then "n" is likely a prime.'''
    # Trivial cases using primes smaller than 1000.
    assert n > 1, "Please insert an integer bigger than 1."
    if (n in SMALL_PRIMES): return True
    for p in SMALL_PRIMES:
        if ((n % p) == 0): return False
    # Fermat's little theorem for base 2 is used as a "filter".
    # Any number failing Fermat's little theorem is surely composite.
    # Note that, at this stage, n is not a multiple of 3, hence (n % 3) is either 1 or 2.
    if (pow(2, n - 1, n) == 1): return check_primality_conditions(n, ((n % 3) > 1))
    return False



if __name__ == "__main__":
    
    # In the following, we will import the "isprime" function from sympy to compare its outputs with the above tests.
    from sympy.ntheory import isprime

    # The following lines can be used to print your current Python version and your current Sympy version.
    # import sys
    # import sympy
    # print("Current Python version is: ", sys.version)
    # print("Current Sympy version is: ", sympy.__version__)

    def compare_test_with_sympyisprime(test, user_exponent):
        '''This function compares the output of the sympy "isprime" function with the one of "test", for odd integers ranging between 2 and 2**{user_exponent}'''
        assert user_exponent <= 64, "An exponent bigger than 64 does not ensure primality for the isprime Sympy method!"
        assert user_exponent > 7, "The exponent should be between 8 and 64, extremes included."

        print("The following primality test will be compared against sympy 'isprime': ", test)
        for even in range(2, 100, 2):
            assert isprime(even) == test(even), "The provided primality test misclassifies even integers"
        
        stop = (2**user_exponent)
        modulus_percentage = stop // 200 # We are analysing only odd integers
        iteration_counter = 0
        percentage = 1
        pseudoprimes = []

        for tested_integer in range(3, stop, 2):
            
            test_boolean = test(tested_integer)
            sympy_boolean = isprime(tested_integer)
            iteration_counter += 1

            # If test classifies a number prime while it isn't, then we have hit a pseudoprime!
            if (test_boolean and (not sympy_boolean)): pseudoprimes.append(tested_integer)
            
            # If the test classifies a prime as a composite we raise an exception!
            if ((not test_boolean) and sympy_boolean): raise Exception("THIS PRIME WAS MISSED BY THE PROVIDED PRIMALITY TEST:", tested_integer)
            
            # Here we print a rough progress percentage value.
            if((iteration_counter % modulus_percentage) == 0):
                print(percentage, "%", end = "\r")
                percentage += 1

        print("There are", len(pseudoprimes), "odd pseudoprimes between 2 and 2**", user_exponent, "(=", 2**user_exponent, "), extremes EXCLUDED:")
        print(pseudoprimes)
    
    compare_test_with_sympyisprime(is_prime_basic, 22)
    compare_test_with_sympyisprime(is_prime_strong, 25)
    # It requires around 3 hours on my laptop to test is_prime_basic up to 2**32
