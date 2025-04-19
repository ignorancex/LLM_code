#include <bits/stdc++.h>

register uint64_t l asm ("rsi");
register uint64_t ss asm ("r8");
register uint64_t sns asm ("r9");
register uint64_t f asm ("r10");
register uint64_t *fc asm ("r11");
register uint64_t *matrix asm ("r12");
register uint64_t n asm ("r13");
register uint64_t a asm ("r14");
register int64_t s asm ("r15");

void recurse_unrolled() {
    if (__builtin_popcountll(a) + ((uint64_t) s) < l) {
        return;
    }
    if (s > 2) {
        --s; ss>>=1; sns<<=1;
        recurse_unrolled();
        a += ss;
        matrix += n;
        fc[s] = f;
        f |= (a << (s+1));
        recurse_unrolled();
        a -= ss;
        a += (sns >> 1);
        f = fc[s] | (sns >> 2);
        recurse_unrolled();
        ss<<=1; sns>>=1;
        a -= sns;
        f = fc[s];
        ++s;
        matrix -= n;
    } else {
        ss>>=1; sns<<=1;


        ++matrix[ __builtin_popcountll(f)];
        matrix += n;
        uint64_t b = a + 2;
        ++matrix[__builtin_popcountll(f | b)];
        ++matrix[__builtin_popcountll(f | sns)];
        uint64_t g = f | (sns >> 1);
        ++matrix[ __builtin_popcountll(g)];
        matrix += n;
        ++matrix[ __builtin_popcountll(g | (b + sns))];
        ++matrix[__builtin_popcountll(g | sns)];
        b += 2;
        g = f | (b << 1);
        ++matrix[ __builtin_popcountll(g | b)];
        ++matrix[__builtin_popcountll(g | sns)];
        matrix -= n;
        ++matrix[ __builtin_popcountll(g)];


        a += ss;
        uint64_t h = f | (a << 2);


        ++matrix[ __builtin_popcountll(h)];
        matrix += n;
        b = a + 2;
        ++matrix[__builtin_popcountll(h | b)];
        ++matrix[__builtin_popcountll(h | sns)];
        g = h | (sns >> 1);
        ++matrix[ __builtin_popcountll(g)];
        matrix += n;
        ++matrix[ __builtin_popcountll(g | (b + sns))];
        ++matrix[__builtin_popcountll(g | sns)];
        b += 2;
        g = h | (b << 1);
        ++matrix[ __builtin_popcountll(g | b)];
        ++matrix[__builtin_popcountll(g | sns)];
        matrix -= n;
        ++matrix[ __builtin_popcountll(g)];
        matrix -= n;


        a -= ss;
        a += (sns >> 1);
        h = f | (sns >> 2);


        ++matrix[ __builtin_popcountll(h)];
        matrix += n;
        b = a + 2;
        ++matrix[__builtin_popcountll(h | b)];
        ++matrix[__builtin_popcountll(h | sns)];
        g = h | (sns >> 1);
        ++matrix[ __builtin_popcountll(g)];
        matrix += n;
        ++matrix[ __builtin_popcountll(g | (b + sns))];
        ++matrix[__builtin_popcountll(g | sns)];
        b += 2;
        g = h | (b << 1);
        ++matrix[ __builtin_popcountll(g | b)];
        ++matrix[__builtin_popcountll(g | sns)];
        matrix -= n;
        ++matrix[ __builtin_popcountll(g)];
        matrix -= n;


        ss<<=1; sns>>=1;
        a -= sns;
        matrix -= n;
    }
}

void recurse() {
    if (s >= 0) {
        --s; ss>>=1; sns<<=1;
        recurse();
        ++s; ss<<=1;

        a += ss;
        matrix += n;
        fc[s] = f;
        f |= (a << s);
        --s; ss>>=1;
        recurse();
        ++s; ss<<=1; sns>>=1;
        a -= ss;

        a += sns;
        f = fc[s] | (sns >> 1);
        --s; ss>>=1; sns<<=1;
        recurse();
        ++s; ss<<=1; sns>>=1;
        a -= sns;
        f = fc[s];
        matrix -= n;

    } else {
        ++matrix[ __builtin_popcountll(f)];
    }
}

uint64_t* compute(int safen, int safel) {
    uint64_t* fcsafe = new uint64_t[(safen+1)/2];
    uint64_t *safematrix = new uint64_t[(safen+1)/2*safen];

    memset(safematrix, 0, (safen+1)/2*safen*sizeof(uint64_t));

    f = 0;
    a = 1;
    ss = 1ull << ((safen-1)/2);
    sns = 1ull << (safen - (safen-1)/2);
    n = safen;
    l = safel;
    s=(safen-3)/2;
    fc = fcsafe;
    matrix = safematrix;
    if (n >= 7) {
        ss <<= 1;
        recurse_unrolled();
    } else {
        recurse();
    }
    memset(safematrix, 0, safel*safen*sizeof(uint64_t));
    return safematrix;
}

int main(int argc, char *argv[])
{
    int safen, safel;
    if (argc > 3) {
        std::cerr << "Too many arguments." << std::endl;
        return 1;
    } else if (argc >= 2) {
        std::istringstream ss(argv[1]);
        if (!(ss >> safen)) {
            std::cerr << "Bad n." << std::endl;
            return 1;
        }
        if (argc == 3) {
            std::istringstream ss2(argv[2]);
            if (!(ss2 >> safel)) {
                std::cerr << "Bad l." << std::endl;
                return 1;
            }
        } else {
            safel = 0;
            std::cerr << "Defaulting to l=0;" << std::endl;
        }
    } else {
        std::cerr << "No argument, defaulting to n=44, l=0 as demo." << std::endl;
        safen=44;
        safel=0;
    }

    uint64_t *safematrix = compute(safen, safel);
    for (int i = 0; i < (safen+1)/2; ++i) {
        for (int j = 0; j < safen; ++j) {
            std::cout << safematrix[safen*i+safen-1-j] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}


