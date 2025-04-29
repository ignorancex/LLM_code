#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// #include "csr.h"
// #include "hart.h"
// #include "handler.h"
// #include "core_v_mini_mcu.h"
// #include "gpio.h"

// #define VCD_TRIGGER_GPIO 0

// static gpio_t gpio;

// void dump_on(void);
// void dump_off(void);

int padd_test() {
    uint16_t a, b, c, d, e, f, g;

    a = 0x7F94;  // Posit  671088640
    b = 0x7F6F;  // Posit  197132288
    c = 0x806E;  // Posit -603979776
    d = 0;
    e = 0;
    f = 0;
    g = 0;

    asm volatile(
        "plw    pt0,0(%4)      \n"
        "plw    pt1,0(%5)      \n"
        "plw    pt2,0(%6)      \n"

        "padd.s pt3,pt0,pt1    \n"
        "padd.s pt4,pt1,pt0    \n"
        "psw    pt3,0(%7)      \n"
        "psw    pt4,0(%8)      \n"

        "padd.s pt5,pt2,pt1    \n"
        "padd.s pt6,pt0,pt2    \n"
        "psw    pt5,0(%9)      \n"
        "psw    pt6,0(%10)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f), "=rm"(g)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g)
        :);

    if (d == 0x7F9A && e == 0x7F9A && f == 0x8078 && g == 0x7F40) {
        printf("PADD test OK\n");
        return 0;
    } else {
        printf("PADD test FAIL - Values: %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g);
        return 1;
    }
}

int psub_test() {
    uint16_t a, b, c, d, e, f, g;

    a = 0x7F94;  // Posit  671088640
    b = 0x7F6F;  // Posit  197132288
    c = 0x806E;  // Posit -603979776
    d = 0;
    e = 0;
    f = 0;
    g = 0;

    asm volatile(
        "plw    pt0,0(%4)      \n"
        "plw    pt1,0(%5)      \n"
        "plw    pt2,0(%6)      \n"

        "psub.s pt3,pt0,pt1    \n"  // 7f8b987e
        "psub.s pt4,pt1,pt0    \n"  // 80747682
        "psw    pt3,0(%7)      \n"
        "psw    pt4,0(%8)      \n"

        "psub.s pt5,pt2,pt1    \n"  // 80689423
        "psub.s pt6,pt0,pt2    \n"  // 7fa2984e
        "psw    pt5,0(%9)      \n"
        "psw    pt6,0(%10)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f), "=rm"(g)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g)
        :);

    if (d == 0x7F8C && e == 0x8074 && f == 0x8068 && g == 0x7FA3) {
        printf("PSUB test OK\n");
        return 0;
    } else {
        printf("PSUB test FAIL - Values: %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g);
        return 1;
    }
}

int pmul_test() {
    uint16_t a, b, c, d, e, f, g;

    a = 0x6F12;  // Posit  226.25
    b = 0x7B9B;  // Posit  52608
    c = 0x87E9;  // Posit -4464
    d = 0;
    e = 0;
    f = 0;
    g = 0;

    asm volatile(
        "plw    pt0,0(%4)      \n"
        "plw    pt1,0(%5)      \n"
        "plw    pt2,0(%6)      \n"

        "pmul.s pt3,pt0,pt1    \n"
        "pmul.s pt4,pt1,pt0    \n"
        "psw    pt3,0(%7)      \n"
        "psw    pt4,0(%8)      \n"

        "pmul.s pt5,pt2,pt1    \n"
        "pmul.s pt6,pt0,pt2    \n"
        "psw    pt5,0(%9)      \n"
        "psw    pt6,0(%10)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f), "=rm"(g)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g)
        :);

    if (d == 0x7EDB && e == 0x7EDB && f == 0x8088 && g == 0x8209) {
        printf("PMUL test OK\n");
        return 0;
    } else {
        printf("PMUL test FAIL - Values: %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g);
        return 1;
    }
}

int pdiv_test() {
    uint16_t a, b, c, d, e, f, g, h, zero;

    a = 0x6F11;  // Posit  226.125
    b = 0x7B9B;  // Posit  52608
    c = 0x87E9;  // Posit -4464
    e = 0;
    f = 0;
    g = 0;
    h = 1;
    zero = 0;

    asm volatile(
        "plw    pt0,0(%5)      \n"
        "plw    pt1,0(%6)      \n"
        "plw    pt2,0(%7)      \n"

        "pdiv.s pt3,pt0,pt1    \n"  // 0.00429916 -> 0x1067
        "pdiv.s pt4,pt1,pt0    \n"  // 232.625 -> 0x6F45
        "psw    pt3,0(%8)      \n"
        "psw    pt4,0(%9)      \n"

        "pdiv.s pt3,pt2,pt1    \n"  // -0.0848389 -> 0xDD24
        "pdiv.s pt4,pt0,pt2    \n"  // -0.0506592 -> 0xE184
        "plw    pt6,0(%13)     \n"
        "pdiv.s pt5,pt0,pt6    \n"  // Division by zero
        "psw    pt3,0(%10)     \n"
        "psw    pt4,0(%11)     \n"
        "psw    pt5,0(%12)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f), "=rm"(g), "=rm"(h)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g), "r"(&h), "r"(&zero)
        :);

    if (d == 0x1067 && e == 0x6F45 && f == 0xDD24 && g == 0xE184 && h == 0x8000) {
        printf("PDIV test OK\n");
        return 0;
    } else {
        printf("PDIV test FAIL - Values: %x %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g, h);
        return 1;
    }
}

int psqrt_test() {
    uint16_t a, b, c, d, e, f;

    a = 0x6F11;  // Posit  226.125
    b = 0x7B9B;  // Posit  52608
    c = 0x87E9;  // Posit -4464
    d = 0;
    e = 0;
    f = 0;

    asm volatile(
        "plw    pt0,0(%3)     \n"
        "plw    pt1,0(%4)     \n"
        "plw    pt2,0(%5)     \n"

        "psqrt.s pt3,pt0      \n"  // 15.0391 -> 0x5F0A
        "psqrt.s pt4,pt1      \n"  // 229.375 -> 0x6F2B
        "psqrt.s pt5,pt2      \n"  // NaR -> 80000000
        "psw    pt3,0(%6)     \n"
        "psw    pt4,0(%7)     \n"
        "psw    pt5,0(%8)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        :);

    if (d == 0x5F0A && e == 0x6F2B && f == 0x8000) {
        printf("PSQRT test OK\n");
        return 0;
    } else {
        printf("PSQRT test FAIL - Values: %x %x %x %x %x %x\n", a, b, c, d, e, f);
        return 1;
    }
}

int pmin_test() {
    uint16_t a, b, c, d, e, f, g;

    a = 0x7F94;  // Posit  671088640
    b = 0x7F6F;  // Posit  197132288
    c = 0x806E;  // Posit -603979776
    d = 0;
    e = 0;
    f = 0;
    g = 0;

    asm volatile(
        "plw    pt0,0(%4)      \n"
        "plw    pt1,0(%5)      \n"
        "plw    pt2,0(%6)      \n"

        "pmin.s pt3,pt0,pt1    \n"
        "pmin.s pt4,pt1,pt0    \n"
        "pmin.s pt5,pt2,pt0    \n"
        "pmin.s pt6,pt2,pt2    \n"
        "psw    pt3,0(%7)      \n"
        "psw    pt4,0(%8)      \n"
        "psw    pt5,0(%9)      \n"
        "psw    pt6,0(%10)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f), "=rm"(g)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g)
        :);

    if (d == 0x7F6F && e == 0x7F6F && f == 0x806E && g == 0x806E) {
        printf("PMIN test OK\n");
        return 0;
    } else {
        printf("PMIN test FAIL - Values: %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g);
        return 1;
    }
}

int pmax_test() {
    uint16_t a, b, c, d, e, f, g;

    a = 0x7F94;  // Posit  671088640
    b = 0x7F6F;  // Posit  197132288
    c = 0x806E;  // Posit -603979776
    d = 0;
    e = 0;
    f = 0;
    g = 0;

    asm volatile(
        "plw    pt0,0(%4)      \n"
        "plw    pt1,0(%5)      \n"
        "plw    pt2,0(%6)      \n"

        "pmax.s pt3,pt0,pt1    \n"
        "pmax.s pt4,pt1,pt0    \n"
        "pmax.s pt5,pt2,pt0    \n"
        "pmax.s pt6,pt2,pt2    \n"
        "psw    pt3,0(%7)      \n"
        "psw    pt4,0(%8)      \n"
        "psw    pt5,0(%9)      \n"
        "psw    pt6,0(%10)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f), "=rm"(g)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g)
        :);

    if (d == 0x7F94 && e == 0x7F94 && f == 0x7F94 && g == 0x806E) {
        printf("PMAX test OK\n");
        return 0;
    } else {
        printf("PMAX test FAIL - Values: %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g);
        return 1;
    }
}

int quire_test() {
    uint16_t a, b, c, d, e, f;

    a = 0x4000;  // Posit 1
    b = 0x4000;  // Posit 1
    c = 0x4C00;  // Posit 3
    d = 0;
    e = 0;
    f = 0;

    asm volatile(
        "plw    pt3,0(%3)     \n"
        "plw    pt4,0(%4)     \n"
        "plw    pt5,0(%5)     \n"

        "qclr.s               \n"  // Quire: 0
        "qmadd.s  pt3,pt4     \n"  // Quire: 1
        "qmadd.s  pt5,pt5     \n"  // Quire: 10
        "qmsub.s  pt4,pt5     \n"  // Quire: 7
        "qround.s pt6         \n"
        "psw      pt6,0(%6)   \n"
        "qneg.s               \n"  // Quire: -7
        "qround.s pt6         \n"
        "psw      pt6,0(%7)   \n"
        "qclr.s               \n"  // Quire: 0
        "qround.s pt6         \n"
        "psw      pt6,0(%8)   \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        :);

    if (d == 0x5600 && e == 0xAA00 && f == 0x0000) {
        printf("QUIRE test OK\n");
        return 0;
    } else {
        printf("QUIRE test FAIL - Values: %x %x %x %x %x %x\n", a, b, c, d, e, f);
        return 1;
    }
}

int pcvtws_test() {  // Pos to int
    uint16_t a, b, c;
    int32_t d, e, f;

    a = 0x7C01;
    b = 0x7A32;
    c = 0x8433;
    d = 0;
    e = 0;
    f = 0;

    asm volatile(
        "plw    pt0,0(%3)     \n"
        "plw    pt1,0(%4)     \n"
        "plw    pt2,0(%5)     \n"

        "pcvt.w.s t4,pt0      \n"
        "pcvt.w.s t5,pt1      \n"
        "pcvt.w.s t6,pt2      \n"
        "sw       t4,0(%6)    \n"
        "sw       t5,0(%7)    \n"
        "sw       t6,0(%8)    \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        : "t4", "t5", "t6");

    if (d == 66048 && e == 19584 && f == -59008) {
        printf("PCVTWS test OK\n");
        return 0;
    } else {
        printf("PCVTWS test FAIL - Values: %x %x %x %d %d %d\n", a, b, c, d, e, f);
        return 1;
    }
}

int pcvtwus_test() {  // Pos to unsigned int
    uint16_t a, b, c;
    uint32_t d, e, f;

    a = 0x7D20;
    b = 0x7CC0;
    c = 0x8433;
    d = 0;
    e = 0;
    f = 1;

    asm volatile(
        "plw    pt0,0(%3)     \n"
        "plw    pt1,0(%4)     \n"
        "plw    pt2,0(%5)     \n"

        "pcvt.wu.s t4,pt0     \n"
        "pcvt.wu.s t5,pt1     \n"
        "pcvt.wu.s t6,pt2     \n"
        "sw       t4,0(%6)    \n"
        "sw       t5,0(%7)    \n"
        "sw       t6,0(%8)    \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        : "t4", "t5", "t6");

    if (d == 327680 && e == 196608 && f == 0) {
        printf("PCVTWUS test OK\n");
        return 0;
    } else {
        printf("PCVTWUS test FAIL - Values: %x %x %x %d %d %d\n", a, b, c, d, e, f);
        return 1;
    }
}

int pcvtsw_test() {  // Int to pos
    int32_t a, b, c;
    uint16_t d, e, f;

    a = 658124290;   // Round -> 671088640 (Posit 0x7F94)
    b = 196128700;   // Round -> 197132288 (Posit 0x7F6F)
    c = -589761023;  // Round -> -603979776 (Posit 0x806E)
    d = 0;
    e = 0;
    f = 0;

    asm volatile(
        "lw       t0,0(%3)   \n"
        "lw       t1,0(%4)   \n"
        "lw       t2,0(%5)   \n"

        "pcvt.s.w pt4,t0     \n"
        "pcvt.s.w pt5,t1     \n"
        "pcvt.s.w pt6,t2     \n"
        "psw      pt4,0(%6)  \n"
        "psw      pt5,0(%7)  \n"
        "psw      pt6,0(%8)  \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        : "t0", "t1", "t2");

    if (d == 0x7F94 && e == 0x7F6F && f == 0x806E) {
        printf("PCVTSW test OK\n");
        return 0;
    } else {
        printf("PCVTSW test FAIL - Values: %d %d %d %x %x %x\n", a, b, c, d, e, f);
        return 1;
    }
}

int pcvtswu_test() {  // Unsigned int to pos
    uint32_t a, b, c;
    uint16_t d, e, f;

    a = 3285968897;  // Round -> 3221225472 (Posit 0x7FB8)
    b = 196128700;   // Round -> 197132288 (Posit 0x7F6F)
    c = 0;           // Round -> 0 (Posit 0x0000)
    d = 0;
    e = 0;
    f = 0;

    asm volatile(
        "lw        t0,0(%3)   \n"
        "lw        t1,0(%4)   \n"
        "lw        t2,0(%5)   \n"

        "pcvt.s.wu pt4,t0     \n"
        "pcvt.s.wu pt5,t1     \n"
        "pcvt.s.wu pt6,t2     \n"
        "psw       pt4,0(%6)  \n"
        "psw       pt5,0(%7)  \n"
        "psw       pt6,0(%8)  \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        : "t0", "t1", "t2");

    if (d == 0x7FB8 && e == 0x7F6F && f == 0) {
        printf("PCVTSWU test OK\n");
        return 0;
    } else {
        printf("PCVTSWU test FAIL - Values: %d %d %d %x %x %x\n", a, b, c, d, e, f);
        return 1;
    }
}

// Sign injection
// Takes all bits except the sign bit from rs1 and the sign bit from rs2.
int psgnj_test() {
    uint16_t a, b, c, d, e, f, g, h;

    a = 0x7839;  // Posit  5008
    b = 0x7568;  // Posit  1744
    c = 0x87C7;  // Posit -5008
    d = 0x89B6;  // Posit -2344
    e = 0;
    f = 0;
    g = 0;
    h = 0;

    asm volatile(
        "plw     pt0,0(%4)      \n"
        "plw     pt1,0(%5)      \n"
        "plw     pt2,0(%6)      \n"
        "plw     pt3,0(%7)      \n"

        "psgnj.s pt4,pt0,pt1    \n"
        "psgnj.s pt5,pt0,pt2    \n"
        "psw     pt4,0(%8)      \n"
        "psw     pt5,0(%9)      \n"
        "psgnj.s pt4,pt2,pt1    \n"
        "psgnj.s pt5,pt2,pt3    \n"
        "psw     pt4,0(%10)     \n"
        "psw     pt5,0(%11)     \n"

        : "=rm"(e), "=rm"(f), "=rm"(g), "=rm"(h)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g), "r"(&h)
        :);

    if (e == 0x7839 && f == 0x87C7 && g == 0x7839 && h == 0x87C7) {
        printf("PSGNJ test OK\n");
        return 0;
    } else {
        printf("PSGNJ test FAIL - Values: %x %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g, h);
        return 1;
    }
}

// Takes all bits except the sign bit from rs1 and the sign bit the opposite from rs2.
int psgnjn_test() {
    uint16_t a, b, c, d, e, f, g, h;

    a = 0x7839;  // Posit  5008
    b = 0x7568;  // Posit  1744
    c = 0x87C7;  // Posit -5008
    d = 0x89B6;  // Posit -2344
    e = 0;
    f = 0;
    g = 0;
    h = 0;

    asm volatile(
        "plw     pt0,0(%4)      \n"
        "plw     pt1,0(%5)      \n"
        "plw     pt2,0(%6)      \n"
        "plw     pt3,0(%7)      \n"

        "psgnjn.s pt4,pt0,pt1   \n"
        "psgnjn.s pt5,pt0,pt2   \n"
        "psw      pt4,0(%8)     \n"
        "psw      pt5,0(%9)     \n"
        "psgnjn.s pt4,pt2,pt1   \n"
        "psgnjn.s pt5,pt2,pt3   \n"
        "psw      pt4,0(%10)    \n"
        "psw      pt5,0(%11)    \n"

        : "=rm"(e), "=rm"(f), "=rm"(g), "=rm"(h)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g), "r"(&h)
        :);

    if (e == 0x87C7 && f == 0x7839 && g == 0x87C7 && h == 0x7839) {
        printf("PSGNJN test OK\n");
        return 0;
    } else {
        printf("PSGNJN test FAIL - Values: %x %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g, h);
        return 1;
    }
}

// Takes all bits except the sign bit from rs1 and the sign bit the xor
// of the sign bits of rs1 and rs2.
int psgnjnx_test() {
    uint16_t a, b, c, d, e, f, g, h;

    a = 0x7839;  // Posit  5008
    b = 0x7568;  // Posit  1744
    c = 0x87C7;  // Posit -5008
    d = 0x89B6;  // Posit -2344
    e = 0;
    f = 0;
    g = 0;
    h = 0;

    asm volatile(
        "plw     pt0,0(%4)      \n"
        "plw     pt1,0(%5)      \n"
        "plw     pt2,0(%6)      \n"
        "plw     pt3,0(%7)      \n"

        "psgnjx.s pt4,pt0,pt1   \n"
        "psgnjx.s pt5,pt0,pt2   \n"
        "psw      pt4,0(%8)     \n"
        "psw      pt5,0(%9)     \n"
        "psgnjx.s pt4,pt2,pt1   \n"
        "psgnjx.s pt5,pt2,pt3   \n"
        "psw      pt4,0(%10)    \n"
        "psw      pt5,0(%11)    \n"

        : "=rm"(e), "=rm"(f), "=rm"(g), "=rm"(h)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g), "r"(&h)
        :);

    if (e == 0x7839 && f == 0x87C7 && g == 0x87C7 && h == 0x7839) {
        printf("PSGNJNX test OK\n");
        return 0;
    } else {
        printf("PSGNJNX test FAIL - Values: %x %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g, h);
        return 1;
    }
}

int pmvxw_test() {  // Move pos to int reg
    uint16_t a, b, c;
    int32_t d, e, f;

    a = 0x7F93;
    b = 0x7F6E;
    c = 0x806E;
    d = 0;
    e = 0;
    f = 0;

    asm volatile(
        "plw    pt0,0(%3)      \n"
        "plw    pt1,0(%4)      \n"
        "plw    pt2,0(%5)      \n"

        "pmv.x.w  t4,pt0       \n"
        "pmv.x.w  t5,pt1       \n"
        "pmv.x.w  t6,pt2       \n"
        "sw       t4,0(%6)     \n"
        "sw       t5,0(%7)     \n"
        "sw       t6,0(%8)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        : "t4", "t5", "t6");

    if (d == 32659 && e == 32622 && f == -32658) {
        printf("PMVXW test OK\n");
        return 0;
    } else {
        printf("PMVXW test FAIL - Values: %x %x %x %d %d %d\n", a, b, c, d, e, f);
        return 1;
    }
}

int pmvwx_test() {  // Move int to pos reg
    int32_t a, b, c;
    uint16_t d, e, f;

    a = 32659;
    b = 32622;
    c = -32658;
    d = 0;
    e = 0;
    f = 0;

    asm volatile(
        "lw       t0,0(%3)     \n"
        "lw       t1,0(%4)     \n"
        "lw       t2,0(%5)     \n"

        "pmv.w.x  pt4,t0       \n"
        "pmv.w.x  pt5,t1       \n"
        "pmv.w.x  pt6,t2       \n"
        "psw      pt4,0(%6)    \n"
        "psw      pt5,0(%7)    \n"
        "psw      pt6,0(%8)    \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        : "t0", "t1", "t2");

    if (d == 0x7F93 && e == 0x7F6E && f == 0x806E) {
        printf("PMVWX test OK\n");
        return 0;
    } else {
        printf("PMVWX test FAIL - Values: %x %x %x %x %x %x\n", a, b, c, d, e, f);
        return 1;
    }
}

int peq_test() {
    uint16_t a, b, c;
    int32_t d, e, f;

    a = 0x7F94;  // Posit  671088640
    b = 0x7F6F;  // Posit  197132288
    c = 0x806E;  // Posit -603979776
    d = 1;
    e = 0;
    f = 0;

    asm volatile(
        "plw    pt0,0(%3)      \n"
        "plw    pt1,0(%4)      \n"
        "plw    pt2,0(%5)      \n"

        "peq.s  t3,pt0,pt1    \n"
        "peq.s  t4,pt1,pt1    \n"
        "peq.s  t5,pt2,pt2    \n"
        "sw     t3,0(%6)      \n"
        "sw     t4,0(%7)      \n"
        "sw     t5,0(%8)      \n"

        : "=rm"(d), "=rm"(e), "=rm"(f)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f)
        : "t3", "t4", "t5");

    if (d == 0 && e == 1 && f == 1) {
        printf("PEQ test OK\n");
        return 0;
    } else {
        printf("PEQ test FAIL - Values: %x %x %x %x %x %x\n", a, b, c, d, e, f);
        return 1;
    }
}

int plt_test() {
    uint16_t a, b, c;
    int32_t d, e, f, g, h;

    a = 0x7F94;  // Posit  671088640
    b = 0x7F6F;  // Posit  197132288
    c = 0x806E;  // Posit -603979776
    d = 1;
    e = 0;
    f = 0;
    g = 1;
    h = 1;

    asm volatile(
        "plw    pt0,0(%5)     \n"
        "plw    pt1,0(%6)     \n"
        "plw    pt2,0(%7)     \n"

        "plt.s  t3,pt0,pt1    \n"
        "plt.s  t4,pt1,pt0    \n"
        "plt.s  t5,pt2,pt0    \n"
        "plt.s  t6,pt0,pt2    \n"
        "sw     t3,0(%8)      \n"
        "sw     t4,0(%9)      \n"
        "sw     t5,0(%10)     \n"
        "sw     t6,0(%11)     \n"
        "plt.s  t6,pt2,pt2    \n"
        "sw     t6,0(%12)     \n"

        : "=rm"(d), "=rm"(e), "=rm"(f), "=rm"(g), "=rm"(h)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g), "r"(&h)
        : "t3", "t4", "t5", "t6");

    if (d == 0 && e == 1 && f == 1 && g == 0 && h == 0) {
        printf("PLT test OK\n");
        return 0;
    } else {
        printf("PLT test FAIL - Values: %x %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g, h);
        return 1;
    }
}

int ple_test() {
    uint16_t a, b, c;
    int32_t d, e, f, g;

    a = 0x7F94;  // Posit  671088640
    b = 0x7F6F;  // Posit  197132288
    c = 0x806E;  // Posit -603979776
    d = 1;
    e = 0;
    f = 0;
    g = 0;

    asm volatile(
        "plw    pt0,0(%4)  \n"
        "plw    pt1,0(%5)  \n"
        "plw    pt2,0(%6)  \n"

        "ple.s  t3,pt0,pt1    \n"
        "ple.s  t4,pt1,pt0    \n"
        "ple.s  t5,pt2,pt0    \n"
        "ple.s  t6,pt2,pt2    \n"
        "sw     t3,0(%7)  \n"
        "sw     t4,0(%8)  \n"
        "sw     t5,0(%9)  \n"
        "sw     t6,0(%10)  \n"

        : "=rm"(d), "=rm"(e), "=rm"(f), "=rm"(g)
        : "r"(&a), "r"(&b), "r"(&c), "r"(&d), "r"(&e), "r"(&f), "r"(&g)
        : "t3", "t4", "t5", "t6");

    if (d == 0 && e == 1 && f == 1 && g == 1) {
        printf("PLE test OK\n");
        return 0;
    } else {
        printf("PLE test FAIL - Values: %x %x %x %x %x %x %x\n", a, b, c, d, e, f, g);
        return 1;
    }
}

// void dump_on(void) {
//   gpio_params_t gpio_params;
//   gpio_params.base_addr = mmio_region_from_addr((uintptr_t)GPIO_AO_START_ADDRESS);
//   gpio_init(gpio_params, &gpio);
//   gpio_output_set_enabled(&gpio, VCD_TRIGGER_GPIO, true);

//   gpio_write(&gpio, VCD_TRIGGER_GPIO, true);
// }

// void dump_off(void) {
//   gpio_write(&gpio, VCD_TRIGGER_GPIO, false);
// }

int main() {
    printf("Hello PHEE16!\n");

    padd_test();
    psub_test();
    pmul_test();
    pdiv_test();
    psqrt_test();
    pmin_test();
    pmax_test();

    quire_test();

    pcvtws_test();
    pcvtwus_test();
    pcvtsw_test();
    pcvtswu_test();

    psgnj_test();
    psgnjn_test();
    psgnjnx_test();

    pmvxw_test();
    pmvwx_test();

    peq_test();
    plt_test();
    ple_test();

    return 0;
}
