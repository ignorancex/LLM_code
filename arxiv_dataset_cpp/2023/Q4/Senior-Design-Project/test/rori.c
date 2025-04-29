

#define XLEN 32
#include<stdint.h>
#define uint_xlen_t uint32_t

uint_xlen_t rotimm(uint_xlen_t rs1)
{
uint_xlen_t a = 0;
a = ((rs1>>2) | (rs1<<(XLEN-2)));
return a;
}
