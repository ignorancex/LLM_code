// Copyright 2022 Robert Guralnick, John Shareshian, and Russ Woodroofe

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <sys/param.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <primesieve.h>

// Size (in bytes) of a "window" of numbers with largest pp divisor identified
//  (We'll need a bunch of additional scratch memory to compute the pp divs)
#define WNDSIZE (65536)

// type to use for factors.  Need to fit log^2(n)*5, so 4 bytes should take
//  us out quite a ways indeed.
// smaller type will fit more in cache
#define INT_F_T uint32_t
#define INT_F_SIZE (sizeof(INT_F_T))
#define WND_LEN (WNDSIZE / INT_F_SIZE)

long factors_segcurrent;
long factors_bigpp_marker;

INT_F_T *factors_ppdivswnd=NULL;
long *factors_partlyfactored=NULL; // need long for multiplying factors
long *factors_wheelfactors=NULL;
INT_F_T *factors_wheelppdivs=NULL;
#define FACTORS_WHEELLEN (2*3*2*5*7*2*3)
#define FACTORS_WHEEL_NUMP 7

struct primepower
{
  INT_F_T pp;
  INT_F_T p;
};
struct primepower *factors_smallpps=NULL;
long factors_smallpps_count=0;

#define FREE(ptr)     ({ if (ptr) { free(ptr); ptr=NULL; } })

long log2int(const long arg)
{
  long i = 0, j=arg;
  if (arg==0)  // should never happen
    return -1;
  while (j >>= 1) ++i;
  return i;
}

void fastfactor_destroy()
{
  FREE(factors_smallpps);
  FREE(factors_ppdivswnd);
  FREE(factors_partlyfactored);
  FREE(factors_wheelfactors);
  FREE(factors_wheelppdivs);
}

void fastfactor_init(long m)
{
  long i,j;
  INT_F_T p,pp;

  // a value big enough, assuming a medium weak form of Cramer's conjecture,
  //  that we should never see a prime gap of this size
  factors_bigpp_marker = log2int(m)*log2int(m)*5;

  //allocate plenty of space for smallpps
  factors_smallpps = valloc(sizeof(struct primepower)*factors_bigpp_marker);

  factors_ppdivswnd = valloc(WND_LEN * INT_F_SIZE);
  factors_partlyfactored = valloc(WND_LEN * sizeof(long));
  factors_wheelfactors = valloc(sizeof(long) * FACTORS_WHEELLEN);
  factors_wheelppdivs = valloc(INT_F_SIZE * FACTORS_WHEELLEN);

  if (!factors_smallpps || !factors_ppdivswnd || !factors_partlyfactored
      || !factors_wheelfactors || !factors_wheelppdivs)
  {
    printf("# ERROR: unable to allocate memory.\n");
    exit(1);
  }

  factors_segcurrent = -1;
  for (i=0; i < WND_LEN; i++)
    factors_ppdivswnd[i]=1;

  if (WND_LEN <= factors_bigpp_marker)
  {
    printf("# ERROR: unexpectedly big end number found.\n");
    exit(1);
  }

  for (i=1; i<factors_bigpp_marker; i++)
  {
    p = ( (i+1) / factors_ppdivswnd[i]); // possibly prime
    // If the largest pp div of i+1 is divisible by p, then p is a prime
    //  and i+1 is a pp.
    if (factors_ppdivswnd[i]==1 || (factors_ppdivswnd[i] % p) == 0 )
    {
      factors_smallpps[factors_smallpps_count].pp=i+1;
      factors_smallpps[factors_smallpps_count++].p = p;
      for (j=i ; j <= WND_LEN ; j+=i+1 )
        factors_ppdivswnd[j]=i+1;
    }
  }

  // initialize wheelfactors and wheelppdivs to 1
  for (i=0; i<FACTORS_WHEELLEN; i++)
  {
    factors_wheelfactors[i]=1;
    factors_wheelppdivs[i]=1;
  }

  for (i=0; i<FACTORS_WHEEL_NUMP; i++)
  {
    pp=factors_smallpps[i].pp;
    p=factors_smallpps[i].p;
    for (j=pp-1; j < FACTORS_WHEELLEN; j+=pp )
    {
      factors_wheelppdivs[j]=pp;
      factors_wheelfactors[j]*=p;
    }
  }
}

void fastfactor_setsegment(long segment)
{
  long seg_zero, seg_end, i, j, pp, p, smooth_threshold;

  // consider numbers from WND_LEN * segment + 1 to WND_LEN*(segment+1)
  seg_zero = WND_LEN * segment;
  seg_end = seg_zero + WND_LEN;

  factors_segcurrent = segment;

  // Replacing this initialization with a faster memsetter doesn't seem to help,
  //  at least under macOS
  for (i=0 ; i < WND_LEN ; i++)
    factors_ppdivswnd[i] = 1;

  // for segment 0, we will have some numbers whose greatest pp divisor is in the
  //  'wheel'.  Some special handling is needed here and when marking large pp divisors.
  if (segment == 0)
    memcpy(factors_ppdivswnd, factors_wheelppdivs, sizeof(INT_F_T) * FACTORS_WHEELLEN);

  // initialize partial factorizations with those by small primes in one step
  // (this complicated-looking code just cyclically copies from factors_wheelfactors)
  memcpy(factors_partlyfactored,
         factors_wheelfactors + (seg_zero % FACTORS_WHEELLEN),
         sizeof(long)*(FACTORS_WHEELLEN - (seg_zero % FACTORS_WHEELLEN) ) );
  for (i=1; (i+1)*FACTORS_WHEELLEN - (seg_zero % FACTORS_WHEELLEN)< WND_LEN ; i++ )
  {
    memcpy( factors_partlyfactored + i*FACTORS_WHEELLEN - (seg_zero % FACTORS_WHEELLEN),
           factors_wheelfactors, sizeof(long)*FACTORS_WHEELLEN);
  }
  memcpy(factors_partlyfactored + i*FACTORS_WHEELLEN - (seg_zero % FACTORS_WHEELLEN),
           factors_wheelfactors,
           sizeof(long)*(WND_LEN - i*FACTORS_WHEELLEN + seg_zero % FACTORS_WHEELLEN ));

  // look for smallpp divisors of each number in window
  for (i=FACTORS_WHEEL_NUMP; i<factors_smallpps_count ; i++)
  {
    pp = factors_smallpps[i].pp;
    p  = factors_smallpps[i].p;

    for (j=(seg_zero/pp)*pp + pp - seg_zero - 1; j < WND_LEN; j+=pp )
    {
      factors_ppdivswnd[j] = pp;
      factors_partlyfactored[j] *= p;
    }
  }

  smooth_threshold = segment ? seg_zero : FACTORS_WHEELLEN;
  for (i= segment ? 0 : FACTORS_WHEELLEN; i < WND_LEN; i++ )
  {
    // if we didn't find all the factors, then we have a big enough pp
    //  factor that we'll just mark it as "non-power-smooth"
    //if (i + seg_zero + 1 == 9999248256)  // debug code to inspect factors of particular n
    //  printf("# db %ld %ld %u\n", i + seg_zero + 1, factors_partlyfactored[i], factors_ppdivswnd[i]);
    if (factors_partlyfactored[i] <= smooth_threshold)
      factors_ppdivswnd[i] = factors_bigpp_marker;
  }
}

void fastfactor_test_print_wheel()
{
  long i;
  printf("\n");
  for (i=0; i < FACTORS_WHEELLEN; i++)
    printf("%ld ", factors_wheelfactors[i]);
  printf("\n");
}

void fastfactor_test_print_smallpps()
{
  long i;
  for (i=0; i < factors_smallpps_count; i++ )
  {
    printf("%u %u\n",factors_smallpps[i].p,factors_smallpps[i].pp);
  }
}

void fastfactor_test_printfactors(long m)
{
  long i, j, seg_zero, smoothcount=0;
  printf("Smoothness threshold: %ld", factors_bigpp_marker);
  for (i=1 ; i < m / WND_LEN; i++)
  {
    fastfactor_setsegment(i);
    printf("==Segment %ld==\n",i);
    seg_zero = WND_LEN * i;
    for (j=0; j <WND_LEN ; j++)
    {
      if (factors_ppdivswnd[j] < factors_bigpp_marker)
      {
        printf("%ld  |  %ld\n", (long)factors_ppdivswnd[j], seg_zero+j+1);
        smoothcount++;
      }
      else
      {
        printf("%ld is not power smooth\n", seg_zero+j+1);
      }
    }
  }
  printf("======\n%ld power smooth numbers considered\n", smoothcount);
}

int main(int argc, char* argv[])
{
  long n, seg_zero, seg_end, l=0,m, smoothcount=0, bigprime, nextprime;
  clock_t starttime = clock();

  // parse command line
  if (argc <= 1)
  {
    printf("Specify a range for computation\n");
    exit(1);
  }
  else if (argc == 2)
  {
    m=atol(argv[1]);
  }
  else if (argc == 3)
  {
    l=atol(argv[1]);
    m=atol(argv[2]);
  }
  l=MAX(l,25);
  m=MAX(m,l);

  fprintf(stderr, "# Looking for over-smooth numbers between %ld and %ld.\n", l,m);
  printf("invgen_oversmooth_range:=[%ld..%ld];\n", l, m);

  fastfactor_init(m);
  primesieve_iterator it;
  primesieve_init(&it);

  primesieve_skipto(&it, l-2, m);
  bigprime=primesieve_prev_prime(&it);
  nextprime=primesieve_next_prime(&it);

  if (bigprime >= l-2 || nextprime < l-2)
    fprintf(stderr, "# Warning: initial primes out of range.\n");

  printf("invgen_oversmooth := [");
  seg_end=0;
  // we're indexing n as the integer we're looking at (counting starting at 1)
  //  rather than C conventions (starting at 0).  Need to subtract 1 for array accesses
  for (n=l ; n <= m ; n++)
  {
    // update the current segment, if necessary
    if (n > seg_end)
    {
      fastfactor_setsegment( (n-1) / WND_LEN );
      seg_zero = factors_segcurrent * WND_LEN;
      seg_end = seg_zero + WND_LEN;
    }

    if (bigprime + (long)(factors_ppdivswnd[n-1-seg_zero]) <= n)
    {
      // By [Jones:2014], a cycle fixing 2 points is in a primitive subgroup
      //  only if n-1 is a prime power.  If n-2 is prime, also n is a power of 2.
      // in this case, we can use n-2 as our bigprime
      if ( n-2 == nextprime && ( (2^(log2int(n-1))) != n-1) )
      {
        // we're good!
      }
      else
      {
        printf("\n  [ %ld, %ld ], # bp %ld ", n, (long)factors_ppdivswnd[n-1-seg_zero], bigprime);
        smoothcount++;
        if (factors_ppdivswnd[n-1-seg_zero] == factors_bigpp_marker)
        {
          fprintf(stderr, "# Smooth number %ld found with pp divisor marked as large\n", n);
          printf("\n# WARNING: Smooth number %ld found with pp divisor marked as large\n", n);
        }
      }
    }
    if (n==8999787328)
      printf("# %ld %ld %ld\n", n, bigprime, (long)factors_ppdivswnd[n-1-seg_zero]);

    if (n-2 == nextprime)
    {
      bigprime=nextprime;
      nextprime=primesieve_next_prime(&it);
      // NOTE: a possible improvement would be to increase n by the least
      //  prime power divisor of integers in our range, rather than 1. Would
      //  require keeping track of lots of corner cases, and would only avoid
      //  a few additions (though also avoid a bit of memory access)
    }
  }
  printf("\n];\n");

  primesieve_free_iterator(&it);
  fastfactor_destroy();

  printf("# %ld overly power-smooth numbers encountered in %.3lf s\n", smoothcount,
         (double) (clock() - starttime) / CLOCKS_PER_SEC  );
  return 0;

}
