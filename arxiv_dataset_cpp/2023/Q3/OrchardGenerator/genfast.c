#include <stdio.h>
#include "consts.h"

/*
  Constants are defined in consts.h
  The following are mandatory
  - N            number of leaves
  - R            number of reticulations
  - COND         restriction, could be none, tc, sf

  These are optional
  - PRINTCOUNT   print the final count
  - PRINTSEQ     print the generated MCRS
  - PARTIALS     generate all MCRSs with at most R reticulations
  - MORECOUNTS   print intermediate counts for all r=0..R
*/
#define L (N + R - 1)

/* Enums */
typedef enum { NONE = 0, TREE = 1, PARENT = 2, SIBLING = 3 } state;

/* Variables */
int S[2 * L];             // sequence (MCRS)
int R1[L][N * 2 / 3];     // first coordinate of ARP for each generation
int R2[L][N * 2 / 3];     // second coordinate of ARP for each generation
int Ch[L][N * 2 / 3];     // character of ARP for each generation
int C[L];                 // count of ARP for each generation
int Xcount;               // number of leaves in current network
state Xstate[N];          // state of the network
unsigned long long count; // total count
unsigned long long Rcounts[R + 1]; // total count for each reticulation

/* Declarations */
int condnone(int a);                // generate all orchard networks
int condsf(int a);                  // only stack-free
int condtc(int a);                  // only tree-child
int lt(int a, int b, int c, int d); // checks (a,b) < (c,d)
void increment(int k);              // increment the counters
int addpair(int a, int b, int isret, int k, int *chleaf, int *chstate);
int addcherry(int a, int b, int k);
int addretcherry(int a, int b, int k, int *chleaf, int *chstate);
int addsafe(int ai, int bi, int ci, int flip, int k);
void generate(int k);
void printSeq(int k);

/* Implementations */
int condnone(int a) { return 1; }
int condsf(int a) { return Xstate[a - 1] != PARENT; }
int condtc(int a) { return Xstate[a - 1] < 2; }
int lt(int a, int b, int c, int d) { return a < c || (a == c && b < d); }

void increment(int k) {
#ifdef PRINTCOUNT
    count++;
#ifdef MORECOUNTS
    Rcounts[k - Xcount + 1]++;
#endif
#endif
}

int addpair(int a, int b, int isret, int k, int *chleaf, int *chstate) {
    int r = isret ? addretcherry(a, b, k, chleaf, chstate) : addcherry(a, b, k);
    r     = r || C[k] == 1 || lt(a, b, R1[k][1], R2[k][1]);
    if (r) {
        // set first pair (a, b) (isret)
        R1[k][0] = a;
        R2[k][0] = b;
        Ch[k][0] = isret;
    }
    return r;
}

int addcherry(int a, int b, int k) {
    // Compute the new ARP and check if (a,b) is a candidate extension
    int smallest = 0; // if (a,b) is the smallest
    C[k]         = 1; // set first pair as (a, b)

    // transform all pairs
    for (int i = 0; i < C[k - 1]; i++) {
        // compare with first survival
        if (!smallest && C[k] > 1) {
            if (lt(a, b, R1[k][1], R2[k][1]))
                smallest = 1; // it's the smallest. Transform remaining pairs
            else
                return 0; // it's not the smallest. Avort.
        }

        // process pair (ai, bi) (ci)
        int ai = R1[k - 1][i];
        int bi = R2[k - 1][i];
        int ci = Ch[k - 1][i];

        // Case 1: remains invariant
        if (ai != b && bi != b) {
            R1[k][C[k]] = ai;
            R2[k][C[k]] = bi;
            Ch[k][C[k]] = ci;
            C[k]++;
        }
        // Case 2: do nothing
    }
    return smallest;
}

int addretcherry(int a, int b, int k, int *chleaf, int *chstate) {
    // Compute the new ARP, check if (a,b) is a candidate extension and
    // recalculate Xstate
    int smallest = 0;  // if (a,b) is the smallest
    int flip     = -1; // index of the flip (-1: not detected, -2: fixed)
    C[k]         = 1;  // set first pair as (a, b)

    // transform all pairs
    for (int i = 0; i < C[k - 1]; i++) {
        // compare with first survival
        if (!smallest && C[k] > 1) {
            if (lt(a, b, R1[k][1], R2[k][1]))
                smallest = 1; // it's the smallest. Transform remaining pairs
            else
                return 0; // it's not the smallest. Avort.
        }

        // process pair (ai, bi) (ci)
        int ai = R1[k - 1][i];
        int bi = R2[k - 1][i];
        int ci = Ch[k - 1][i];

        // Case 1: remains invariant
        if (ai != a && ai != b && bi != a && bi != b)
            flip = addsafe(ai, bi, ci, flip, k);

        // Case 2: from cherry to reticulated cherry (happens only once)
        else if (!ci && flip == -1) {
            if (a == ai && b != bi) {
                flip           = addsafe(ai, bi, 1, -2, k);
                *chleaf        = bi;
                *chstate       = Xstate[bi - 1];
                Xstate[bi - 1] = SIBLING;
            } else if (a == bi && b != ai) {
                flip           = i;
                *chleaf        = ai;
                *chstate       = Xstate[ai - 1];
                Xstate[ai - 1] = SIBLING;
            }
        }
        // Case 3: do nothing
    }
    // check if flip persists
    if (flip >= 0)
        addsafe(R2[k - 1][flip], R1[k - 1][flip], 1, -2, k);

    return smallest;
}

int addsafe(int ai, int bi, int ci, int flip, int k) {
    // When adding a reticulated cherry, handle correctly the flip
    // check if we have to change the flip
    if (flip >= 0 && lt(R2[k - 1][flip], R1[k - 1][flip], ai, bi)) {
        R1[k][C[k]] = R2[k - 1][flip];
        R2[k][C[k]] = R1[k - 1][flip];
        Ch[k][C[k]] = 1;
        C[k]++;
        flip = -2;
    }
    // add (ai, bi) (ci)
    R1[k][C[k]] = ai;
    R2[k][C[k]] = bi;
    Ch[k][C[k]] = ci;
    C[k]++;

    // update flip
    return flip;
}

void generate(int k) {
    for (int a = 1; a <= N; a++) {
        int isret = Xstate[a - 1] != 0; // 1: reticulated-cherry, 0: cherry

        // check if we are forced to add a cherry
        if (isret && (L - N == k - Xcount || !COND(a)))
            continue;

#ifdef PRINTSEQ
        S[2 * k] = a;
#endif
        int stA = Xstate[a - 1];
        Xstate[a - 1] = 1 + isret;
        Xcount += !isret;

        for (int b = 1 + a * (!isret); b <= N; b++) {
			// try to add (a, b)
            if (a == b || Xstate[b - 1] == NONE)
                continue;

            int chleaf = 1, chstate = Xstate[0];
            if (addpair(a, b, isret, k, &chleaf, &chstate)) {
                // it's the smallest, prepare for augmenting
#ifdef PRINTSEQ
                S[2 * k + 1] = b;
#endif
                int stB = Xstate[b - 1];
                Xstate[b - 1] = 1 + 2 * isret;
                if (k + 1 == L) { // end
                    increment(k + 1);
                    printSeq(k + 1);
                } else { // continue
#ifdef PARTIALS
                    if (Xcount == N) {
                        increment(k + 1);
                        printSeq(k + 1);
                    }
#endif
                    generate(k + 1); // keep going
                }
                // undo changes
                Xstate[b - 1] = stB;
            }
            // undo changes
            Xstate[chleaf - 1] = chstate;
        }
        // undo changes
        Xcount -= !isret;
        Xstate[a - 1] = stA;
    }
}

void printSeq(int k) {
#ifdef PRINTSEQ
    // NOTE: sequence is stored in reversed order
    for (int i = 2 * k; i > 0; i -= 2)
        printf("(%d,%d)", S[i - 2], S[i - 1]);
    printf("\n");
#endif
}

int main(int argc, char *argv[]) {
    count = 0;
#if defined(PRINTCOUNT) && defined(MORECOUNTS)
    for (int i = 0; i <= R; i++)
        Rcounts[i] = 0;
#endif

    for (int i = 1; i < N; i++) {
		// prepare network (i,N)
        C[0]     = 1;
        R1[0][0] = i;
        R2[0][0] = N;
        Ch[0][0] = 0;

		Xcount        = 2;
        Xstate[i - 1] = TREE;
        Xstate[N - 1] = TREE;

#ifdef PRINTSEQ
        S[0] = i;
        S[1] = N;
#endif
#if N == 2 && (defined(PARTIALS) || R == 0)
        printSeq(1);
        increment(1);
#endif
        generate(1);

        // prepare next iteration
        Xstate[i - 1] = NONE;
	}
#ifdef PRINTCOUNT
#ifdef MORECOUNTS
    for (int i = 0; i <= R; i++)
        printf("R = %d: %llu\n", i, Rcounts[i]);

    printf("Total: %llu\n", count);
#else
    printf("%llu\n", count);
#endif
#endif
    return 0;
}
