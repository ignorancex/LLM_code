#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "./check_syscalls.h"
#include "./io/stringparse.h"
#include <math.h>

#define EXTRA_HALO_INFO                                                        \
    int64_t descid, np;                                                        \
    float   alt_m[4], J[3], spin, bullock_spin, Xoff, Voff, b_to_a, c_to_a,    \
        A[3], klypin_rs, kin_to_pot, m_all, m_pe_b, m_pe_d, b_to_a2, c_to_a2,  \
      A2[3], halfmass_radius, rvmax, chi2, A_I[3], A2_I[3], \
      inertia_tensor[6], inertia_tensor2[6];

#include "read_tree.h"

double BOX_SIZE = 250;

struct halo_list all_halos = {0};
struct halo_tree halo_tree = {0};

#define GROUP_LIST     all_halos.halos
#define RADIUS         vmax_r
#define FAST3TREE_TYPE struct halo
#include "./fast3tree.c"
#define parent pid
#include "./parents.c"
#undef parent

void read_hlist(char *filename, float *bounds) {
    int64_t     i, n, c = 0;
    FILE       *input;
    struct halo h = {0};
    char        buffer[1024];

    SHORT_PARSETYPE;

    /*
#if defined(OUTPUT_RVMAX) && defined(OUTPUT_INERTIA_TENSOR)
  #define NUM_INPUTS 54
#elif defined(OUTPUT_RVMAX)
  #define NUM_INPUTS 42
#elif defined(OUTPUT_INERTIA_TENSOR)
  #define NUM_INPUTS 53
#else
  #define NUM_INPUTS 41
#endif
    */

#define NUM_INPUTS 61

    int ncolumns = 41;

#ifdef OUTPUT_RVMAX
    ncolumns ++;
#endif
#ifdef OUTPUT_NFW_CHI2
    ncolumns ++;
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
    ncolumns += 6;
#endif
#ifdef OUTPUT_INERTIA_TENSOR
    ncolumns += 12;
#endif
    fprintf( stderr, "ncolumns= %d\n", ncolumns);

    enum short_parsetype stypes[NUM_INPUTS] = {
        D64, D64, F,   F, F, //  #id desc_id mvir vmax vrms
        F,   F,   D64, F,    //  Rvir Rs Np x
        F,   F,   F,   F, F, // y z vx vy vz
        F,   F,   F,   F, F, // JX JY JZ Spin rs_klypin
        F,   F,   F,   F, F, // M_all M1 M2 M3 M4
        F,   F,   F,   F, F, // Xoff Voff spin_bullock b_to_a c_to_a
        F,   F,   F,   F, F, // A[x] A[y] A[z] b_to_a500 c_to_a500
        F,   F,   F,   F, F, //  A2[x] A2[y] A2[z] T/|U| M_PE_Behroozi
        F,   F               // M_PE_Diemer Halfmass_Radius
#ifdef OUTPUT_RVMAX
	, F  //Rvmax
#endif
#ifdef OUTPUT_NFW_CHI2
	, F  //Rvmax
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
	, F, F, F, F, F, F
#endif
#ifdef OUTPUT_INERTIA_TENSOR
	, F, F, F, F, F, F // Inertial tensor
	, F, F, F, F, F, F // Inertial tensor(500c)
#endif
    };

    enum parsetype types[NUM_INPUTS];
    void          *data[NUM_INPUTS] = {&(h.id),
                                       &(h.descid),
                                       &(h.mvir),
                                       &(h.vmax),
                                       &(h.vrms),
                                       &(h.rvir),
                                       &(h.rs),
                                       &(h.np),
                                       &(h.pos[0]),
                                       &(h.pos[1]),
                                       &(h.pos[2]),
                                       &(h.vel[0]),
                                       &(h.vel[1]),
                                       &(h.vel[2]),
                                       &(h.J[0]),
                                       &(h.J[1]),
                                       &(h.J[2]),
                                       &(h.spin),
                                       &(h.klypin_rs),
                                       &(h.m_all),
                                       &(h.alt_m[0]),
                                       &(h.alt_m[1]),
                                       &(h.alt_m[2]),
                                       &(h.alt_m[3]),
                                       &(h.Xoff),
                                       &(h.Voff),
                                       &(h.bullock_spin),
                                       &(h.b_to_a),
                                       &(h.c_to_a),
                                       &(h.A[0]),
                                       &(h.A[1]),
                                       &(h.A[2]),
                                       &(h.b_to_a2),
                                       &(h.c_to_a2),
                                       &(h.A2[0]),
                                       &(h.A2[1]),
                                       &(h.A2[2]),
                                       &(h.kin_to_pot),
                                       &(h.m_pe_b),
                                       &(h.m_pe_d),
                                       &(h.halfmass_radius)
#ifdef OUTPUT_RVMAX 
	, &(h.rvmax)
#endif
#ifdef OUTPUT_NFW_CHI2
	, &(h.chi2)
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
       , &(h.A_I[0]), &(h.A_I[1]), &(h.A_I[2]), &(h.A2_I[0]), &(h.A2_I[1]), &(h.A2_I[2])
#endif
#ifdef OUTPUT_INERTIA_TENSOR
	, &(h.inertia_tensor[0]), &(h.inertia_tensor[1]), &(h.inertia_tensor[2])
	, &(h.inertia_tensor[3]), &(h.inertia_tensor[4]), &(h.inertia_tensor[5])
	, &(h.inertia_tensor2[0]), &(h.inertia_tensor2[1]), &(h.inertia_tensor2[2])
	, &(h.inertia_tensor2[3]), &(h.inertia_tensor2[4]), &(h.inertia_tensor2[5])
#endif
	};

    //for (n = 0; n < NUM_INPUTS; n++)
    for (n = 0; n < ncolumns; n++)
        types[n] = stypes[n];
    input = check_fopen(filename, "r");
    while (fgets(buffer, 1024, input)) {
        if (buffer[0] == '#') {
            if (c == 0) {
                c                          = 1;
                buffer[strlen(buffer) - 1] = 0;
                printf("%s PID\n", buffer);
            } else {
                printf("%s", buffer);
            }
        }
        n = stringparse(buffer, data, (enum parsetype *)types, NUM_INPUTS);
        //if (n < NUM_INPUTS)
	if (n < ncolumns)
            continue;
        if (bounds) {
            float rvir = h.rvir / 1.0e3; // in Mpc/h
            for (i = 0; i < 3; i++) {
                if (((h.pos[i] + rvir < bounds[i]) &&
                     (h.pos[i] - rvir + BOX_SIZE > bounds[i + 3])) ||
                    ((h.pos[i] - rvir > bounds[i + 3]) &&
                     (h.pos[i] + rvir - BOX_SIZE < bounds[i])))
                    break;
            }
            if (i < 3)
                continue;
        }
        if (!(all_halos.num_halos % 3000))
            all_halos.halos = check_realloc(all_halos.halos,
                                            sizeof(struct halo) *
                                                (all_halos.num_halos + 3000),
                                            "Allocating Halos.");

        all_halos.halos[all_halos.num_halos] = h;
        all_halos.num_halos++;
    }
    fclose(input);

    all_halos.halos = check_realloc(all_halos.halos,
                                    sizeof(struct halo) * all_halos.num_halos,
                                    "Allocating Halos.");

    for (n = 0; n < all_halos.num_halos; n++) {
        all_halos.halos[n].vmax_r = all_halos.halos[n].rvir;
    }

    find_parents(all_halos.num_halos);

    for (n = 0; n < all_halos.num_halos; n++) {
        struct halo *th = all_halos.halos + n;
        if (bounds) {
            for (i = 0; i < 3; i++)
                if ((th->pos[i] < bounds[i]) || (th->pos[i] >= bounds[i + 3]))
                    break;
            if (i < 3)
                continue;
        }
        printf("%" PRId64 " %" PRId64 " %.3e %.2f %.2f %.3f %.3f %" PRId64
               " %.5f %.5f %.5f %.2f %.2f %.2f %.3e %.3e %.3e %.5f %.5f %.4e "
               "%.4e %.4e %.4e %.4e %.5f %.2f %.5f %.5f %.5f %.5f %.5f %.5f "
               "%.5f %.5f %.5f %.5f %.5f %.4f %.3e %.3e %.3f"
#ifdef OUTPUT_RVMAX
	       " %.3f"
#endif
#ifdef OUTPUT_NFW_CHI2
	       " %.4e"
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
	       " %.5f %.5f %.5f %.5f %.5f %.5f"
#endif
#ifdef OUTPUT_INERTIA_TENSOR
	       " %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e"
#endif
               " %" PRId64 "\n",
               th->id, th->descid, th->mvir, th->vmax, th->vrms, th->rvir,
               th->rs, th->np, th->pos[0], th->pos[1], th->pos[2], th->vel[0],
               th->vel[1], th->vel[2], th->J[0], th->J[1], th->J[2], th->spin,
               th->klypin_rs, th->m_all, th->alt_m[0], th->alt_m[1],
               th->alt_m[2], th->alt_m[3], th->Xoff, th->Voff, th->bullock_spin,
               th->b_to_a, th->c_to_a, th->A[0], th->A[1], th->A[2],
               th->b_to_a2, th->c_to_a2, th->A2[0], th->A2[1], th->A2[2],
               th->kin_to_pot, th->m_pe_b, th->m_pe_d, th->halfmass_radius
#ifdef OUTPUT_RVMAX
	       ,th->rvmax
#endif
#ifdef OUTPUT_NFW_CHI2
	       ,th->chi2
#endif
#ifdef OUTPUT_INTERMEDIATE_AXIS
	       ,
	       th->A_I[0], th->A_I[1], th->A_I[2], 
               th->A2_I[0], th->A2_I[1], th->A2_I[2]
#endif
#ifdef OUTPUT_INERTIA_TENSOR
               ,
               th->inertia_tensor[0], th->inertia_tensor[1], th->inertia_tensor[2], 
               th->inertia_tensor[3], th->inertia_tensor[4], th->inertia_tensor[5], 
               th->inertia_tensor2[0], th->inertia_tensor2[1], th->inertia_tensor2[2], 
               th->inertia_tensor2[3], th->inertia_tensor2[4], th->inertia_tensor2[5]
#endif
	      ,th->pid);
    }
}

int main(int argc, char **argv) {
    int64_t i;
    float   bounds[6] = {0};
    if (argc < 2 || ((argc > 3 && argc < 9))) {
        printf("Usage: %s out_XYZ.list box_size [x_min y_min z_min x_max y_max "
               "z_max]\n",
               argv[0]);
        printf("Note: all dimensions must be present if selecting halos within "
               "a fraction of the box volume.\n");
        exit(EXIT_FAILURE);
    }
    if (argc > 2)
        BOX_SIZE = atof(argv[2]);
    if (argc >= 9) {
        for (i = 0; i < 6; i++)
            bounds[i] = atof(argv[3 + i]);
        read_hlist(argv[1], bounds);
    } else
        read_hlist(argv[1], NULL);
    return 0;
}
