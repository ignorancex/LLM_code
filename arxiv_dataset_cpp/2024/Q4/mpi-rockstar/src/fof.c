#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "check_syscalls.h"
#include "fof.h"
#include "particle.h"
#include "config_vars.h"

void init_fofinfo(struct FOFInfo *fofinfo) {
    fofinfo->fofs     = NULL;
    fofinfo->num_fofs = fofinfo->num_alloced_fofs = 0;
    fofinfo->num_boundary_fofs                    = 0;

    fofinfo->smallfofs     = NULL;
    fofinfo->num_smallfofs = fofinfo->num_alloced_smallfofs = 0;

    fofinfo->root_p             = NULL;
    fofinfo->particle_smallfofs = NULL;
    fofinfo->num_particles = fofinfo->num_alloced_particles = 0;
}

void free_fofinfo(struct FOFInfo *fofinfo) {
    check_realloc_s(fofinfo->fofs, 0, 0);
    check_realloc_s(fofinfo->smallfofs, 0, 0);
    check_realloc_s(fofinfo->particle_smallfofs, 0, 0);
    init_fofinfo(fofinfo);
}

void init_particle_smallfofs(int64_t num_p, struct particle *particles,
                             struct FOFInfo *fofinfo) {
    int64_t i;
    if (num_p > fofinfo->num_alloced_particles) {
        fofinfo->particle_smallfofs =
            check_realloc(fofinfo->particle_smallfofs, sizeof(int64_t) * num_p,
                          "Allocating particle smallfof links.");
        fofinfo->num_alloced_particles = num_p;
    }
    for (i = 0; i < num_p; i++)
        fofinfo->particle_smallfofs[i] = -1;
    fofinfo->root_p            = particles;
    fofinfo->num_particles     = num_p;
    fofinfo->num_boundary_fofs = fofinfo->num_fofs = fofinfo->num_smallfofs = 0;
}

int64_t add_new_smallfof(struct FOFInfo *fofinfo) {
    if (fofinfo->num_smallfofs >= fofinfo->num_alloced_smallfofs) {
        fofinfo->smallfofs = (struct smallfof *)check_realloc(
            fofinfo->smallfofs,
            sizeof(struct smallfof) * (fofinfo->num_smallfofs + 1000),
            "Allocating SmallFOFs.\n");
        fofinfo->num_alloced_smallfofs += 1000;
    }
    fofinfo->smallfofs[fofinfo->num_smallfofs].root = fofinfo->num_smallfofs;
    fofinfo->num_smallfofs++;
    return (fofinfo->num_smallfofs - 1);
}

void _collapse_smallfof(struct smallfof *f, const struct FOFInfo *fofinfo) {
    struct smallfof *r = fofinfo->smallfofs + f->root, *n;
    if (fofinfo->smallfofs[f->root].root == f->root)
        return;
    while (r != (fofinfo->smallfofs + r->root))
        r = fofinfo->smallfofs + r->root;
    while (f != r) {
        n       = fofinfo->smallfofs + f->root;
        f->root = r->root;
        f       = n;
    }
}

void merge_smallfofs(struct smallfof *f1, struct smallfof *f2,
                     const struct FOFInfo *fofinfo) {
    struct smallfof *r = NULL, *d;
    int64_t          f1root;
    if (f2->root == f1->root)
        return;
    _collapse_smallfof(f1, fofinfo);
    if (f2 == fofinfo->smallfofs + f2->root) {
        f2->root = f1->root;
        return;
    }
    f1root = f1->root;
    d      = fofinfo->smallfofs + f1root;
    while (r != d) {
        r        = fofinfo->smallfofs + f2->root;
        f2->root = f1root;
        f2       = r;
    }
}

int64_t tag_boundary_particle(struct particle *p, struct FOFInfo *fofinfo) {
#define SMALLFOF_OF(a) fofinfo->particle_smallfofs[(a)-fofinfo->root_p]
    int64_t f = SMALLFOF_OF(p);
    if (f < 0) {
        SMALLFOF_OF(p) = add_new_smallfof(fofinfo);
        fofinfo->num_boundary_fofs++;
        return fofinfo->num_boundary_fofs - 1;
    }
    _collapse_smallfof(fofinfo->smallfofs + f, fofinfo);
    if (fofinfo->smallfofs[f].root >=
        fofinfo->num_smallfofs - fofinfo->num_boundary_fofs)
        return (fofinfo->smallfofs[f].root -
                (fofinfo->num_smallfofs - fofinfo->num_boundary_fofs));
    int64_t new_fof = add_new_smallfof(fofinfo);
    fofinfo->smallfofs[fofinfo->smallfofs[f].root].root = new_fof;
    fofinfo->num_boundary_fofs++;
    return (fofinfo->num_boundary_fofs - 1);
#undef SMALLFOF_OF
}

void link_particle_to_fof(struct particle *p, int64_t n,
                          struct particle **links, struct FOFInfo *fofinfo) {
    int64_t i;
#define SMALLFOF_OF(a) fofinfo->particle_smallfofs[(a)-fofinfo->root_p]
    int64_t f = SMALLFOF_OF(p);
    if (n < 2)
        return;
    if (f < 0) {
        for (i = 0; i < n; i++) {
            if (SMALLFOF_OF(links[i]) == -1)
                continue;
            else {
                f = SMALLFOF_OF(links[i]);
                break;
            }
        }
        if (f < 0) {
            f = add_new_smallfof(fofinfo);
        }
    }
    for (i = 0; i < n; i++) {
        if (SMALLFOF_OF(links[i]) == -1)
            SMALLFOF_OF(links[i]) = f;
        else
            merge_smallfofs(fofinfo->smallfofs + SMALLFOF_OF(links[i]),
                            fofinfo->smallfofs + f, fofinfo);
    }
#undef SMALLFOF_OF
}

void link_fof_to_fof(struct particle *p, int64_t n, struct particle **links,
                     const struct FOFInfo *fofinfo) {
    int64_t i;
#define SMALLFOF_OF(a) fofinfo->particle_smallfofs[(a)-fofinfo->root_p]
    int64_t f = SMALLFOF_OF(p);
    if ((n < 2) || (f < 0))
        return;
    for (i = 0; i < n; i++) {
        if (SMALLFOF_OF(links[i]) == -1)
            continue;
        if (SMALLFOF_OF(links[i]) == f)
            continue;
        merge_smallfofs(fofinfo->smallfofs + SMALLFOF_OF(links[i]),
                        fofinfo->smallfofs + f, fofinfo);
    }
#undef SMALLFOF_OF
}

void collapse_smallfofs(const struct FOFInfo *fofinfo) {
    int64_t i;
    for (i = 0; i < fofinfo->num_smallfofs; i++)
        _collapse_smallfof(fofinfo->smallfofs + i, fofinfo);
    for (i = 0; i < fofinfo->num_particles; i++) {
        if (fofinfo->particle_smallfofs[i] < 0)
            continue;
        fofinfo->particle_smallfofs[i] =
            fofinfo->smallfofs[fofinfo->particle_smallfofs[i]].root;
    }
}

int64_t add_new_fof(struct FOFInfo *fofinfo) {
    if (fofinfo->num_fofs >= fofinfo->num_alloced_fofs) {
        fofinfo->fofs = (struct fof *)check_realloc(
            fofinfo->fofs, sizeof(struct fof) * (fofinfo->num_fofs + 1000),
            "Allocating FOFs.\n");
        memset(fofinfo->fofs + fofinfo->num_fofs, 0, sizeof(struct fof) * 1000);
        fofinfo->num_alloced_fofs = fofinfo->num_fofs + 1000;
    }
    fofinfo->num_fofs++;
    return (fofinfo->num_fofs - 1);
}

void partition_sort_particles(int64_t min, int64_t max,
                              struct particle *particles,
                              int64_t         *assignments) {
    int64_t         minpivot, maxpivot, pivot, i, si, tmp;
    struct particle tmp_p;
    if (max - min < 2)
        return;

    maxpivot = minpivot = assignments[min];
    for (i = min + 1; i < max; i++) {
        if (assignments[i] > maxpivot)
            maxpivot = assignments[i];
        if (assignments[i] < minpivot)
            minpivot = assignments[i];
    }
    if (minpivot == maxpivot)
        return;
    pivot = minpivot + (maxpivot - minpivot) / 2;
    si    = max - 1;
#define SWAP(a, b)                                                             \
    {                                                                          \
        tmp_p          = particles[a];                                         \
        particles[a]   = particles[b];                                         \
        particles[b]   = tmp_p;                                                \
        tmp            = assignments[a];                                       \
        assignments[a] = assignments[b];                                       \
        assignments[b] = tmp;                                                  \
    }

    for (i = min; i < si; i++)
        if (assignments[i] > pivot) {
            SWAP(i, si);
            si--;
            i--;
        }
    if (i == si && assignments[si] <= pivot)
        si++;
#undef SWAP
    partition_sort_particles(min, si, particles, assignments);
    partition_sort_particles(si, max, particles, assignments);
}

void build_fullfofs(struct FOFInfo *fofinfo) {
    int64_t i, sf = -1, last_sf = -1, f = -1;
    collapse_smallfofs(fofinfo);
    partition_sort_particles(0, fofinfo->num_particles, fofinfo->root_p,
                             fofinfo->particle_smallfofs);
    for (i = 0; i < fofinfo->num_particles; i++) {
        if (fofinfo->particle_smallfofs[i] < 0)
            continue;
        sf = fofinfo->particle_smallfofs[i];
        if (sf != last_sf) {
            if (f > -1) {
                fofinfo->fofs[f].num_p =
                    (fofinfo->root_p + i) - fofinfo->fofs[f].particles;
            }
            if ((f < 0) || (fofinfo->fofs[f].num_p >= MIN_HALO_PARTICLES) ||
                last_sf >= fofinfo->num_smallfofs - fofinfo->num_boundary_fofs)
                f = add_new_fof(fofinfo);
            fofinfo->fofs[f].particles = fofinfo->root_p + i;
            last_sf                    = sf;
        }
    }
    if (f > -1) {
        fofinfo->fofs[f].num_p =
            (fofinfo->root_p + i) - fofinfo->fofs[f].particles;
        if (fofinfo->fofs[f].num_p < MIN_HALO_PARTICLES &&
            (sf < fofinfo->num_smallfofs - fofinfo->num_boundary_fofs))
            fofinfo->num_fofs--;
    }
    fofinfo->num_smallfofs = 0;
}

struct fof *return_fullfofs(int64_t *num_f, int64_t *num_bf,
                            struct FOFInfo *fofinfo) {
    struct fof *f_all_fofs;
    fofinfo->num_smallfofs = fofinfo->num_alloced_smallfofs = 0;
    fofinfo->smallfofs =
        check_realloc(fofinfo->smallfofs, 0, "Freeing SmallFOFs.");
    fofinfo->particle_smallfofs = check_realloc(fofinfo->particle_smallfofs, 0,
                                                "Freeing particle smallfofs.");
    fofinfo->num_alloced_particles = 0;
    f_all_fofs                     = fofinfo->fofs;
    *num_f                         = fofinfo->num_fofs;
    *num_bf                        = fofinfo->num_boundary_fofs;
    fofinfo->fofs                  = NULL;
    fofinfo->num_boundary_fofs = fofinfo->num_alloced_fofs = fofinfo->num_fofs =
        0;
    return f_all_fofs;
}

void copy_fullfofs(struct fof **base, int64_t *num_f, int64_t *num_alloced_f,
                   struct FOFInfo *fofinfo) {
    if ((*num_f) + fofinfo->num_fofs > (*num_alloced_f)) {
        *base = check_realloc(
            *base, sizeof(struct fof) * ((*num_f) + fofinfo->num_fofs + 1000),
            "Allocating copy space for FOFs.");
        *num_alloced_f = (*num_f) + fofinfo->num_fofs + 1000;
    }
    memcpy((*base) + (*num_f), fofinfo->fofs,
           sizeof(struct fof) * fofinfo->num_fofs);
    *num_f = (*num_f) + fofinfo->num_fofs;

    // Make sure to delete last fof, if necessary.
    if (fofinfo->num_fofs < fofinfo->num_alloced_fofs)
        fofinfo->num_fofs++;
    memset(fofinfo->fofs, 0, sizeof(struct fof) * fofinfo->num_fofs);
    fofinfo->num_boundary_fofs = fofinfo->num_fofs = 0;
}
