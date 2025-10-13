/**
 * @file    outgas.c
 * @brief   Add outgassing forces, based on Kramer & Laüter 2019, see https://www.aanda.org/articles/aa/full_html/2019/10/aa35229-19/aa35229-19.html
 * @author  Rainer Marquardt-Demen
 * 
 * @section     LICENSE
 *
 * This file is part of reboundx.
 *
 * reboundx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * reboundx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 * The section after the dollar signs gets built into the documentation by a script.  All lines must start with space * space like below.
 * Tables always must be preceded and followed by a blank line.  See http://docutils.sourceforge.net/docs/user/rst/quickstart.html for a primer on rst.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "reboundx.h"

static void rebx_calculate_outgas_forces(struct rebx_extras* const rebx, struct reb_simulation* const sim, double alpha, double n, double k, double m, double r0, double a01, double a02, double a03, const int source_index, struct reb_particle* const particles, const int N){
    const struct reb_particle source = particles[source_index];
    
    for (int i=0; i<N; i++){
        if(i == source_index) continue;
        
        struct reb_particle* p = &particles[i];
        
        double A1[3] = {source.x - p->x, source.y - p->y, source.z - p->z};
        double R = sqrt(A1[0]*A1[0] + A1[1]*A1[1] + A1[2]*A1[2]);
        
        // Normalize A1
        A1[0] /= R; A1[1] /= R; A1[2] /= R;
        
        double A2[3] = {p->vx, p->vy, p->vz};
        double norm_A2 = sqrt(A2[0]*A2[0] + A2[1]*A2[1] + A2[2]*A2[2]);
        
        // Normalize A2
        A2[0] /= norm_A2; A2[1] /= norm_A2; A2[2] /= norm_A2;
        
        double A3[3] = {
            A1[1]*A2[2] - A1[2]*A2[1],
            A1[2]*A2[0] - A1[0]*A2[2],
            A1[0]*A2[1] - A1[1]*A2[0]
        };
        A1[0] *= a01; A1[1] *= a01; A1[2] *= a01;
        A2[0] *= a02; A2[1] *= a02; A2[2] *= a02;
        A3[0] *= a03; A3[1] *= a03; A3[2] *= a03;
        
        // Scaling factor
        double scale = alpha * pow(R/r0, -m) * pow(1 + pow(R/r0, n), -k);
        
        // Apply scaling
        A1[0] *= scale; A1[1] *= scale; A1[2] *= scale;
        A2[0] *= scale; A2[1] *= scale; A2[2] *= scale;
        A3[0] *= scale; A3[1] *= scale; A3[2] *= scale;
        
        // Update particle acceleration
        p->ax += A1[0] + A2[0] + A3[0];
        p->ay += A1[1] + A2[1] + A3[1];
        p->az += A1[2] + A2[2] + A3[2];
    }
}

void rebx_outgas_forces(struct reb_simulation* const sim, struct rebx_force* const outgas_forces, struct reb_particle* const particles, const int N){
    struct rebx_extras* const rebx = sim->extras;
    
    double* alpha = rebx_get_param(rebx, outgas_forces->ap, "alpha");
    double* n = rebx_get_param(rebx, outgas_forces->ap, "n");
    double* k = rebx_get_param(rebx, outgas_forces->ap, "k");
    double* m = rebx_get_param(rebx, outgas_forces->ap, "m");
    double* r0 = rebx_get_param(rebx, outgas_forces->ap, "r0");
    double* a01 = rebx_get_param(rebx, outgas_forces->ap, "a01");
    double* a02 = rebx_get_param(rebx, outgas_forces->ap, "a02");
    double* a03 = rebx_get_param(rebx, outgas_forces->ap, "a03");
    
    if (!alpha || !n || !k || !m || !r0 | !a01 | !a02 | !a03){
        reb_simulation_error(sim, "Missing parameters for outgas forces.\n");
        return;
    }
    
    int source_found = 0;
    for (int i = 0; i < N; i++){
        if (rebx_get_param(rebx, particles[i].ap, "outgas_source") != NULL){
            source_found = 1;
            rebx_calculate_outgas_forces(rebx, sim, *alpha, *n, *k, *m, *r0, *a01, *a02, *a03, i, particles, N);
        }
    }
    if (!source_found){
        rebx_calculate_outgas_forces(rebx, sim, *alpha, *n, *k, *m, *r0, *a01, *a02, *a03, 0, particles, N); // default to index 0
    }
}
