#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "primary_sensitivity_impl.h"
#include "students_t.h"


// Main function to compute primary sensitivities
double compute_primary_sensitivities(Point* data, int n, Point* bounds, double* delta_r, double* delta_p) {
    WelfordParams params;
    compute_welford_params(data, n, &params);

    double beta_x = params.s_xy / params.s_x;
    double beta_y = params.s_xy / params.s_y;
    double alpha_x = params.mean_y - beta_x * params.mean_x;
    double alpha_y = params.mean_x - beta_y * params.mean_y;

    // Corners of the boundary
    Point corners[4] = {
        {bounds[0].x, bounds[0].y},
        {bounds[0].x, bounds[1].y},
        {bounds[1].x, bounds[0].y},
        {bounds[1].x, bounds[1].y}
    };

    // Intersections with boundary lines
    Point intersections[N_INTER];
    int num_intersections = 0;

    double x_at_ly = (bounds[0].y - alpha_y) / beta_y;
    if (bounds[0].x <= x_at_ly && x_at_ly <= bounds[1].x) {
        intersections[num_intersections++] = (Point){x_at_ly, bounds[0].y};
    }

    double x_at_uy = (bounds[1].y - alpha_y) / beta_y;
    if (bounds[0].x <= x_at_uy && x_at_uy <= bounds[1].x) {
        intersections[num_intersections++] = (Point){x_at_uy, bounds[1].y};
    }

    // double y_at_lx = beta_x * bounds[0].x + alpha_x;
    double y_at_lx = (bounds[0].x - alpha_x) / beta_x;
    if (bounds[0].y <= y_at_lx && y_at_lx <= bounds[1].y) {
        intersections[num_intersections++] = (Point){bounds[0].x, y_at_lx};
    }

    // double y_at_ux = beta_x * bounds[1].x + alpha_x;
    double y_at_ux = (bounds[1].x - alpha_x) / beta_x;
    if (bounds[0].y <= y_at_ux && y_at_ux <= bounds[1].y) {
        intersections[num_intersections++] = (Point){bounds[1].x, y_at_ux};
    }

    double r_n = params.s_xy / (sqrt(params.s_x) * sqrt(params.s_y));
    double p_n = students_t(r_n, n);

    double max_delta_r = 0.0;
    double max_delta_p = 0.0;

    double max_r_n_plus_1 = -INFINITY;
    double min_r_n_plus_1 = INFINITY;

    // Evaluate corner points
    for (int i = 0; i < 4; ++i) {
        Point p = corners[i];
        WelfordParams updated_params;
        update_welford_params(&params, p, &updated_params);
        double r_n_plus_1 = updated_params.s_xy / (sqrt(updated_params.s_x) * sqrt(updated_params.s_y));
        double delta_r = fabs(r_n - r_n_plus_1);
        if (delta_r > max_delta_r) {
            max_delta_r = delta_r;
        }
        if (r_n_plus_1 > max_r_n_plus_1) {
            max_r_n_plus_1 = r_n_plus_1;
        }
        if (r_n_plus_1 < min_r_n_plus_1) {
            min_r_n_plus_1 = r_n_plus_1;
        }
        double p_n_plus_1 = students_t(r_n_plus_1, n + 1);
        double delta_p = fabs(p_n - p_n_plus_1);
        if (delta_p > max_delta_p) {
            max_delta_p = delta_p;
        }
    }

    // Evaluate intersection points
    for (int i = 0; i < num_intersections; ++i) {
        Point p = intersections[i];
        double delta_x_1 = p.x - params.mean_x;
        double delta_y_1 = p.y - params.mean_y;
        double new_mean_x = params.mean_x + delta_x_1 / (n+1);
        double new_mean_y = params.mean_y + delta_y_1 / (n+1);
        double delta_x_2 = p.x - new_mean_x;
        double delta_y_2 = p.y - new_mean_y;
        double new_s_x = params.s_x + delta_x_1 * delta_x_2;
        double new_s_y = params.s_y + delta_y_1 * delta_y_2;
        double new_s_xy = params.s_xy + delta_x_1 * delta_y_2;
        double r_n_plus_1 = new_s_xy / (sqrt(new_s_x) * sqrt(new_s_y));
        double delta_r = fabs(r_n - r_n_plus_1);
        if (delta_r > max_delta_r) {
            max_delta_r = delta_r;
        }
        if (r_n_plus_1 > max_r_n_plus_1) {
            max_r_n_plus_1 = r_n_plus_1;
        }
        if (r_n_plus_1 < min_r_n_plus_1) {
            min_r_n_plus_1 = r_n_plus_1;
        }
        double p_n_plus_1 = students_t(r_n_plus_1, n + 1);
        double delta_p = fabs(p_n - p_n_plus_1);
        if (delta_p > max_delta_p) {
            max_delta_p = delta_p;
        }
    }

    // Check if max_r_n_plus_1 and min_r_n_plus_1 have opposite signs
    if (max_r_n_plus_1 >= 0 && min_r_n_plus_1 <= 0) {
        if (1 - p_n > max_delta_p) {
            max_delta_p = 1 - p_n;
        }
    }

    *delta_r = max_delta_r;
    *delta_p = max_delta_p;

    return 0;
}
