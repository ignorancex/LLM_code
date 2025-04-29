#include "welford.h"

// Function to create Welford parameters object for an array of points 
void compute_welford_params(Point* data, int n, WelfordParams* params) {
    params->n = 0;
    params->mean_x = 0.0;
    params->mean_y = 0.0;
    params->s_x = 0.0;
    params->s_y = 0.0;
    params->s_xy = 0.0;

    for (int i = 0; i < n; ++i) {
        double delta_x, delta_y;
        double x = data[i].x;
        double y = data[i].y;

        params->n += 1;
        delta_x = x - params->mean_x;
        delta_y = y - params->mean_y;

        params->mean_x += delta_x / params->n;
        params->mean_y += delta_y / params->n;

        params->s_x += delta_x * (x - params->mean_x);
        params->s_y += delta_y * (y - params->mean_y);
        params->s_xy += delta_x * (y - params->mean_y);
    }
}

// Function to update Welford parameters with a new point, outputting to another WelfordParams object
void update_welford_params(WelfordParams* params, Point p, WelfordParams* updated_params) {
    updated_params->n = params->n + 1;

    double delta_x_1 = p.x - params->mean_x;
    double delta_y_1 = p.y - params->mean_y;
    double new_mean_x = params->mean_x + delta_x_1 / updated_params->n;
    double new_mean_y = params->mean_y + delta_y_1 / updated_params->n;
    double delta_x_2 = p.x - new_mean_x;
    double delta_y_2 = p.y - new_mean_y;

    updated_params->mean_x = new_mean_x;
    updated_params->mean_y = new_mean_y;
    updated_params->s_x = params->s_x + delta_x_1 * delta_x_2;
    updated_params->s_y = params->s_y + delta_y_1 * delta_y_2;
    updated_params->s_xy = params->s_xy + delta_x_1 * delta_y_2;
}
