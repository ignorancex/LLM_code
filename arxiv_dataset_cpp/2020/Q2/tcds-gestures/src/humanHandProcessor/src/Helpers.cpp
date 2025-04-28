/*
 * Copyright: (C) 2019 VisLab, Institute for Systems and Robotics,
 *                     Istituto Superior Técnico, Lisbon, Portugal
 * Author: Giovanni Saponaro <gsaponaro@isr.tecnico.ulisboa.pt>
 * CopyPolicy: Released under the terms of the GNU GPL v3.0
 */

#include "Helpers.h"

/******************************************************************************/
double compute_horiz_nz()
{
    // in the MATLAB version, we used the mean distance between shoulders in the
    // (0.45-0.55)*nframes part of the data sequence.

    // empirical value from 2012 training data
    return 0.33;
}

/******************************************************************************/
double compute_vert_nz()
{
    // in the MATLAB version, we used the mean distance between neck and torso
    // in the (0.45-0.55)*nframes part of the data sequence.

    // empirical value from 2012 training data
    return 0.23;
}
