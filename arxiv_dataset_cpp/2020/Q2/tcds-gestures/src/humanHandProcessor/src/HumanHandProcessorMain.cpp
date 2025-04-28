/*
 * Copyright: (C) 2019 VisLab, Institute for Systems and Robotics,
 *                     Istituto Superior Técnico, Lisbon, Portugal
 * Author: Giovanni Saponaro <gsaponaro@isr.tecnico.ulisboa.pt>
 * CopyPolicy: Released under the terms of the GNU GPL v3.0
 */

#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>

#include "HumanHandProcessorModule.h"

using namespace yarp::os;

int main(int argc, char *argv[])
{
    Network yarp;

    ResourceFinder rf;
    rf.setVerbose(false);
    rf.setDefaultContext("humanHandProcessor");
    rf.setDefaultConfigFile("humanHandProcessor.ini");
    rf.setDefault("name", "humanHandProcessor");
    rf.configure(argc, argv);

    if (! yarp::os::Network::checkNetwork())
    {
        yError("yarpserver not available!");
        return 1; // EXIT_FAILURE
    }

    HumanHandProcessorModule mod;
    return mod.runModule(rf);
}
