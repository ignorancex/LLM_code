/*
 * Copyright: (C) 2019 VisLab, Institute for Systems and Robotics,
 *                     Istituto Superior Técnico, Lisbon, Portugal
 * Author: Giovanni Saponaro <gsaponaro@isr.tecnico.ulisboa.pt>
 * CopyPolicy: Released under the terms of the GNU GPL v3.0
 */

#include "Helpers.h"
#include "HumanHandProcessorModule.h"

using namespace std;
using namespace yarp::os;

bool HumanHandProcessorModule::configure(ResourceFinder &rf)
{
    string moduleName;
    moduleName = rf.check( "name",Value("humanHandProcessor") ).asString();
    setName(moduleName.c_str());

    inSkelPort.open(("/"+moduleName+"/skel:i").c_str());
    outHandPort.open(("/"+moduleName+"/hand:o").c_str());

    return true;
}

bool HumanHandProcessorModule::interruptModule()
{
    inSkelPort.interrupt();
    outHandPort.interrupt();
    return true;
}


bool HumanHandProcessorModule::close()
{
    inSkelPort.close();
    outHandPort.close();
    return true;
}

bool HumanHandProcessorModule::updateModule()
{
    const int expSkelSize = 91;
    Bottle *skel = inSkelPort.read(true);
    if (skel != NULL)
    {
        if (skel->size() != expSkelSize)
        {
            yWarning("skeleton bottle size mismatch");
            return true;
        }

        Bottle &b = outHandPort.prepare();
        b.clear();

        // yarp OpenNI2 device driver exports each skeleton joint as:
        // [POS] (position) positionConfidence [ORI] (orientation) orientationConfidence
        const int jointSize = 6;

        // the 15 available joints are:
        // HEAD NECK LEFT_SHOULDER RIGHT_SHOULDER LEFT_ELBOW RIGHT_ELBOW
        // LEFT_HAND RIGHT_HAND TORSO LEFT_HIP RIGHT_HIP LEFT_KNEE RIGHT_KNEE
        // LEFT_FOOT RIGHT_FOOT
        const int rightHandJointIdx = 7;
        const int torsoJointIdx = 8;

        // right hand coordinates, not normalized
        const double rhX = skel->get(1+rightHandJointIdx*jointSize+1).asList()->get(0).asDouble();
        const double rhY = skel->get(1+rightHandJointIdx*jointSize+1).asList()->get(1).asDouble();
        const double rhZ = skel->get(1+rightHandJointIdx*jointSize+1).asList()->get(2).asDouble();

        // torso coordinates, not normalized
        const double tX = skel->get(1+torsoJointIdx*jointSize+1).asList()->get(0).asDouble();
        const double tY = skel->get(1+torsoJointIdx*jointSize+1).asList()->get(1).asDouble();
        const double tZ = skel->get(1+torsoJointIdx*jointSize+1).asList()->get(2).asDouble();

        // frame transformation from sensor to torso
        double ox = rhX - tX;
        double oy = rhY - tY;
        double oz = rhZ - tZ;

        //yDebug("rh (%.2f %.2f %.2f) - t (%.2f %.2f %.2f) = (%.2f %.2f %.2f)", rhX, rhY, rhZ, tX, tY, tZ, ox, oy, oz);

        // x - horizontal, positive right (person's point of view)
        // y - vertical, positive upwards
        // z - orthogonal to human torso (positive towards person's back)

        // horizontal normalization factor, to account for variability in people
        // height
        const double horiz_nz = compute_horiz_nz();

        // vertical normalization factor, to account for variability in people
        // size
        const double vert_nz = compute_vert_nz();

        // divide x,z columns by horizontal normalization factor
        ox = ox / horiz_nz;
        oz = oz / horiz_nz;

        // divide y column by vertical normalization factor
        oy = oy / vert_nz;

        // send to yarp port output
        b.addDouble(ox);
        b.addDouble(oy);
        b.addDouble(oz);

        outHandPort.write();
    }

    return true;
}

double HumanHandProcessorModule::getPeriod()
{
    return 0.0; // sync with incoming data
}
