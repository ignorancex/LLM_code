#!/usr/bin/env python
from __future__ import division

import time
import sys
import os
import numpy as np

import utils2

import rospy
import rospkg

from UnidirMRSPlanner import UnidirMRSPlanner
from UnidirDFSDPPlanner import UnidirDFSDPPlanner
from UnidirCIRSPlanner import UnidirCIRSPlanner

############################### description #########################################
### This class defines a InstanceTester class which
### solves an rearrangement problem/example with 
### the number of the object specified
### It
### (1) asks the execution scene to generate an instance
### (2) asks the pose estimator to get the object poses
### (3) reproduces the instance in task planner
### (4) solves it with all methods
### (4) compares the solutions of all methods within the same instance
#####################################################################################

class InstanceTester(object):

    def __init__(self, args):
        ### set the rospkg path
        rospack = rospkg.RosPack()
        self.rosPackagePath = rospack.get_path("uniform_object_rearrangement")
        self.num_objects = int(args[1])
        self.instance_id = int(args[2])
        self.isNewInstance = True if args[3] == 'g' else False
        self.time_allowed = int(args[4])
        self.instanceFolder = os.path.join(
            self.rosPackagePath, "examples", str(self.num_objects), str(self.instance_id))

    def rosInit(self):
        ### This function specifies the role of a node instance for this class ###
        ### and initializes a ros node
        rospy.init_node("monotone_test", anonymous=True)


def main(args):
    instance_tester = InstanceTester(args)
    instance_tester.rosInit()
    rate = rospy.Rate(10) ### 10hz

    ### generate/load an instance in the execution scene
    initialize_instance_success = utils2.serviceCall_generateInstanceCylinder(
        instance_tester.num_objects, instance_tester.instance_id, instance_tester.isNewInstance)
    if initialize_instance_success:
        ### object pose estimation
        cylinder_objects = utils2.serviceCall_cylinderPositionEstimate()
        ### reproduce the estimated object poses in the planning scene
        initial_arrangement, final_arrangement, reproduce_instance_success = \
                utils2.serviceCall_reproduceInstanceCylinder(cylinder_objects)
        ### generate IK config for start positions for all objects
        ik_generate_success = utils2.serviceCall_generateConfigsForStartPositions("Right_torso")
        
        all_methods_time = []
        all_methods_success = [] ### 0: fail, 1: success
        all_methods_nActions = []

        ###### now using different methods to solve the instance ######
        ### (i) CIRS
        start_time = time.time()
        unidir_cirs_planner = UnidirCIRSPlanner(
            initial_arrangement, final_arrangement, instance_tester.time_allowed)
        cirs_planning_time = time.time() - start_time
        cirs_isSolved = unidir_cirs_planner.isSolved
        cirs_nActions = unidir_cirs_planner.best_solution_cost
        if cirs_nActions == np.inf:
            cirs_nActions = 5000
        cirs_object_ordering = unidir_cirs_planner.object_ordering
        all_methods_time.append(cirs_planning_time)
        all_methods_success.append(float(cirs_isSolved))
        all_methods_nActions.append(cirs_nActions)

        #####################################################################
        reset_instance_success = utils2.resetInstance("Right_torso")
        #####################################################################

        ### (ii) DFSDP
        start_time = time.time()
        unidir_dfsdp_planner = UnidirDFSDPPlanner(
            initial_arrangement, final_arrangement, instance_tester.time_allowed)
        DFSDP_planning_time = time.time() - start_time
        DFSDP_isSolved = unidir_dfsdp_planner.isSolved
        DFSDP_nActions = unidir_dfsdp_planner.best_solution_cost
        if DFSDP_nActions == np.inf:
            DFSDP_nActions = 5000
        DFSDP_object_ordering = unidir_dfsdp_planner.object_ordering
        all_methods_time.append(DFSDP_planning_time)
        all_methods_success.append(float(DFSDP_isSolved))
        all_methods_nActions.append(DFSDP_nActions)

        #####################################################################
        reset_instance_success = utils2.resetInstance("Right_torso")
        #####################################################################

        ### (iv) mRS
        start_time = time.time()
        unidir_mrs_planner = UnidirMRSPlanner(
            initial_arrangement, final_arrangement, instance_tester.time_allowed)
        mRS_planning_time = time.time() - start_time
        mRS_isSolved = unidir_mrs_planner.isSolved
        mRS_nActions = unidir_mrs_planner.best_solution_cost
        if mRS_nActions == np.inf:
            mRS_nActions = 5000
        mRS_object_ordering = unidir_mrs_planner.object_ordering
        all_methods_time.append(mRS_planning_time)
        all_methods_success.append(float(mRS_isSolved))
        all_methods_nActions.append(mRS_nActions)

        #####################################################################
        reset_instance_success = utils2.resetInstance("Right_torso")
        #####################################################################


        print("\n")
        print("Time for CIRS planning is: {}".format(cirs_planning_time))
        print("Success for CIRS planning is: {}".format(cirs_isSolved))
        print("Number of actions for CIRS planning: {}".format(cirs_nActions))
        print("Object ordering for CIRS planning is: {}".format(cirs_object_ordering))
        print("\n")
        print("Time for DFSDP planning is: {}".format(DFSDP_planning_time))
        print("Success for DFSDP planning is: {}".format(DFSDP_isSolved))
        print("Number of actions for DFSDP planning is: {}".format(DFSDP_nActions))
        print("Object ordering for DFSDP planning is: {}".format(DFSDP_object_ordering))
        print("\n")
        print("Time for mRS planning is: {}".format(mRS_planning_time))
        print("Success for mRS planning is: {}".format(mRS_isSolved))
        print("Number of actions for mRS planning is: {}".format(mRS_nActions))
        print("Object ordering for mRS planning is: {}".format(mRS_object_ordering))
        print("\n")

        if instance_tester.isNewInstance:
            ### only keep the option to save instance when it is a new instance
            saveInstance = True if input("save instance? (y/n)") == 'y' else False
            print("save instance: " + str(saveInstance))
            if saveInstance:
                utils2.saveInstance(cylinder_objects, instance_tester.instanceFolder)
        saveSolution = True if input("save solution? (y/n)") == 'y' else False
        print("save solution: " + str(saveSolution))
        if saveSolution:
            utils2.saveSolution(all_methods_time, all_methods_success, all_methods_nActions,
                                instance_tester.instanceFolder)

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)