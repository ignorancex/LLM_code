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
### This class defines a ExampleRunner class which
### solves an rearrangement problem/example with 
### the number of the object specified
### It
### (1) asks the execution scene to generate an instance
### (2) asks the pose estimator to get the object poses
### (3) reproduces the instance in task planner 
### (4) solves it with a specified method
### (5) obtains the solution of the specified method
### (6) asks the execution scene to execute the planned paths
#####################################################################################

class ExampleRunner(object):
    def __init__(self, args):
        ### set the rospkg path
        rospack = rospkg.RosPack()
        self.rosPackagePath = rospack.get_path("uniform_object_rearrangement")
        self.num_objects = int(args[1])
        self.instance_id = int(args[2])
        self.isNewInstance = True if args[3] == 'g' else False
        self.time_allowed = int(args[4])
        self.method_name = args[5]
        self.instanceFolder = os.path.join(
            self.rosPackagePath, "examples", str(self.num_objects), str(self.instance_id))

    def rosInit(self):
        ### This function specifies the role of a node instance for this class ###
        ### and initializes a ros node
        rospy.init_node("run_example", anonymous=True)


def main(args):
    example_runner = ExampleRunner(args)
    example_runner.rosInit()
    rate = rospy.Rate(10) ### 10hz

    ### generate/load an instance in the execution scene
    initialize_instance_success = utils2.serviceCall_generateInstanceCylinder(
        example_runner.num_objects, example_runner.instance_id, example_runner.isNewInstance)
    if initialize_instance_success:
        ### object pose estimation
        cylinder_objects = utils2.serviceCall_cylinderPositionEstimate()
        ### reproduce the estimated object poses in the planning scene
        initial_arrangement, final_arrangement, reproduce_instance_success = \
                utils2.serviceCall_reproduceInstanceCylinder(cylinder_objects)
        ### generate IK config for start positions for all objects
        ik_generate_success = utils2.serviceCall_generateConfigsForStartPositions("Right_torso")

        ###### run an example given the method specified ######
        ### (i) CIRS
        if example_runner.method_name == "CIRS":
            start_time = time.time()
            the_chosen_planner = UnidirCIRSPlanner(
                initial_arrangement, final_arrangement, example_runner.time_allowed)
        ### (ii) CIRS_nonlabeled
        if example_runner.method_name == "CIRS_nonlabeled":
            start_time = time.time()
            the_chosen_planner = UnidirCIRSPlanner(
                initial_arrangement, final_arrangement, example_runner.time_allowed, \
                isLabeledRoadmapUsed=False)
        ### (iii) DFS_DP
        if example_runner.method_name == "DFSDP":
            start_time = time.time()
            the_chosen_planner = UnidirDFSDPPlanner(
                initial_arrangement, final_arrangement, example_runner.time_allowed)
        ### (iv) DFS_DP_nonlabeled
        if example_runner.method_name == "DFSDP_nonlabeled":
            start_time = time.time()
            the_chosen_planner = UnidirDFSDPPlanner(
                initial_arrangement, final_arrangement, example_runner.time_allowed, \
                isLabeledRoadmapUsed=False)
        ### (iv) mRS
        if example_runner.method_name == "mRS":
            start_time = time.time()
            the_chosen_planner = UnidirMRSPlanner(
                initial_arrangement, final_arrangement, example_runner.time_allowed)
        ### (v) mRS_nonlabeled
        if example_runner.method_name == "mRS_nonlabeled":
            start_time = time.time()
            the_chosen_planner = UnidirMRSPlanner(
                initial_arrangement, final_arrangement, example_runner.time_allowed, \
                isLabeledRoadmapUsed=False)

        planning_time = time.time() - start_time
        isSolved = the_chosen_planner.isSolved
        nActions = the_chosen_planner.best_solution_cost
        if nActions == np.inf: nActions = 5000
        object_ordering = the_chosen_planner.object_ordering
        object_paths = the_chosen_planner.object_paths        

        print("\n")
        print("Time for {} planning is: {}".format(example_runner.method_name, planning_time))
        print("Number of actions for {} planning is: {}".format(example_runner.method_name, nActions))
        print("Object ordering for {} planning is: {}".format(example_runner.method_name, object_ordering))

        ### move the robot back to home configuration (optional)
        if (example_runner.method_name == "CIRS") or (example_runner.method_name == "DFSDP") \
            or (example_runner.method_name == "mRS"):
            resetHome_success, resetHome_trajectory = utils2.serviceCall_reset_robot_home("Right_torso")
        if (example_runner.method_name == "CIRS_nonlabeled") or (example_runner.method_name == "DFSDP_nonlabeled") \
            or (example_runner.method_name == "mRS_nonlabeled"):
            resetHome_success, resetHome_trajectory = utils2.serviceCall_reset_robot_home("Right_torso", False)

        if example_runner.isNewInstance:
            ### only keep the option to save instance when it is a new instance
            saveInstance = True if input("save instance? (y/n)") == 'y' else False
            print("save instance: " + str(saveInstance))
            if saveInstance:
                utils2.saveInstance(cylinder_objects, example_runner.instanceFolder)
        
        if isSolved:
            executePath = True if input("Solution found. Execute the solution? (y/n)") == 'y' else False
            print("execute solution: " + str(executePath))
            if executePath:
                if resetHome_success:
                    utils2.executeWholePlan(object_paths, resetHome_trajectory)
                else:
                    utils2.executeWholePlan(object_paths)
            savePath = True if input("Path Executed. Save the path? (y/n)") == 'y' else False
            print("save path: " + str(savePath))
            if savePath:
                if resetHome_success:
                    utils2.saveWholePlan(object_paths, example_runner.instanceFolder, resetHome_trajectory)
                else:
                    utils2.saveWholePlan(object_paths, example_runner.instanceFolder)
            saveOrderingInfo = True if input("Save the ordering for future reference? (y/n)") == 'y' else False
            print("save ordering info: " + str(saveOrderingInfo))
            if saveOrderingInfo:
                utils2.saveOrderingInfo(object_ordering, example_runner.instanceFolder)
        
        print("exeunt")

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)