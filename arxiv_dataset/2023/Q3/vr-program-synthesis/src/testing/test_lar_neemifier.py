import os
import shutil
import unittest

from neem_utils.lar_neemifier import OntologyInfo, LARNeemifier
from neem_utils.neem_parser import NEEMParser

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


class TestLARNeemifier(unittest.TestCase):
    def setUp(self) -> None:
        self.db_credentials = {
            "host": "nb067",
            "database": "lar_local",
            "user": "root",
            "password": "root"
        }
        self.ontology_info = OntologyInfo(env_owl="package://vr_program_synthesis/owl/motherboard_experiment_lab.owl",
                                          env_owl_ind_name="http://knowrob.org/kb/motherboard_experiment_lab.owl#motherboard_experiment_lab",
                                          env_urdf="package://vr_program_synthesis/urdf/maps/motherboard_experiment_lab.urdf",
                                          env_urdf_prefix="http://knowrob.org/kb/motherboard_experiment_lab.owl#",
                                          agent_owl="package://vr_program_synthesis/owl/UR5Robotiq.owl",
                                          agent_owl_ind_name="http://knowrob.org/kb/UR5Robotiq.owl#UR5Robotiq_0",
                                          agent_urdf="package://vr_program_synthesis/urdf/robots/ur5_robotiq.urdf")
        self.neemifier = LARNeemifier(self.ontology_info, self.db_credentials)
        self.neem_parser = NEEMParser()
        self.neem_output_dir = os.path.join(SCRIPT_DIR, "neems")
        if not os.path.exists(self.neem_output_dir):
            os.makedirs(self.neem_output_dir)

    def tearDown(self) -> None:
        if os.path.exists(self.neem_output_dir):
            shutil.rmtree(self.neem_output_dir)

    def testNeemifyRun(self):
        neem_dir_contents_before = os.listdir(self.neem_output_dir)
        self.neemifier.neemify_run(46301, "http://www.ease-crc.org/ont/SOMA.owl#Transporting", self.neem_output_dir)
        new_neem = list(set(os.listdir(self.neem_output_dir)) - set(neem_dir_contents_before))[0]
        parsed_neem = self.neem_parser.parse_neem(os.path.join(self.neem_output_dir, new_neem))
        self.assertEqual(len(parsed_neem.actions), 5)
        self.assertEqual(len(list(filter(lambda act: "CloseGripper" in act.name, parsed_neem.actions))), 1)
        self.assertEqual(len(list(filter(lambda act: "OpenGripper" in act.name, parsed_neem.actions))), 1)
        self.assertEqual(len(list(filter(lambda act: "MoveLinear" in act.name, parsed_neem.actions))), 1)
        self.assertEqual(len(list(filter(lambda act: "MoveToState" in act.name, parsed_neem.actions))), 2)
