from typing import List

from neem_interface_python.neem_interface import NEEMInterface
from neem_interface_python.rosprolog_client import Prolog, atom
from scipy.spatial.transform import Rotation


class Datapoint:
    def __init__(self, timestamp: float, reference_frame: str, pos: List[float], ori: Rotation,
                 wrench: List[float] = None):
        """
        :param timestamp:
        :param reference_frame: e.g. 'world'
        :param pos: [x,y,z]
        :param ori: [qx,qy,qz,qw]
        :param wrench: [fx,fy,fz,mx,my,mz]
        """
        self.timestamp = timestamp
        self.reference_frame = reference_frame
        self.pos = pos
        self.ori = ori
        self.wrench = wrench


class Action:
    """
    Represents a KnowRob individual of class Action, with associated trajectory
    """
    def __init__(self, name: str, type_: str, datapoints: List[Datapoint]):
        self.name = name    # IRI of the Action individual
        self.type = type_   # Class of the Action individual
        self.datapoints = datapoints


class NEEM:
    """
    Represents a NEEM and provides actions to access its properties
    """
    def __init__(self):
        self.neem_interface = NEEMInterface()
        self.prolog = Prolog()
        self.episode = self.prolog.once("kb_call(is_episode(Episode))")["Episode"]

    def get_action_trajectories(self) -> List[Action]:
        action_iris = self.neem_interface.get_all_actions()
        actions = []
        for action_iri in action_iris:
            action_type = self.prolog.once(f"instance_of({atom(act)}, ActionType)")["ActionType"]
            interval = self.neem_interface.get_interval_for_event(act)
            if interval is None:
                continue
            action_start, action_end = interval

            # Always get the TF trajectory
            tf_traj = self.neem_interface.get_tf_trajectory("ee_link", action_start, action_end)
            datapoints = []
            for dp in tf_traj:
                ori = Rotation.from_quat(dp["term"][2][2])
                datapoint = Datapoint(timestamp=dp["term"][1], reference_frame=dp["term"][2][0],
                                      pos=dp["term"][2][1], ori=ori)
                datapoints.append(datapoint)

            # Optionally add wrench data, if available
            try:
                wrench_traj = self.neem_interface.get_wrench_trajectory("ee_link", action_start, action_end)
                assert len(wrench_traj) == len(datapoints)
                for i, dp in enumerate(wrench_traj):
                    datapoints[i].wrench = [item for sublist in dp["term"][2] for item in
                                            sublist]  # TODO: THIS IS WRONG
            except Exception as e:
                print(f"No FT data available: {e}")
            actions.append(Action(action_iri, action_type, datapoints))
        return actions

    def get_transitions(self) -> List[str]:
        """
        Get a list of transition IRIs associated to this NEEM, sorted by time
        """
        top_level_action = self.get_top_level_action()
        res = self.prolog.once(f"""
            kb_call([
                holds({atom(top_level_action)}, 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasTimeInterval', TimeInterval),
                holds(TimeInterval, 'http://www.ease-crc.org/ont/SOMA.owl#hasIntervalBegin', StartTime),
                holds(TimeInterval, 'http://www.ease-crc.org/ont/SOMA.owl#hasIntervalEnd', EndTime)
            ]).
        """)
        start_time = res["StartTime"]
        end_time = res["EndTime"]
        res = self.prolog.all_solutions(f"kb_call(is_transition(Transition)).")
        all_transitions = [sol["Transition"] for sol in res]
        transitions_by_time = dict()
        for transition in all_transitions:
            res = self.prolog.once(f"""
                kb_call([
                    holds({atom(transition)}, 'http://www.ease-crc.org/ont/SOMA.owl#hasInitialScene', InitialScene),
                    is_state(InitialState), holds(InitialScene, 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#includesEvent', InitialState),
                    has_time_interval(InitialState, TransitionStartTime, TransitionEndTime)
                 ]).
            """)
            transition_start_time = res["TransitionStartTime"]
            transition_end_time = res["TransitionEndTime"]
            if transition_start_time >= start_time and transition_end_time <= end_time:
                # This transition is part of this neem
                transitions_by_time[transition_start_time] = transition
        return [kv[1] for kv in sorted(transitions_by_time.items())]

    def get_top_level_action(self) -> str:
        solutions = self.prolog.all_solutions(f"""
            kb_call([
                is_action(Action), is_setting_for({atom(self.episode)}, Action)
            ]).
        """)
        return solutions[0]["Action"]

    @staticmethod
    def load(neem_dir: str):
        NEEMInterface().load_neem(neem_dir)
        return NEEM()


if __name__ == '__main__':
    neem = NEEM.load("/home/lab019/alt/catkin_ws/src/ilias/vr_program_synthesis/neems/1628091371.0404518")
