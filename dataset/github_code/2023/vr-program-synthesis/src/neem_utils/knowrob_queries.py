from typing import Tuple, List

import torch
from neem_interface_python.rosprolog_client import Prolog, atom
from pytransform3d.rotations import quaternion_wxyz_from_xyzw
from scipy.spatial.transform import Rotation


def get_states_for_transition(prolog: Prolog, transition: str) -> Tuple[str, str]:
    res = prolog.once(f"""
        kb_call([
            holds({atom(transition)}, 'http://www.ease-crc.org/ont/SOMA.owl#hasInitialScene', InitialScene),
            holds({atom(transition)}, 'http://www.ease-crc.org/ont/SOMA.owl#hasTerminalScene', TerminalScene),
            holds(InitialScene, 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#includesEvent', InitialState),
            holds(TerminalScene, 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#includesEvent', TerminalState)
        ])""")
    return res["InitialState"], res["TerminalState"]


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

    def to_tensor(self) -> torch.Tensor:
        """
        :return: Tensor [x,y,z,qw,qx,qy,qz]
        """
        return torch.tensor([*self.pos, *quaternion_wxyz_from_xyzw(self.ori.as_quat())], dtype=torch.float32)


def parse_tf_traj(tf_traj: List[dict]) -> List[Datapoint]:
    """
    Parse a trajectory as returned by KnowRob (list of terms (dicts)) into a list of datapoints
    """
    datapoints = []
    timestamps = []
    for dp in tf_traj:
        ori = Rotation.from_quat(dp["term"][2][2])
        timestamp = dp["term"][1]
        if timestamp in timestamps:
            continue    # UGLY HACK: tf_mongo sometimes duplicates trajectories --> filter duplicate datapoints by timestamp
        timestamps.append(timestamp)
        datapoint = Datapoint(timestamp=timestamp, reference_frame=dp["term"][2][0],
                              pos=dp["term"][2][1], ori=ori)
        datapoints.append(datapoint)
    return datapoints


def is_robot(prolog: Prolog, thing: str) -> bool:
    """
    Something is a robot if it is an ArtificialAgent or an Arm.
    """
    try:
        prolog.once(f"""
            kb_call(instance_of({atom(thing)}, 'http://www.ease-crc.org/ont/SOMA.owl#ArtificialAgent'));
            kb_call(instance_of({atom(thing)}, 'http://www.ease-crc.org/ont/SOMA.owl#Arm')).
        """)
        return True
    except:
        return False


def get_end_effector_for_robot(prolog: Prolog, robot: str) -> str:
    try:
        return prolog.once(f"""
            kb_call(holds({atom(robot)}, urdf:'hasEndLinkName', EELinkName))
        """)["EELinkName"]
    except:
        return prolog.once(f"""
            kb_call([
              holds({atom(robot)}, 'http://knowrob.org/kb/srdl2-comp.owl#hasBodyPart', BodyPart),
              instance_of(BodyPart, 'http://www.ease-crc.org/ont/SOMA.owl#Arm'),
              holds(BodyPart, urdf:'hasEndLinkName', EELinkName)
            ])
        """)["EELinkName"]


def get_interval_for_event(prolog: Prolog, event: str) -> Tuple[float, float]:
    res = prolog.once(f"kb_call(has_time_interval({atom(event)}, StartTime, EndTime))")
    return res["StartTime"], res["EndTime"]
