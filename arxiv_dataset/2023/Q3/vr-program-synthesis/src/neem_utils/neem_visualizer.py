from argparse import ArgumentParser
from typing import List

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from neem_interface_python.neem import NEEM

from neem_utils.knowrob_queries import get_states_for_transition, parse_tf_traj, is_robot, get_end_effector_for_robot, \
    get_interval_for_event

FLANGE_INDIVIDUAL = "http://knowrob.org/kb/UR5Robotiq.owl#tool0"


class NEEMVisualizer:
    def __init__(self, neem: NEEM):
        self.neem = neem

    def visualize_trajectories(self):
        transition_iris = self.neem.get_transitions()
        participants = self.neem.get_participants()
        for i, participant in enumerate(participants):
            if is_robot(self.neem.prolog, participant):
                # participants[i] = get_end_effector_for_robot(self.neem.prolog, participant)
                participants[i] = FLANGE_INDIVIDUAL
        transitions = []
        for transition in transition_iris:
            initial_state, terminal_state = get_states_for_transition(self.neem.prolog, transition)
            initial_state_begin, initial_state_end = get_interval_for_event(self.neem.prolog, initial_state)
            terminal_state_begin, terminal_state_end = get_interval_for_event(self.neem.prolog, terminal_state)
            participant_trajs = []
            for i, participant in enumerate(participants):
                participant_traj = self.neem.neem_interface.get_tf_trajectory(participant, initial_state_begin, terminal_state_end)
                participant_trajs.append(parse_tf_traj(participant_traj))
            transitions.append({"begin": initial_state_begin,
                                "end": terminal_state_end,
                                "trajs": participant_trajs})

        fig, axes = plt.subplots(3, 1)
        cmap = plt.get_cmap('tab10')
        cnorm = Normalize(vmin=0, vmax=len(transitions))
        participant_linestyles = ["dashed", "dotted"]
        for i, participant in enumerate(participants):
            print(f"{i}: {participant}")
            for t, transition in enumerate(transitions):
                print(f"\t{t}: {transition['begin']} -> {transition['end']}: {len(transition['trajs'][i])} points")
                timestamps = [dp.timestamp for dp in transition["trajs"][i]]
                for dim in range(len(axes)):
                    data = [dp.pos[dim] for dp in transition["trajs"][i]]
                    axes[dim].plot(timestamps, data, linestyle=participant_linestyles[i], color=cmap(cnorm(t)))
        for ax in axes:
            for transition in transitions:
                ax.axvline(transition["begin"], color="black")
                ax.axvline(transition["end"], color="black")

        # fig.legend()
        plt.show()


def main(args):
    neem = NEEM.load(args.neem_dir)
    viz = NEEMVisualizer(neem)
    viz.visualize_trajectories()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("neem_dir", type=str, help="Directory of a single NEEM")
    main(parser.parse_args())
