import matplotlib.pyplot as plt
import numpy as np
from dmpcpwa.mpc.mpc_mld import MpcMld
from rollout.core.systems import PwaSystem
from rollout.misc.action_mapping import PwaActionMapper
from rollout.misc.regions import Polytope

# from rollout.misc.regions import find_vertices
from sklearn_oblique_tree.oblique import ObliqueTree


class ObliqueDecisionTreeAgent:
    """An agent who trains an oblique decision tree policy, for the switching sequences of a PWA system, in a supervised manner."""

    def __init__(
        self,
        system: PwaSystem,
        mpc: MpcMld,
        N: int,
        num_trees: int = 1,
        first_region_from_policy: bool = False,
        tightened_mpc: MpcMld | None = None,
        learn_infeasible_regions: bool = False,
        num_restarts: int = 20,
        max_perturbations: int = 3,
    ) -> None:
        """Initialize the agent.

        Parameters
        ----------
        system : PwaSystem
            The PWA system.
        mpc : MpcMld
            The mixed-integer MPC.
        N : int
            The prediction horizon.
        num_trees : int, optional
            The number of trees in the forest, by default 1. When more than 1, the trees are all trained for a given data-set, and the
            tree with the lowest number of incorrect vertices is selected.
        first_region_from_policy : bool, optional
            If true the policy outputs sequences of length N, defining also the PWa region at the first timestep, otherwise N-1. By default False.
        tightened_mpc : MpcMld, optional
            A tightened version of the MPC, used for learning feasible region with a small margin. By default None.
        learn_infeasible_regions : bool, optional
            If true, the agent learns also infeasible regions, by default False.
        """
        self.nx = system.A[0].shape[0]
        self.nu = system.B[0].shape[1]
        self.action_mapper = PwaActionMapper(
            len(system.A), N if first_region_from_policy else mpc.N - 1
        )
        self.system = system
        self.mpc = mpc
        self.tightened_mpc = tightened_mpc if tightened_mpc is not None else mpc
        self.N = N
        self.first_region_from_policy = first_region_from_policy
        self.learn_infeasible_regions = learn_infeasible_regions
        self.trees = [
            ObliqueTree(
                splitter="oc1",
                number_of_restarts=num_restarts,
                max_perturbations=max_perturbations,
                random_state=i,
            )
            for i in range(num_trees)
        ]
        # self.tree = ObliqueTree(splitter="oc1, axis_parallel", number_of_restarts=1, max_perturbations=3, random_state=0)
        # self.tree = ObliqueTree(splitter="cart", number_of_restarts=20, max_perturbations=3, random_state=0)

    def evaluate(self):  # TODO implement
        raise NotImplementedError()

    def train(
        self,
        initial_state_set: np.ndarray,
        plot: bool = False,
        interactive: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Train the decision tree policy.

        Parameters
        ----------
        initial_state_set : np.ndarray
            The initial state set for training. Each iteration the state set changes.
        plot : bool, optional
            If true the training is plotted. By default False.
        interactive : bool, optional
            If true the training is plotted interactively. By default False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            The training data that generates the tree policy, and the random state for the tree training.
        """
        if plot and self.nx > 3:
            raise ValueError("Plotting only supported for 2D and 3D.")
        if plot and interactive:
            if self.nx == 2:
                self.fig, self.ax = plt.subplots()
            elif self.nx == 3:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection="3d")
            plt.ioff()

        state_train_set = np.empty((0, self.nx))
        action_train_set = np.empty((0, 1))

        state_set = initial_state_set
        iter = 0
        while True:
            if plot and not interactive:
                if self.nx == 2:
                    self.fig, self.ax = plt.subplots()
                else:
                    self.fig = plt.figure()
                    self.ax = self.fig.add_subplot(111, projection="3d")
            print(f"Training iteration {iter}. {state_set.shape[0]} new states.")
            optimal_states, optimal_actions = self.generate_supervised_learning_data(
                state_set
            )
            state_train_set = np.vstack((state_train_set, optimal_states))
            action_train_set = np.vstack((action_train_set, optimal_actions))
            print(f"Fitting decision tree with {state_train_set.shape[0]} samples.")

            label_map, odt_labels = self.get_oblique_label_map(
                action_train_set
            )  # map labels to odt representation

            for idx, tree in enumerate(self.trees):
                tree.fit(state_train_set, odt_labels)

                partition = tree.get_partition()
                regions = [
                    Polytope(
                        np.vstack((r[:, :-1], self.system.D)),
                        np.vstack((r[:, [-1]], self.system.E)),
                    )
                    for r in partition
                ]

                # label each region using the trained tree # TODO get labels directly from c code
                for region in regions:
                    region.set_label(lambda x: tree.predict(x.T))

                for region in regions:  # TODO remove this check
                    if not region.is_empty:
                        x = region.get_point()
                        label = tree.predict(x.T)
                        if label != region.label:
                            raise ValueError(
                                "Region label does not match tree prediction."
                            )

                infeas_vertices = np.empty((0, self.nx, 1))
                all_vertices = np.empty((0, self.nx))
                for region in regions:
                    if not region.is_empty:
                        vertices = region.V
                        label = label_map[
                            region.label.item()
                        ]  # map labels back to original representation
                        all_vertices = np.vstack((all_vertices, vertices))

                        if self.learn_infeasible_regions and label == -1:
                            for vertex in vertices:
                                _, sol = self.tightened_mpc.solve_mpc(
                                    vertex.reshape(-1, 1),
                                    raises=False,
                                    try_again_if_infeasible=False,
                                )
                                if sol["cost"] != float("inf"):
                                    infeas_vertices = np.vstack(
                                        (infeas_vertices, vertex.reshape(1, -1, 1))
                                    )
                        else:
                            # action mapper returns tensor that is vectorizable. Hence we need to convert it to numpy array and remove extra dims
                            switching_sequence = np.asarray(
                                self.action_mapper.get_action_from_label(
                                    np.asarray(label).reshape(1, 1)
                                )
                            ).reshape(-1)

                            for vertex in vertices:
                                _, sol = self.mpc.solve_mpc_with_switching_sequence(
                                    vertex.reshape(-1, 1),
                                    switching_sequence,
                                    raises=False,
                                )
                                if sol["cost"] == float("inf"):
                                    infeas_vertices = np.vstack(
                                        (infeas_vertices, vertex.reshape(1, -1, 1))
                                    )

                # remove duplicates
                all_vertices = np.array(list(set(map(tuple, all_vertices))))
                if infeas_vertices.shape[0] > 0:
                    infeas_vertices = np.array(
                        list(set(map(tuple, infeas_vertices.squeeze(-1))))
                    )[:, :, None]
                percentage_infeas = (
                    infeas_vertices.shape[0] / all_vertices.shape[0] * 100
                )
                print(f"Tree {idx}:")
                print(f"Number of vertices: {all_vertices.shape[0]}")
                print(f"Number of infeasible vertices: {infeas_vertices.shape[0]}")
                print(f"Percentage of vertices infeas: {percentage_infeas}%")

                if idx == 0 or percentage_infeas < best_percentage_infeas:
                    best_regions = regions
                    best_infeas_vertices = infeas_vertices
                    best_random_state = idx

            if plot:
                self.plot_iteration(
                    self.nx == 2,
                    interactive,
                    best_regions,
                    state_train_set,
                    best_infeas_vertices,
                    odt_labels,
                    label_map,
                )

            state_set = best_infeas_vertices
            if best_infeas_vertices.shape[0] == 0:
                if plot:
                    plt.pause(1e5)
                    plt.ioff()
                return state_train_set, action_train_set, best_random_state
            iter += 1

    def generate_supervised_learning_data(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """For a given set of states, generate optimal switching sequences.

        Parameters
        ----------
        x : np.ndarray
            The states. Shape (num_states, nx, 1)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The states and the optimal switching sequences."""
        valid_states: list[np.ndarray] = []
        valid_actions: list[int] = []
        for idx, state in enumerate(x):
            # print(f"evaluating optimal action for state {idx} of {x.shape[0]}")
            if self.learn_infeasible_regions:
                valid_states.append(state)
            optimal_action = self.get_switching_sequence_from_state(state)
            if optimal_action is not None:
                if not self.learn_infeasible_regions:
                    valid_states.append(state)
                valid_actions.append(
                    self.action_mapper.get_label_from_action(optimal_action)
                )
            elif self.learn_infeasible_regions:
                valid_actions.append(-1)
        # map the lists to numpy arrays
        optimal_states = np.asarray(valid_states).reshape(-1, self.nx)
        optimal_actions = np.asarray(valid_actions).reshape(-1, 1)
        return optimal_states, optimal_actions

    def get_switching_sequence_from_state(self, x: np.ndarray) -> np.ndarray | None:
        """Get the switching sequence from a given state by solving the MPC problem.

        Parameters
        ----------
        x : np.ndarray
            The state.

        Returns
        -------
        np.ndarray | None
            The switching sequence if the problem is feasible, otherwise None."""
        switching_sequence = np.zeros(
            (self.N if self.first_region_from_policy else self.N - 1, 1)
        )
        _, sol = self.mpc.solve_mpc(x, raises=False, try_again_if_infeasible=False)
        if sol["cost"] < float("inf"):
            delta = sol["delta"]  # binary vars that represent PWA regions
            switching_sequence = np.argmax(delta, axis=0).reshape(-1, 1)
            return (
                switching_sequence
                if self.first_region_from_policy
                else switching_sequence[1:]
            )
        else:
            return None

    def get_oblique_label_map(
        self, labels: np.ndarray
    ) -> tuple[dict[int, int], np.ndarray]:
        """Get a mapping from the original labels to the labels used in the oblique tree.

        Parameters
        ----------
        labels : np.ndarray
            The original labels.

        Returns
        -------
        dict[int, int]
            The mapping.
        np.ndarray
            The labels used in the oblique tree."""
        unique_labels = list(np.unique(labels))
        label_map = {idx: label for idx, label in enumerate(unique_labels)}
        new_labels = np.array(
            [unique_labels.index(label) for label in list(labels.squeeze())]
        )
        return label_map, new_labels

    def check_training_data(
        self,
        tree: ObliqueTree,
        x: np.ndarray,
        y: np.ndarray,
        plot: bool = False,
        ax: plt.axes = None,
    ) -> None:
        """Check if the training data is correctly classified by the tree.

        Parameters
        ----------
        x : np.ndarray
            The states.
        y : np.ndarray
            The labels.
        plot : bool, optional
            If True, the incorrectly classified data is plotted, by default False.
        ax : plt.axes, optional
            The axes to plot on, by default None."""
        y_pred = tree.predict(x)
        if not np.all(y == y_pred):
            print(
                f"{np.count_nonzero(y - y_pred)} training data is not correctly classified by the tree."
            )
            # raise ValueError("Training data is not correctly classified by the tree.")
            if plot and ax is not None:
                for idx, (state, label, pred) in enumerate(zip(x, y, y_pred)):
                    if label != pred:
                        ax.plot(
                            state[0],
                            state[1],
                            color="green",
                            marker="o",
                            markersize=10,
                            linestyle="None",
                        )
        # print("Training data is correctly classified by the tree.")

    def plot_iteration(
        self,
        is_2D: bool,
        interactive: bool,
        regions: list[Polytope],
        state_train_set: np.ndarray,
        infeas_vertices: np.ndarray,
        odt_labels: np.ndarray,
        label_map: dict[int, int],
    ) -> None:
        """Plot the current iteration of the training. This plots the current partition, the training data, and the infeasible vertices.

        Parameters
        ----------
        is_2D : bool
            If true the plot is 2D, otherwise 3D.
        interactive : bool
            If true the plot is interactive and the axis updates dynamically. Otherwise the figures are blocking.
        regions : list[Polytope]
            The regions of the current SVM policy.
        state_train_set : np.ndarray
            The training data.
        infeas_vertices : np.ndarray
            The current infeasible vertices."""
        if interactive:
            self.ax.clear()

        self.ax.set_xlim(-21, 21)
        self.ax.set_ylim(-21, 21)

        # self.check_training_data(state_train_set, odt_labels, plot=True, ax=self.ax)
        for label in label_map.values():
            self.ax.set_title(f"label: {label}")
            for region in [
                r
                for r in regions
                if not r.is_empty and label_map[r.label.item()] == label
            ]:
                if not region.is_empty:
                    region.plot(
                        ax=self.ax,
                        color=(
                            f"C{int(region.label.item())}"
                            if label_map[region.label.item()] >= 0
                            else "black"
                        ),
                        alpha=0.5,
                    )
                    # plt.pause(1)

        if is_2D:
            self.ax.plot(
                state_train_set[:, 0],
                state_train_set[:, 1],
                color="blue",
                marker="o",
                markersize=2,
                linestyle="None",
            )
            self.ax.plot(
                infeas_vertices[:, 0],
                infeas_vertices[:, 1],
                color="red",
                marker="x",
                markersize=5,
                linestyle="None",
            )
            plt.xlim(-21, 21)
            plt.ylim(-21, 21)
        else:
            self.ax.scatter(
                state_train_set[:, 0],
                state_train_set[:, 1],
                state_train_set[:, 2],
                color="blue",
                marker="o",
                s=2,
            )
            self.ax.scatter(
                infeas_vertices[:, 0],
                infeas_vertices[:, 1],
                infeas_vertices[:, 2],
                color="red",
                marker="x",
                s=5,
            )
            self.ax.set_zlim(-12, 12)

        if not interactive:
            plt.show()
        else:
            self.fig.canvas.draw()
            plt.pause(0.01)
