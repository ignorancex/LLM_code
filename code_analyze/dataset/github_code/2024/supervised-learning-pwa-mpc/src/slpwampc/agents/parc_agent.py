import matplotlib.pyplot as plt
import numpy as np
from dmpcpwa.mpc.mpc_mld import MpcMld
from gymnasium import Env

from slpwampc.core.parc import Parc
from slpwampc.core.systems import PwaSystem
from slpwampc.misc.action_mapping import PwaActionMapper
from slpwampc.misc.regions import Polytope


class ParcAgent:
    """An agent who trains an oblique decision tree policy, for the switching sequences of a PWA system, in a supervised manner."""

    def __init__(
        self,
        system: PwaSystem,
        mpc: MpcMld,
        N: int,
        first_region_from_policy: bool = False,
        tightened_mpc: MpcMld | None = None,
        learn_infeasible_regions: bool = False,
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
        self.parc = Parc(
            K=30,
            alpha=1.0e2,
            maxiter=150,
            sigma=10,
            separation="Softmax",
            verbose=0,
            min_number=1,
        )
        # len(system.A)**N

    def get_switching_sequence(self, x: np.ndarray) -> np.ndarray | None:
        """Get the switching sequence for a given state.

        Parameters
        ----------
        x : np.ndarray
            The state.

        Returns
        -------
        np.ndarray | None
            The switching sequence or none if out of feasible set."""
        pred = self.parc.predict(x.T)[0]
        if pred == -1:
            return None
        else:
            return self.action_mapper.get_action_from_label(pred)[0]

    def evaluate(
        self,
        env: Env,
        num_episodes: int,
        seed: int = 0,
        use_learned_policy: bool = True,
        K_term: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Evaluate the policy in the environment.

        Parameters
        ----------
        env : Env
            The environment.
        num_episodes : int
            The number of episodes.
        seed : int, optional
            The seed for reseting environment, by default 0.
        use_learned_policy : bool, optional
            If true the learned policy selects the switching sequences, otherwise the mixed-integer MPC is used, by default True.
        K_term : np.ndarray, optional
            An optional terminal controller, used to guarentee recursive feasibility.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[np.ndarray]]
            The costs, initial states, and solve times for each episode.
        """
        costs = np.zeros(num_episodes)
        states = np.zeros((num_episodes, self.nx))
        times: list[np.ndarray] = []
        for i in range(num_episodes):
            x, _ = env.reset(seed=seed + i)
            previous_seq = None
            states[i] = x.squeeze()
            truncated, terminated, timestep = False, False, 0
            solve_times: list[float] = []
            while not truncated and not terminated:
                if use_learned_policy:
                    seq = self.get_switching_sequence(x)
                    if seq is None:
                        if previous_seq is None:
                            raise ValueError(
                                "Sequence returned for initial state not feasible."
                            )
                        else:
                            if K_term is None:
                                raise ValueError("Terminal controller not provided.")
                            x_N = info["x"][
                                :, -1
                            ]  # terminal state from previous iteration
                            _, shifted_region = self.system.next_state(
                                x_N, K_term @ x_N
                            )
                            seq = np.vstack((previous_seq[1:], shifted_region))
                    u, info = self.mpc.solve_mpc_with_switching_sequence(x, seq)
                    if info["cost"] == float("inf"):
                        raise ValueError("Infeasible problem encountered.")
                    previous_seq = seq
                else:
                    u, info = self.mpc.solve_mpc(x)
                solve_times.append(info["run_time"])
                x, r, truncated, terminated, _ = env.step(u)
                costs[i] += r
                timestep += 1
            times.append(np.array(solve_times))
        return costs, states, times

    def train(
        self,
        initial_state_set: np.ndarray,
        plot: bool = False,
        interactive: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
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
        tuple[np.ndarray, np.ndarray, dict]
            The training data that generates the policy and an info dict."""
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
            print(f"Fitting parc with {state_train_set.shape[0]} samples.")

            self.parc.fit(state_train_set, action_train_set.ravel(), categorical=[True])

            regions = self.parc.get_partition(self.system.D, self.system.E)

            # label each region using the trained tree # TODO get labels directly from c code
            for region in regions:
                region.set_label(lambda x: self.parc.predict(x.T)[0].item())

            for region in regions:  # TODO remove this check
                if not region.is_empty:
                    x = region.get_point()
                    label = self.parc.predict(x.T)[0].item()
                    if label != region.label:
                        raise ValueError("Region label does not match tree prediction.")

            infeas_vertices = np.empty((0, self.nx, 1))
            all_vertices = np.empty((0, self.nx))
            num_regions = 0
            for region in regions:
                if not region.is_empty:
                    num_regions += 1
                    vertices = region.V
                    label = region.label
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
                                vertex.reshape(-1, 1), switching_sequence, raises=False
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
            percentage_infeas = infeas_vertices.shape[0] / all_vertices.shape[0] * 100
            print(f"Number of vertices: {all_vertices.shape[0]}")
            print(f"Number of infeasible vertices: {infeas_vertices.shape[0]}")
            print(f"Percentage of vertices infeas: {percentage_infeas}%")

            if plot:
                self.plot_iteration(
                    iter,
                    self.nx == 2,
                    interactive,
                    regions,
                    state_train_set,
                    infeas_vertices,
                    np.unique(action_train_set),
                )

            state_set = infeas_vertices
            if infeas_vertices.shape[0] == 0:
                if plot:
                    plt.pause(1e5)
                    plt.ioff()
                return (
                    state_train_set,
                    action_train_set,
                    {"iters": iter, "num_regions": num_regions},
                )
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
            optimal_action = self.get_switching_sequence_from_mpc(state)
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

    def get_switching_sequence_from_mpc(self, x: np.ndarray) -> np.ndarray | None:
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

    def plot_iteration(
        self,
        iter: int,
        is_2D: bool,
        interactive: bool,
        regions: list[Polytope],
        state_train_set: np.ndarray,
        infeas_vertices: np.ndarray,
        unique_labels: np.ndarray,
    ) -> None:
        """Plot the current iteration of the training. This plots the current partition, the training data, and the infeasible vertices.

        Parameters
        ----------
        iter : int
            The current iteration.
        is_2D : bool
            If true the plot is 2D, otherwise 3D.
        interactive : bool
            If true the plot is interactive and the axis updates dynamically. Otherwise the figures are blocking.
        regions : list[Polytope]
            The regions of the current SVM policy.
        state_train_set : np.ndarray
            The training data.
        infeas_vertices : np.ndarray
            The current infeasible vertices.
        unique_labels : np.ndarray
            The unique labels."""
        if interactive:
            self.ax.clear()

        self.ax.set_xlim(-21, 21)
        self.ax.set_ylim(-21, 21)

        for label in unique_labels:
            self.ax.set_title(f"label: {label}")
            for region in [r for r in regions if not r.is_empty and r.label == label]:
                if not region.is_empty:
                    region.plot(
                        ax=self.ax,
                        color=f"C{int(region.label)}" if region.label >= 0 else "black",
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

        # fig, ax = plt.subplots()
        # ax.set_xlim(-21, 21)
        # ax.set_ylim(-21, 21)
        # self.parc.plot_partition([-15, -15], [15, 15])
        # plt.show()

        # if iter in [0, 10, 25]:
        #     save2tikz(plt.gcf())
        if not interactive:
            plt.show()
        else:
            self.fig.canvas.draw()
            plt.pause(0.01)

    def save(self, path: str) -> None:
        """Save the predictor to a file.

        Parameters
        ----------
        path : str
            The path to the file."""
        self.parc.save(path)

    def load(self, path: str) -> None:
        """Load the predictor from a file.

        Parameters
        ----------
        path : str
            The path to the file."""
        self.parc.load(path)
