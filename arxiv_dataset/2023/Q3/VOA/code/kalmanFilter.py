from typing import List
import numpy as np
from util import*

# 0.0026 is the remainder form 3 standart diviations (%99.74)


class belief:
    def __init__(self, mu: np.matrix, Sigma: np.matrix) -> None:
        """Generate the Gaussian belief values in 2D

        Args:
            mu (np.matrix): Mean values for the 2D Gaussian distribution
            Sigma (np.matrix): Variance values for the 2D Gaussian distribution
        """
        self.mu = mu
        self.Sigma = Sigma


class KalmanFilter:
    def __init__(self, A: np.matrix, B: np.matrix, C: np.matrix, R: np.matrix, Q: np.matrix, mu_0: np.matrix, Sigma_0: np.matrix, x_0: np.matrix, k: float) -> None:
        """Creating a Kalman Filater variable containing all necesary values
        for it to be able to calculate location uncertainty.

        Args:
            A (np.matrix): State transition model matrix
            B (np.matrix): Control input model matrix
            C (np.matrix): Observation model matrix
            R (np.matrix): State transition noise covariance matrix
            Q (np.matrix): Measurment noise covariance matrix
            mu_0 (np.matrix): Belief position mean matrix
            Sigma_0 (np.matrix): Belief position covariance matrix
            x_0 (np.matrix): Initial location
            k (float): Variable determining amount of standard diviations
        """
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        self.x_0 = x_0
        self.belief_0 = belief(mu_0, Sigma_0)
        self.k = k

    def PropagateUpdateBelief(self, belief_minus_1: belief, u_t: np.matrix, z_t: np.matrix = np.matrix('0,0'), with_observations: bool = False) -> belief:
        """Propogate new belief state acording to Kalman Filter and the previos belief state

        Args:
            belief_minus_1 (belief): Previos belief state
            u_t (np.matrix): Motion model
            z_t (np.matrix, optional): observation model. Defaults to np.matrix('0,0').
            with_observations (bool, optional): does the agent have observation abilities. Defaults to False.

        Returns:
            belief: Next belief state
        """
        mu_bar_t = self.A @ belief_minus_1.mu + self.B @ u_t
        Sigma_bar_t = self.A @ belief_minus_1.Sigma @ self.A.T + self.R

        if with_observations == True:
            K = Sigma_bar_t @ self.C.T @ (self.C @
                                          Sigma_bar_t @ self.C.T + self.Q).I
            mu_t = mu_bar_t + K @ (z_t - self.C @ mu_bar_t)
            Sigma_t = (np.matrix(np.eye(len(K @ self.C))) -
                       K @ self.C) @ Sigma_bar_t

        else:
            mu_t = mu_bar_t
            Sigma_t = Sigma_bar_t

        belief_t = belief(mu_t, Sigma_t)
        return belief_t

    def SampleMotionModel(self, x: np.matrix, u: np.matrix) -> np.matrix:
        """Generate a new location for an agent with Gaussian location uncertainy

        Args:
            x (np.matrix): Current real location of the agent
            u (np.matrix): Given motion

        Returns:
            np.matrix: next position given by the motion model
        """
        epsilon_mean = np.transpose(np.zeros(len(x)))
        epsilon = np.matrix(
            np.random.multivariate_normal(epsilon_mean, self.R, 1))
        epsilon = epsilon.T
        x_next = self.A @ x + self.B @ u + epsilon
        return x_next

    def priorySimWithOneCourseCorrection(self, action_speed: float, sol: List[NodePRM], help_step: int = -1, route_point: int = -1):
        route_points = initializeAction(sol)
        if route_point != -1:
            route_points = [self.belief_0.mu] + route_points[route_point:]
        Belief = [self.belief_0]
        num_of_steps = 0

        for k in range(len(route_points) - 1):
            curr_pos = (((Belief[-1]).mu).A1).tolist()
            next_pos = route_points[k+1]
            U = actionsWithPositions(curr_pos, next_pos, action_speed)

            if num_of_steps == help_step:
                break

            for i in range(len(U)):
                belief_new = self.PropagateUpdateBelief(Belief[-1], U[i])
                Belief.append(belief_new)
                num_of_steps += 1
                if num_of_steps == help_step:
                    break

        return Belief[-1].mu, Belief[-1].Sigma, Belief, k

    def runSimWithOneCourseCorrection(self, action_speed, sol, help_step=-1):
        route_points = initializeAction(sol)
        X = [self.x_0]
        Belief = [self.belief_0]
        X_cor = [self.x_0]
        Belief_cor = [self.belief_0]
        num_of_steps = 0
        help_happened = False

        relocation = True
        localization = False

        for k in range(len(route_points) - 1):
            curr_pos = (((Belief[-1]).mu).A1).tolist()
            next_pos = route_points[k+1]
            U = actionsWithPositions(curr_pos, next_pos, action_speed)

            for i in range(len(U)):
                x_new = self.SampleMotionModel(X[-1], U[i])
                belief_new = self.PropagateUpdateBelief(Belief[-1], U[i])
                X.append(x_new)
                Belief.append(belief_new)

                if help_step == num_of_steps and help_happened == False:
                    curr_pos_cor = ((X[-1]).A1).tolist()
                    next_pos_cor = [((Belief[-1].mu).A)[0][0],
                                    ((Belief[-1].mu).A)[1][0]]
                    belief_new = belief(
                        Belief[-1].mu, self.belief_0.Sigma)
                    Belief_cor.append(belief_new)

                    if relocation == True:
                        X_cor.append(Belief[-1].mu)

                    if localization == True:
                        X_cor.append(x_new)
                        U_correction = actionsWithPositions(
                            curr_pos_cor, next_pos_cor, action_speed)

                        for j in range(len(U_correction)):
                            x_new = self.SampleMotionModel(
                                X_cor[-1], U_correction[j])
                            belief_new = self.PropagateUpdateBelief(
                                Belief_cor[-1], U_correction[j])

                            X_cor.append(x_new)
                            Belief_cor.append(belief_new)

                    help_happened = True

                if num_of_steps < help_step:
                    X_cor.append(x_new)
                    Belief_cor.append(belief_new)

                if num_of_steps > help_step:
                    x_new_cor = self.SampleMotionModel(X_cor[-1], U[i])
                    belief_new_cor = self.PropagateUpdateBelief(
                        Belief_cor[-1], U[i])
                    X_cor.append(x_new_cor)
                    Belief_cor.append(belief_new_cor)

                num_of_steps += 1

        return X_cor, X


def actionsWithPositions(curr_pos: List[float], next_pos: List[float], action_speed: float):
    actions = []

    deg = np.arctan2(
        (next_pos[1] - curr_pos[1]),
        (next_pos[0] - curr_pos[0]))
    x_speed = action_speed * np.cos(deg)
    y_speed = action_speed * np.sin(deg)
    u = np.matrix([[x_speed], [y_speed]])

    distance = np.sqrt(
        (next_pos[0] - curr_pos[0]) ** 2 +
        (next_pos[1] - curr_pos[1]) ** 2)
    num_of_actions = int(round(distance/action_speed))
    for j in range(num_of_actions):
        actions.append(u)

    return actions


def initializeAction(sol: List[NodePRM]) -> List[List[float]]:
    """Transferring from a general path from a PRM graph,
    to a the discreet location of an agent along a 2D path, 
    depicting the movement of the agent for consecutive time steps.

    Args:
        sol (List[NodePRM]): Nodes on the PRM graph depicting a route.

    Returns:
        List[List[float]]: A list containing the discreet location of an agent along a 2D path, 
        depicting the movement of the agent for consecutive time steps.
    """

    route_points = []

    for i in range(len(sol)):
        point_location = [sol[-i-1].pos.bounds[0],
                          sol[-i-1].pos.bounds[1]]
        route_points.append(point_location)

    return route_points
