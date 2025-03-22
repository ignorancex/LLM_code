from itertools import product

import numpy as np


class PwaActionMapper:
    """A class that maps integer labels to sequences of specified PWA regions."""

    @property
    def actions(self):
        return self._actions

    @property
    def num_actions(self):
        return len(self._actions)

    def __init__(self, l: int, N: int) -> None:
        """Initialise the action mapper.

        Parameters
        ----------
        l : int
            The number of PWA regions.
        N : int
            The length of a switching sequence. Corresponds to MPC prediction horizon.
        """
        self.N = N
        self.map = np.asarray(list(product(range(l), repeat=N)), dtype=int)[
            :, :, None
        ]  # all possible switching sequences. Shape: (l^N, N, 1)  # TODO fix type error
        self._actions = [
            np.asarray(item).reshape(len(item), 1)
            for item in product(range(l), repeat=N)
        ]  # each possible switching sequence in a list # TODO is it used??

    def get_label_from_action(self, action: np.ndarray) -> int:
        """Get the integer label of a switching sequence.

        Parameters
        ----------
        action : np.ndarray
            A switching sequence. Shape: (N, 1)

        Returns
        -------
        int
            The integer label of the switching sequence."""
        if action.shape != (self.N, 1):
            raise ValueError(
                f"Invalid action shape. Expected: (N, 1), got: {action.shape}."
            )
        match = (self.map == action).all(axis=(1, 2)).astype(int)
        # match = (
        #     (self.map == torch.from_numpy(action)).all(dim=(1, 2)).to(dtype=int)
        # )  # see if action is in the map # TODO fix type error
        if match.sum() == 0:
            raise ValueError("Invalid action, not in map.")
        if match.sum() > 1:
            raise ValueError("Invalid map, multiple matches found.")
        return match.argmax().item()

    def get_action_from_label(self, label: np.ndarray) -> np.ndarray:
        """Get switching sequences action from integer labels.

        Parameters
        ----------
        label : np.ndarray
            Integer labels. Shape: (batch, 1)

        Returns
        -------
        np.ndarray
            Switching sequences. Shape: (batch, N, 1)
        """
        if (label >= len(self.map)).any():
            raise ValueError(f"Invalid label, out of bound {len(self.map)}.")
        return self.map[label[:, 0].astype(int)]

    def get_valid_actions(self, region: int) -> list[np.ndarray]:
        """Get all valid actions for a given PWA region.

        Parameters
        ----------
        region : int
            The PWA region.

        Returns
        -------
        list[np.ndarray]
            All valid actions for the given PWA region."""
        return [action for action in self._actions if action[0, 0] == region]
