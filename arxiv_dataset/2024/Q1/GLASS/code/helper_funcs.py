import numpy as np

def cal_utility(n_correct, n_wrong, t):
    """
    Calculates the BCI utility in bits per unit time, assuming a 6x6 keyboard layout.

    Args:
        n_correct (int): Number of correct character selections.
        n_wrong (int): Number of incorrect character selections.
        t (float): Duration of time over which utility is calculated, in any consistent time unit.

    Returns:
        float: BCI utility, representing bits per unit time based on the time unit of `t`.
    """
    n_correct = n_correct - n_wrong
    return np.log(36-1)/np.log(2)*n_correct/t

def cal_t_selection(n_flash, t0 = 3500, t1 = 31.25+125):
    """
    Calculates the total time required for a character selection in minutes.

    Args:
        n_flash (int): Number of flashes required for a character selection.
        t0 (float, optional): Baseline delay before the next character in milliseconds. Defaults to 3500 ms.
        t1 (float, optional): Duration of one flash, including flash time and inter-flash interval, in milliseconds. 
                              Defaults to 156.25 ms (31.25 + 125).

    Returns:
        float: Total selection time in minutes.
    """
    return (n_flash*t1+t0)/1000/60

def evaluate_chr_accu(yhat, y):
    """
    Calculates character-level prediction accuracy by comparing predicted and true target stimuli.

    Args:
        yhat (np.ndarray): Predicted scores for each stimulus, with shape 
                           [# of characters, # of sequences, 2, # of stimuli in a half sequence].
        y (np.ndarray): True binary labels for each stimulus, where 1 indicates target and 0 indicates non-target. 
                        Should have the same shape as `yhat`.

    Returns:
        float: Character-level prediction accuracy.
    """
    correct = yhat.cumsum(axis=1).argmax(axis=3) == y.argmax(axis=3)
    correct = correct.sum(axis=2) == 2
    return np.mean(correct, axis=0)

def evaluate(yhat, y):
    """
    Same as `evaluate_chr_accu` with additional details on which characters are correct.
    """
    correct = yhat.cumsum(axis=1).argmax(axis=3) == y.argmax(axis=3)
    correct = correct.sum(axis=2) == 2
    result = {
        'accuracy': np.mean(correct, axis=0).tolist(),
        'correct': correct.tolist()
    }
    return result