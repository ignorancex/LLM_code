import signal

from typing_extensions import deprecated
from automata.pda.npda import NPDA


@deprecated("This function is deprecated since NPDA is deprecated.")
def accepts_input_with_timeout(npda: NPDA, e: str, timeout=5):
    """
    This function is used to check if the given NPDA accepts the given input string or not.
    Since current implementation of NPDA will not halt for some inputs due to empty terminal thus not consuming all inputs leading to infinite running,
    we need to set a timeout for it and 5 seconds is a reasonable value and as the default.
    Normally all inputs should be accepted/rejected within 5 seconds.
    :param npda: NPDA
    :param e: example
    :param timeout: time limit
    :return:
    """

    def timeout_handler(signum, frame):
        raise TimeoutError()

    # Set the signal handler and a timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = npda.accepts_input(e)
    except TimeoutError:
        result = False
    finally:
        # Disable the alarm after the function execution
        signal.alarm(0)
    return result
