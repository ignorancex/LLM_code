import os
import subprocess


def get_head_hash(pack):
    """Get the long hash of the commit on HEAD
    Paramters:
        Pack : The package you are using
    Returns: The hash as a binary
     """

    repo_path = os.path.dirname(pack.__file__)
    proc = subprocess.Popen('git -C '+repo_path+' rev-parse HEAD', stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    assert err is None
    return out