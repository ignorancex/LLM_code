from enum import IntEnum, auto


class OracleType(IntEnum):
    UNKNOWN = auto()  # 01
    SYNTAX_ERROR = auto()  # 02
    BASE_EXCEPTION = auto()  # 03
    BASE_SUCCESS = auto()  # 04
    TRANSFER_EXCEPTION = auto()  # 05
    TARGET_EXCEPTION = auto()  # 06
    TARGET_SUCCESS = auto()  # 07
    MISALIGN = auto()  # 08
    INCON = auto()  # 09
    SUCCESS_WITH_NEWCOV = auto()  # 10
    SUCCESS_WITHOUT_NEWCOV = auto()  # 11
    SIGKILL = auto()  # 12


# In the feedback.py, we also define a feedback_code to record these situations!
"""
01:Initial unknown state
02:Syntax error in code
03:Eager failed
04:Eager succeed
05:backend transfer failed
06:Compiler failed
07:Compiler succeed
08:Compiler succeed,but Eager failed
09:Compiler succeed,but value inconsistent
10:Compiler succeed,and value consistent,but has new coverage
11:Compiler succeed,but value consistent,and no new coverage

12: Process is killed by system
"""
