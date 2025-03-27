# 2023.12.11 Yixuan Mei

from typing import List, Dict


class LogEntry:
    def __init__(self, log_time: float, entity_name: str, activity: str, description: str, is_empty: bool) -> None:
        """
        A log entry.

        :param log_time: when does the activity happen
        :param entity_name: who writes this log
        :param activity: what happens
        :param description: a detailed description of the activity
        :param is_empty: whether this log entry corresponds to an empty activity (i.e. empty schedule)
        :return: None
        """
        self.log_time: float = log_time
        self.entity_name: str = entity_name
        self.activity: str = activity
        self.description: str = description
        self.is_empty: bool = is_empty

    def print(self) -> None:
        """
        Print this log entry.

        :return: None
        """
        print(f"[t={self.log_time:.3f}] {self.entity_name}: {self.activity} (Description: {self.description})")


class Logger:
    def __init__(self) -> None:
        """
        Centralized logger in the cluster.

        :return: None
        """
        self.last_log_time: float = 0
        self.log_history: List[LogEntry] = []
        self.entity_log_history: Dict[str, List[LogEntry]] = {}

    def add_log(self, log_time: float, entity_name: str, activity: str, description: str,
                is_empty: bool = False) -> None:
        """
        Add an entry into log history.

        :param log_time: when does the activity happen
        :param entity_name: who writes this log
        :param activity: what happens
        :param description: a detailed description of the activity
        :param is_empty: whether this log entry corresponds to an empty activity (i.e. empty schedule)
        :return: None
        """
        # first check log time and create new log entry
        assert log_time >= self.last_log_time, f"Found a log item with wrong time ordering!"
        new_log_entry = LogEntry(log_time=log_time, entity_name=entity_name, activity=activity,
                                 description=description, is_empty=is_empty)

        # append the entry
        self.log_history.append(new_log_entry)
        if entity_name not in self.entity_log_history:
            self.entity_log_history[entity_name] = []
        self.entity_log_history[entity_name].append(new_log_entry)
