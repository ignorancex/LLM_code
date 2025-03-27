# 2023.12.11 Yixuan Mei

from enum import Enum
from typing import Any, Dict


class EventHandler(Enum):
    """
    Here we list all event handler names.
    """
    # --------------------- External -------------------- #
    CommandNewRequest = "EventHandler.Command.NewRequest"
    CommandLoadModel = "EventHandler.Command.LoadModel"
    # --------------------- Internal -------------------- #
    StartTransmission = "EventHandler.StartTransmission"
    FinishSending = "EventHandler.FinishSending"
    FinishTransmission = "EventHandler.FinishTransmission"
    GatherFinished = "EventHandler.GatherFinished"
    StartExecution = "EventHandler.StartExecution"
    FinishExecution = "EventHandler.FinishExecution"
    StartLoadingModel = "EventHandler.StartLoadingModel"
    FinishLoadingModel = "EventHandler.FinishLoadingModel"
    # --------------------- Unknown --------------------- #
    Unknown = "EventHandler.Unknown"


class EventDescription:
    def __init__(self, who: str, at_when: float, does_what: str) -> None:
        """
        Record who does what at when and trigger this event / who needs to do what at when.

        :param who: who
        :param at_when: at when
        :param does_what: does what / should do what
        """
        self.who: str = who
        self.at_when: float = at_when
        self.does_what: str = does_what


class Event:
    def __init__(self, event_uid: int, event_time: float, event_handler: EventHandler, args: Dict[str, Any],
                 background: EventDescription, description: EventDescription):
        """
        An event which abstracts anything that could happen in the cluster.

        :param event_uid: unique identifier of this event
        :param event_time: when this event will happen
        :param event_handler: name of the event handler
        :param args: arguments needed by the event handler
        :param background: background information (who did what at when and caused this event)
        :param description: description of this event (who will do what at when)
        """
        # basic event information
        self.event_uid: int = event_uid
        self.event_time: float = event_time
        self.event_handler: EventHandler = event_handler
        self.args: Dict[str, Any] = args

        # some more logging info
        self.background: EventDescription = background
        self.description: EventDescription = description

    def __lt__(self, other: "Event"):
        return self.event_uid < other.event_uid
