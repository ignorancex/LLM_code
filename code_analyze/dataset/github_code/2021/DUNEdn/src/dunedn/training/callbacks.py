"""This module implements onjects to keep track of training details. """
from pprint import pformat


class Callback:
    def __init__(self):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, logs=None):
        pass

    def on_train_batch_end(self, logs=None):
        pass

    def on_epoch_begin(self, logs=None):
        pass

    def on_epoch_end(self, logs=None):
        pass

    def on_eval_begin(self, logs=None):
        pass

    def on_eval_end(self, logs=None):
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback]):
        self.callback_list = callbacks

    def hook(self, hook_name: str, logs: dict):
        """An utility function to call each callback method.

        Parameters
        ----------
        hook_name: str
            The name of the method to be called.
        logs: dict
            The dictionary to be logged.
        """
        for callback in self.callback_list:
            hook = getattr(callback, hook_name)
            hook(logs)

    def on_train_begin(self, logs=None):
        self.hook("on_train_begin", logs)

    def on_train_end(self, logs=None):
        self.hook("on_train_end", logs)

    def on_train_batch_begin(self, logs=None):
        self.hook("on_train_batch_begin", logs)

    def on_train_batch_end(self, logs=None):
        self.hook("on_train_batch_end", logs)

    def on_epoch_begin(self, logs=None):
        self.hook("on_epoch_begin", logs)

    def on_epoch_end(self, logs=None):
        self.hook("on_epoch_end", logs)

    def on_eval_begin(self, logs=None):
        self.hook("on_eval_begin", logs)

    def on_eval_end(self, logs=None):
        self.hook("on_eval_end", logs)


class History(Callback):
    def __init__(self):
        self.logs = None

    def __repr__(self):
        return pformat(self.logs, indent=2)

    def on_train_begin(self, logs=None):
        self.reset()

    def on_train_batch_end(self, logs=None):
        self.append(logs)

    def on_epoch_end(self, logs=None):
        self.append(logs)

    def append(self, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.logs.setdefault(k, []).append(v)

    def reset(self):
        self.logs = {}
