import time
import numpy as np


class Callback(object):
	"""Base class for callbacks to monitor training."""

	def __init__(self, *args, **kwargs):
		"""Class constructor."""
		pass

	def start_of_training(self):
		"""Called before training loop."""
		pass

	def start_of_epoch(self, epoch):
		"""Called at the start of each epoch.

		Args:
			epoch: Type int, number of epochs that have already completed.
		"""
		pass

	def end_of_epoch(self, epoch, batch_logs):
		"""Called at the end of each epoch.
			epochs: Type int, number of epochs that have already completed,
				not including the current one.
			batch_logs: Dict of metrics on the training minibatches of this epoch.
		"""
		pass

	def end_of_training(self):
		"""Called after the training loop."""
		pass


class Training(Callback):
	"""Monitor metrics on training set based on minibatches."""

	def __init__(self):
		super(Training, self).__init__()
		self.history = {"epoch": [], "time": [], "loss": []}

	def start_of_training(self):
		self._initial_epochs = self.history["epoch"][-1] if len(self.history["epoch"]) > 0 else 0
		if isinstance(self.history["epoch"], np.ndarray):
			self.history = {key: list(value) for (key, value) in self.history.items()}

	def start_of_epoch(self, epoch):
		self._initial_time = time.time()

	def end_of_epoch(self, epoch, batch_logs):
		delta_time = time.time() - self._initial_time
		loss = np.mean(np.asarray(batch_logs["loss"]))
		self.history["epoch"].append(self._initial_epochs+epoch+1)
		self.history["time"].append( self.history["time"][-1]+delta_time ) if len(self.history["time"]) > 0 else self.history["time"].append(delta_time)
		self.history["loss"].append(loss)
		print("Epoch {} loss {:.3f}".format(self._initial_epochs+epoch+1, loss))

	def end_of_training(self):
		self.history = {key: np.array(value) for (key, value) in self.history.items()}


class Validation(Callback):
	"""Monitor metrics on validation set."""

	def __init__(self, trainer, val_loader, epoch_interval=10):
		"""Args:
			trainer: trainer.Trainer object, 
				used to access trainer and model methods and parameters.
			val_loader: DataLoader object,
				Validation data loader.
			epoch_interval: Int, 
				Interval between epochs when validation metrics should be calculated.
		"""
		super(Validation, self).__init__()
		self.trainer = trainer
		self.val_loader = val_loader
		self.epoch_interval = epoch_interval
		self.history = {"val_epoch": [], "val_loss": []}

	def end_of_epoch(self, epoch, batch_logs):
		if (epoch+1)%self.epoch_interval != 0:
			pass
		else:
			loss = self.trainer.evaluate(self.val_loader)
			self.history["val_epoch"].append(epoch)
			self.history["val_loss"].append(loss)
			print(". . . val loss {:.3f}".format(loss))

	def end_of_training(self):
		self.history = {key: np.array(value) for (key, value) in self.history.items()}
