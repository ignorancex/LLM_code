import time
import pytorch_lightning as pl

class TimingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.timings = {"train": [], "val": [], "test": [], "predict": []}

    def _time_per_sample(self, start_time, end_time, batch_size):
        return 1000 * (end_time - start_time) / batch_size if batch_size > 0 else 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Start timing for training batches
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Record timing for training batches
        if hasattr(self, 'start_time'):
            batch_size = len(batch[0]) if isinstance(batch, list) else len(batch)
            self.timings['train'].append(self._time_per_sample(self.start_time, time.time(), batch_size))

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Start timing for validation batches
        self.start_time = time.time()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Record timing for validation batches
        if hasattr(self, 'start_time'):
            batch_size = len(batch[0]) if isinstance(batch, list) else len(batch)
            self.timings['val'].append(self._time_per_sample(self.start_time, time.time(), batch_size))

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Start timing for test batches
        self.start_time = time.time()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Record timing for test batches
        if hasattr(self, 'start_time'):
            batch_size = len(batch[0]) if isinstance(batch, list) else len(batch)
            self.timings['test'].append(self._time_per_sample(self.start_time, time.time(), batch_size))

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Start timing for predict batches
        self.start_time = time.time()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Record timing for predict batches
        if hasattr(self, 'start_time'):
            batch_size = len(batch[0]) if isinstance(batch, list) else len(batch)
            self.timings['predict'].append(self._time_per_sample(self.start_time, time.time(), batch_size))
