# EarlyStopping class
class EarlyStopping:

    def __init__(self, patience, percentage_threshold=None, warmup_epochs=0):
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.percentage_threshold = percentage_threshold
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.warmup_done = False

    # early stopping using warm-up and relative decrease
    def should_stop_1(self, current_loss):

        if not self.warmup_done:
            self.warmup_epochs -= 1
            if self.warmup_epochs <= 0:
                self.warmup_done = True
            return False
        else:
            relative_loss_decrease = self.best_loss - current_loss
            if relative_loss_decrease <= self.percentage_threshold:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    return True
                return False
            else:
                self.epochs_without_improvement = 0
                self.best_loss = current_loss
                return False

    # early stopping saving the best models using patience of 20 epochs
    def should_save_checkpoint(self, current_loss):

        if current_loss < self.best_loss and self.epochs_without_improvement < self.patience:
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def should_stop_training(self):
        if self.epochs_without_improvement > self.patience:
            return True
        else:
            return False


if __name__ == "__main__":
    early_stopping = EarlyStopping(10, 0.05, 4)
    early = early_stopping.should_stop(5)
    print(early)
    print(early)
    print(early)
    print(early)
    print(early)
    print(early)
    print(early)
    print(early)
