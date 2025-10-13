class DynamicLossWeight:
    def __init__(self, initial_weights, adjustment_factor=1.1, max_weight=1.5):
        self.loss_weights = initial_weights
        self.adjustment_factor = adjustment_factor
        self.max_weight = max_weight

    def update_weights(self, task_accuracies):
        threshold = sum(task_accuracies.values()) / len(task_accuracies)
        for task, acc in task_accuracies.items():
            if acc < threshold:
                self.loss_weights[task] = min(self.loss_weights[task] * self.adjustment_factor, self.max_weight)
            else:
                # Decay weight if accuracy converges
                self.loss_weights[task] /= 1.05