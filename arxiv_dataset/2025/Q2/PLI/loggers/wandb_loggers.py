import wandb

from loggers.abc import AbstractBaseLogger


class WandbSimplePrinter(AbstractBaseLogger):
    def __init__(self, prefix):
        self.prefix = prefix

    def log(self, log_data, step, commit=False):
        log_metrics = {self.prefix + k: v for k, v in log_data.items() if not isinstance(v, dict)}
        wandb.log(log_metrics, step=step, commit=commit)

    def complete(self, log_data, step):
        self.log(log_data, step)


class WandbBestPrinter(AbstractBaseLogger):
    """
    TODO: Summary metrics describe the best model performance, and will be shown in the wandb summary table.
    """
    def __init__(self, summary_prefix, metric_key='10', metric_prefix='recall_@'):
        """
        :param summary_prefix: the prefix of the summary metrics, e.g. 'best_model/'
        :param metric_key: the key of the metric, e.g. '10' or '10,50'
        :param metric_prefix: the prefix of the metric, e.g. 'recall_@', which is matched with log_data keys
        """
        self.summary_prefix = summary_prefix
        self.metric_key = [metric_key] if isinstance(metric_key, str) else list(metric_key.split(','))
        self.metric_prefix = metric_prefix
        self.previous_best_vals = -9e9

    def log(self, log_data, step, commit=False):
        """
        Calculate the average value of the metric_key and update the summary table
        :param log_data: all_val_results, which key is the prefix_key + summary_key
        :param step: the epoch or step, default is epoch
        """
        # TODO: abstract the following code to a function, which is same as the one in BestModelTracker
        recent_values = 0
        num_values = 0
        for log_key in log_data:
            for m_key in self.metric_key:
                key = self.metric_prefix + m_key
                if key not in log_key:
                    continue
                recent_values += log_data[log_key]
                num_values += 1
        assert num_values > 0, "The key: {} is not in logged data. Not saving best model".format(self.metric_key)
        # calculate the average value of the metric_key
        recent_value = recent_values / num_values
        if recent_value > self.previous_best_vals:
            self.previous_best_vals = recent_value
            # update the summary table, sumaary_key e.g. 'best_model/recall_@10' or 'best_model/avg_recall_@10,50'
            middle_key = "" if len(self.metric_key) == 1 else "avg_"
            summary_key = self.summary_prefix + middle_key + self.metric_prefix + ','.join(self.metric_key)
            wandb.run.summary[summary_key] = recent_value
            # summary other metrics of the best model
            wandb.run.summary.update({self.summary_prefix + k: v for k, v in log_data.items() if self.metric_prefix in k})

    def complete(self, log_data, step):
        self.log(log_data, step)

class WandbSummaryPrinter(AbstractBaseLogger):
    def __init__(self, prefix, summary_keys: list, prefix_keys: list):
        self.prefix = prefix
        self.summary_keys = [prefix + '_' + key for key in summary_keys for prefix in prefix_keys]
        self.previous_best_vals = {key: 0 for key in self.summary_keys}

    def log(self, log_data, step, commit=False):
        for key in self.summary_keys:
            if key in log_data:
                log_value = log_data[key]
                if log_value > self.previous_best_vals[key]:
                    wandb.run.summary[self.prefix+key] = log_value
                    self.previous_best_vals[key] = log_value

    def complete(self, log_data, step):
        self.log(log_data, step)

if __name__ == "__main__":
    # Update metrics for a run, after the run has finished
    import wandb

    api = wandb.Api()

    run = api.run("junyang/MGUR/98x8xgto")
    print(run.summary)
    # compute average of shirt, dress and toptee metric recall_@1,5,10,50
    run.summary["best_model/avg_fashionIQ_recall_@1"] = (run.summary["best_model/fashionIQ_shirt_recall_@1"] + run.summary["best_model/fashionIQ_dress_recall_@1"] + run.summary["best_model/fashionIQ_toptee_recall_@1"]) / 3
    run.summary["best_model/avg_fashionIQ_recall_@5"] = (run.summary["best_model/fashionIQ_shirt_recall_@5"] + run.summary["best_model/fashionIQ_dress_recall_@5"] + run.summary["best_model/fashionIQ_toptee_recall_@5"]) / 3
    run.summary["best_model/avg_fashionIQ_recall_@10"] = (run.summary["best_model/fashionIQ_shirt_recall_@10"] + run.summary["best_model/fashionIQ_dress_recall_@10"] + run.summary["best_model/fashionIQ_toptee_recall_@10"]) / 3
    run.summary["best_model/avg_fashionIQ_recall_@50"] = (run.summary["best_model/fashionIQ_shirt_recall_@50"] + run.summary["best_model/fashionIQ_dress_recall_@50"] + run.summary["best_model/fashionIQ_toptee_recall_@50"]) / 3
    print("recall_@1: ", run.summary["best_model/avg_fashionIQ_recall_@1"])
    print("recall_@1: ", round(run.summary["best_model/avg_fashionIQ_recall_@1"] * 100, 2))
    print("recall_@5: ", run.summary["best_model/avg_fashionIQ_recall_@5"])
    print('recall_@5', round(run.summary["best_model/avg_fashionIQ_recall_@5"] * 100, 2))
    print("recall_@10: ", run.summary["best_model/avg_fashionIQ_recall_@10"])
    print("recall_@10: ", round(run.summary["best_model/avg_fashionIQ_recall_@10"] * 100, 2))
    print("recall_@50: ", run.summary["best_model/avg_fashionIQ_recall_@50"])
    print("recall_@50: ", round(run.summary["best_model/avg_fashionIQ_recall_@50"] * 100, 2))
    run.summary.update()