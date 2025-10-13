import torch


class Evaluator:
    def __init__(self, metrics):
        """ Evaluator
        
        :param metrics: metrics for evaluation (dictionary with keys being metric objects)
        """
        super().__init__()
        self._metrics = metrics
    
    def error_step(self, network, item, device):
        """ Perform a step of an evaluation
        
        :param network: network to use
        :param item: item to use
        :param device: device to use
        """
        
        with torch.no_grad():
            embeddings = item['embeddings'].to(device)
            labels = item['labels'].to(device)

            predictionsA = network(embeddings[:,0].unsqueeze(-1).unsqueeze(-1))
            predictionsA = {k: v[:,:,0,0] for k,v in predictionsA.items()}
            predictionsB = network(embeddings[:,1].unsqueeze(-1).unsqueeze(-1))
            predictionsB = {k: v[:,:,0,0] for k,v in predictionsB.items()}

            for k in self._metrics.keys():
                self._metrics[k](predictionsA, predictionsB, labels)
    
    def evaluate(self, network, data, device, use_loader):
        """ Evaluate 
        
        :param network: network to use
        :param data: data to use
        :param device: device to use
        :param use_loader: if true, assume an iterative loader and perform multiple steps (generally slower)
                           and meant for use when the dataset is too big for VRAM.
        """

        network.eval()
        network.to(device)
        for k in self._metrics.keys():
            self._metrics[k].to(device)
            self._metrics[k].reset()
        
        with torch.no_grad():
            if use_loader:
                for item in data:
                    self.error_step(network, item, device)
            else:
                self.error_step(network, data.get_all(), device)
            results = {k: self._metrics[k].compute().cpu().item() for k in self._metrics.keys()}
        
        return results
