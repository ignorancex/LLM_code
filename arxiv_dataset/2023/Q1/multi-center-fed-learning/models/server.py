import numpy as np
import random
import copy

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server:
    
    def __init__(self, client_model):
        self.client_model = client_model
        if client_model is not None:
            self.model = client_model.get_params()
        self.selected_clients = []
        self.updates = []

    def select_clients(self, possible_clients, num_clients=20, my_round=100):
        """Selects num_clients clients randomly from possible_clients.
        

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
            
        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round + np.random.randint(10000)) 
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]
    
    def train_model(self, single_center, num_epochs=1, batch_size=10, minibatch=None, clients=None, apply_prox=False):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.
        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in clients:
            if single_center is not None:
                c.model.set_params(single_center)
            else:
                c.model.set_params(self.model)
            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch, apply_prox)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update))

        return sys_metrics, self.updates

    # A variant of aggregating similiar to momentum update
    # The e is a hyperparameter set to 
    # 0.6, but need to find optimal using grid
    # search
    def update_model_dist(self):
        w_t = copy.deepcopy(self.model)
        first = True
        for (client_samples, client_model) in self.updates:
            for i, v in enumerate(client_model):
                if first:
                    w_t[i] = self.model[i] - v.astype(np.float64)
                else:
                    sum_value = self.model[i] - v.astype(np.float64)
                    w_t[i] += sum_value 
            first = False
        effic = 0.6 / len(self.updates)
        updated = [u - v * effic for u,v in zip(self.model, w_t)]
        self.model = updated
        self.updates = []

    # A variant of aggregating similiar to momentum update
    # with a dataset size as weight
    # The e is a hyperparameter set to 
    # 0.02        
    def update_model_distws(self):
        total_weight = 0.
        w_t = copy.deepcopy(self.model)
        first = True
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                if first:
                    w_t[i] = client_sample * (self.model[i] - v.astype(np.float64))
                else:
                    sum_value = client_sample * (self.model[i] - v.astype(np.float64))
                    w_t[i] += sum_value 
            first = False
        effic = 1.0 / (len(self.updates) * total_weight)
        updated = [u - v * effic for u,v in zip(self.model, w_t)]
        self.model = updated
        self.updates = []       
    
    # A variant of update model weighted averaging
    # this one is corresponding to 
    # two terms of our paper "Fedlearn" & "FedDist-WS"
    def update_model_wmode(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []
        return self.model
    
    # A variant of averging using only 1/|number_clients|
    # as weight for aggregation
    def update_model_nowmode(self):
        total_weight = len(self.updates)
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model) in self.updates:
            for i, v in enumerate(client_model):
                base[i] += (1 * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []
        return self.model

    def test_model(self, clients_to_test, set_to_use='test'):
        """
          Test model for different comparison of metrics
          and save them to report file for plotting
        
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.
        Returns info about self.selected_clients if clients=None;
        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples
    
    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        return
        # Save server model
#         self.client_model.set_params(self.model)
#         model_sess =  self.client_model.sess
#         return self.client_model.saver.save(model_sess, path)

    def close_model(self):
#         self.client_model.close()
        return

class SGDServer(Server):
    def __init__(self, client_model):
        super(SGDServer, self).__init__(client_model)
        
    def train_model(self, single_center, batch_size=10, clients=None, apply_prox=False): 
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in clients:
            if single_center is not None:
                c.model.set_params(single_center)
            else:
                c.model.set_params(self.model)
            comp, num_samples, update = c.train(1, batch_size, -1, apply_prox)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update))

        return sys_metrics, self.updates    

class MDLpoisonServer(Server):
    
        def __init__(self, client_model, clients, num_agents, num_classes, agent_scale = 40):
            self.scale = agent_scale
            ids = [c.id for c in clients]
            self.adversaries = random.sample(ids, num_agents)
            for c in clients:
                if c.id in self.adversaries:
                    c.train_data = self._read_adver_agent_data(c.train_data, num_classes)
            super(MDLpoisonServer, self).__init__(client_model)

        def _read_adver_agent_data(self, train_data, num_classes):
            # random some sample then replace the true label
            # TODO the random value is possible to be the true label
            ys = train_data['y']
            max_v = num_classes
            ran_ys = np.random.rand(len(ys)) * max_v
            new_ys = ran_ys.astype(int)
            train_data['y'] = new_ys
            return train_data

  
        def train_model(self, single_center, num_epochs=1, batch_size=10, minibatch=None, clients=None, apply_prox=False):
            if clients is None:
                clients = self.selected_clients
            sys_metrics = {
                c.id: {BYTES_WRITTEN_KEY: 0,
                        BYTES_READ_KEY: 0,
                        LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
            for c in clients:
                if single_center is not None:
                    c.model.set_params(single_center)
                else:
                    c.model.set_params(self.model)
                comp, num_samples, update = c.train(num_epochs, batch_size, minibatch, apply_prox)
                if c.id in self.adversaries:
                    update = [np.multiply(np.array(v), self.scale) for v in update ]

                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

                self.updates.append((num_samples, update))

            return sys_metrics, self.updates
