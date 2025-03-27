import numpy as np
import torch.nn as nn
import torch.optim as optim
from Locals.meta_local import LocalMethod
from tqdm import tqdm


class Local(LocalMethod):
    def __init__(self, args):
        super().__init__(args)

    def loc_update(
        self,
        w_global,
        task,
        online_clients_list,
        clients_list,
        datamanagers_list,
        global_protos,
        local_protos,
    ):

        w_local_list = []
        for i, client_id in enumerate(online_clients_list):
            print(
                f"Client {client_id} local updating... [{i+1}/{len(online_clients_list)}]"
            )
            w_local, agg_protos = self.local_train(
                w_global,
                task,
                clients_list[client_id],
                datamanagers_list[client_id],
                global_protos,
            )
            w_local_list.append(w_local)
            key = f"{client_id}-{task}"
            local_protos[key] = agg_protos
        return w_local_list

    def local_train(self, w_global, task, client, data_manager, global_protos):
        train_loader = client.before_task(task, data_manager)
        w_local, agg_protos = client.train(w_global, train_loader, global_protos)
        client.after_task(data_manager)
        return w_local, agg_protos
