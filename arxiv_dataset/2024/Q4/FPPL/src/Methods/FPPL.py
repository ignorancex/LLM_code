from Methods.meta_method import FederatedMethod


class Method(FederatedMethod):
    def __init__(self, args):
        super().__init__(args)
        self.local_protos = {}
        self.global_protos = {}

    def check_online_clients_list(
        self, task, clients_list, datamanagers_list, online_clients_list
    ):
        self.online_clients_list = online_clients_list
        self.online_clients_list = self.local_model.local_check(
            task=task,
            online_clients_list=self.online_clients_list,
            clients_list=clients_list,
            datamanagers_list=datamanagers_list,
        )
        self.global_net.prompt_fc_tasks = range(task + 1)
        for client in clients_list:
            client.network.prompt_fc_tasks = range(task + 1)

    def local_update(
        self, w_global, task, clients_list, datamanagers_list, global_epoch
    ):
        w_local_list = self.local_model.loc_update(
            w_global=w_global,
            task=task,
            online_clients_list=self.online_clients_list,
            clients_list=clients_list,
            datamanagers_list=datamanagers_list,
            global_protos=self.global_protos,
            local_protos=self.local_protos,
        )
        return w_local_list

    def server_update(
        self, w_local_list, local_task, clients_list, datamanager_global, global_epoch
    ):
        (
            w_global,
            self.global_protos,
        ) = self.server_model.sever_update(
            fed_aggregation=self.fed_aggregation,
            online_clients_list=self.online_clients_list,
            global_net=self.global_net,
            clients_list=clients_list,
            local_protos=self.local_protos,
            w_local_list=w_local_list,
        )
        return w_global
