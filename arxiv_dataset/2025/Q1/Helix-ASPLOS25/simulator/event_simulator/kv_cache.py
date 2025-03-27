# 2023.12.11 Yixuan Mei

from typing import List, Dict


class KVTracker:
    def __init__(self, total_num_layers: int) -> None:
        """
        Tracking of which node holds the KV cache & activation backup of a request.

        :param total_num_layers: total number of layers in a model
        :return: None
        """
        # basic parameter
        self.total_num_layers: int = total_num_layers

        # layer id -> kv cache locations
        self.layer_kv_locations: Dict[int, List[int]] = {}
        for i in range(total_num_layers):
            self.layer_kv_locations[i] = []

        # layer id -> activation backup locations
        self.layer_activation_backup_locations: Dict[int, List[int]] = {}
        for i in range(total_num_layers):
            self.layer_activation_backup_locations[i] = []

    def add_kv_cache_location(self, layer_id: int, kv_location: int) -> None:
        """
        Set kv cache locations for a layer.

        :param layer_id: layer id
        :param kv_location: a new location of kv cache for that layer
        :return: None
        """
        self.layer_kv_locations[layer_id].append(kv_location)

    def get_kv_cache_locations(self, layer_id: int) -> List[int]:
        """
        Get kv cache locations for a layer

        :param layer_id: layer id
        :return: a list of node uids
        """
        return self.layer_kv_locations[layer_id]

    def add_activation_backup_location(self, layer_id: int, backup_location: int) -> None:
        """
        Set activation backup locations for a layer.

        :param layer_id: layer id
        :param backup_location: a new location of activation cache backup for that layer
        :return: None
        """
        self.layer_activation_backup_locations[layer_id].append(backup_location)

    def get_activation_backup_locations(self, layer_id: int) -> List[int]:
        """
        Get activation backup locations for a layer.

        :param layer_id: layer id
        :return: a list of node uids
        """
        return self.layer_activation_backup_locations[layer_id]


class KVCache:
    def __init__(self, layer_ids: List[int], max_capacity: int) -> None:
        """
        KV cache.
        Note: suppose there are 3 layers, a request of 10 tokens will take up 30 units of capacity.

        :param layer_ids: id of layers on current node
        :param max_capacity: max capacity in number of tokens
        """
        # basic information
        self.layer_ids: List[int] = layer_ids
        self.max_capacity: int = max_capacity

        # kv cache
        # layer id -> {query_id -> num tokens in cache}
        self.available_capacity: int = max_capacity
        self.kv_cache: Dict[int, Dict[int, int]] = {}
        for layer_id in self.layer_ids:
            self.kv_cache[layer_id] = {}

    def initialize_query_kv_cache(self, layers: List[int], query_uid: int, num_tokens: int) -> None:
        """
        Initialize kv cache of a query.

        :param layers: which layers this query uses on this node
        :param query_uid: uid of the query
        :param num_tokens: total number of tokens in this query
        :return: None
        """
        # check whether we can hold
        total_num_tokens: int = num_tokens * len(layers)
        assert self.available_capacity >= total_num_tokens, "Exceed KV-cache capacity!"

        # save the query in kv cache
        self.available_capacity -= total_num_tokens
        for layer_id in layers:
            assert query_uid not in self.kv_cache[layer_id], "Query already initialized!"
            self.kv_cache[layer_id][query_uid] = num_tokens

    def grow_query_kv_cache(self, layers: List[int], query_uid: int) -> None:
        """
        Grow kv cache of a query by 1.

        :param layers: layers to grow the kv cache
        :param query_uid: uid of the query
        :return: None
        """
        # check whether we can hold
        total_increment: int = len(layers)
        assert self.available_capacity >= total_increment, "Exceed KV-cache capacity"

        # save the query in kv cache
        self.available_capacity -= total_increment
        for layer_id in layers:
            assert query_uid in self.kv_cache[layer_id], "Query not initialized!"
            self.kv_cache[layer_id][query_uid] += 1

    def check_kv_cache(self, layers: List[int], query_uid: int, num_prev_tokens: int) -> None:
        """
        Check whether number of tokens in the kv cache for given query is correct. This determines
        whether the request can be executed on a give node.
        Note: If checking fails, an assertion error will be thrown.

        :param layers: layers to check (all layers that will be used in inference of the request on current node)
        :param query_uid: uid of the query
        :param num_prev_tokens: number of previous tokens
        :return: None
        """
        for layer_id in layers:
            assert query_uid in self.kv_cache[layer_id], "No kv cache found for query!"
            assert self.kv_cache[layer_id][query_uid] == num_prev_tokens, "Token count mismatch in kv cache!"

    def exist_duplicate(self, layer_id: int, query_uid: int) -> bool:
        """
        Check whether the given query uid appear in the layer specified or not.

        :param layer_id: id of the layer to check
        :param query_uid: uid of the query
        :return: whether there is a duplicate
        """
        return query_uid in self.kv_cache[layer_id]

    def remove_query_kv_cache(self, layers: List[int], query_uid: int) -> None:
        """
        Remove kv cache of a query. (called when the whole query finishes)

        :param layers: layers to remove cache from
        :param query_uid: uid of the query
        :return: None
        """
        total_freed_space: int = 0
        for layer_id in layers:
            total_freed_space += self.kv_cache[layer_id][query_uid]
            del self.kv_cache[layer_id][query_uid]
        self.available_capacity += total_freed_space


class ActivationBackupCache:
    def __init__(self, layer_ids: List[int], max_capacity: int) -> None:
        """
        Activation backup cache.
        Note: 1. activation backup cache of layer 2 stores the outputs of layer 1 (i.e. activation of layer 2)
              2. not all layers on node will have activation backup, usually the backup is only for the first layer
                 (i.e. backup activations sent to another node during normal execution)

        :param layer_ids: id of layers on current node
        :param max_capacity: max capacity in number of tokens
        :return: None
        """
        # basic information
        self.layer_ids: List[int] = layer_ids
        self.max_capacity: int = max_capacity

        # activation backup cache
        # layer id -> {query_id -> num tokens in cache}
        self.available_capacity: int = max_capacity
        self.activation_backup_cache: Dict[int, Dict[int, int]] = {}
        for layer_id in self.layer_ids:
            self.activation_backup_cache[layer_id] = {}

    def initialize_query_backup(self, layer_id: int, query_uid: int, num_tokens: int) -> None:
        """
        Initialize activation backup for a query.

        :param layer_id: id of the layer that this query is backing up activations for
        :param query_uid: uid of the query
        :param num_tokens: total number of tokens in this query
        :return: None
        """
        # check whether we can hold
        assert self.available_capacity >= num_tokens, "Exceed activation backup capacity!"

        # save the query in activation backup cache
        self.available_capacity -= num_tokens
        assert query_uid not in self.activation_backup_cache[layer_id], "Query already initialized!"
        self.activation_backup_cache[layer_id][query_uid] = num_tokens

    def grow_query_backup(self, layer_id: int, query_uid: int) -> None:
        """
        Grow activation backup of a query by 1.

        :param layer_id: id of the layer to grow the activation backup
        :param query_uid: uid of the query
        :return: None
        """
        # check whether we can hold
        assert self.available_capacity >= 1, "Exceed activation backup capacity"

        # save the query in kv cache
        self.available_capacity -= 1
        assert query_uid in self.activation_backup_cache[layer_id], "Query not initialized!"
        self.activation_backup_cache[layer_id][query_uid] += 1

    def check_activation_backup(self, layer_id: int, query_uid: int, num_prev_tokens: int) -> None:
        """
        Check whether number of tokens in the activation backup for given query is correct.
        Note: If checking fails, an assertion error will be thrown.

        :param layer_id: id of the layer to check
        :param query_uid: uid of the query
        :param num_prev_tokens: number of previous tokens
        :return: None
        """
        assert query_uid in self.activation_backup_cache[layer_id], "No activation backup found for query!"
        assert self.activation_backup_cache[layer_id][query_uid] == num_prev_tokens, "Token count mismatch!"

    def remove_activation_backup(self, layer_id: int, query_uid: int) -> None:
        """
        Remove activation backup of a query. (called when the whole query finishes)

        :param layer_id: id of the layer to remove cache from
        :param query_uid: uid of the query
        :return: None
        """
        self.available_capacity += self.activation_backup_cache[layer_id][query_uid]
        del self.activation_backup_cache[layer_id][query_uid]
