import difflib
from pathlib import Path

from stark_qa.skb import SKB

from llm_bridge import LlmBridge
from logger import Logger, Step3Result
from settings import Settings
from triplet import Triplet, TripletEnd
from vss import VSS


def add_node_to_node_dict(node_dict: dict[str, int|list[int]], node_alias: str, node_id: int) -> None:
    """
    Helper function for create_node_dict_[dataset]. Adds a new node to skb_bridge.nodes_alias2id
    Args:
        node_dict:
        node_alias:
        node_id:
    """
    node_alias = node_alias.lower()
    if node_alias in node_dict:
        if isinstance(node_dict[node_alias], list):
            node_dict[node_alias].append(node_id)
        else:
            node_dict[node_alias] = [node_dict[node_alias], node_id]
    else:
        node_dict[node_alias] = node_id


def create_node_dict_prime(skb: SKB, nodes_alias2id: dict[str, dict[str,int|list[int]]]):
    for i in range(skb.num_nodes()):
        node = skb.node_info[i]
        n_type = node["type"]
        n_name = node['name']
        add_node_to_node_dict(nodes_alias2id[n_type], n_name, i)

        if 'details' in node:
            if 'alias' in node['details']:
                alias = node['details']['alias']
                if isinstance(alias, list):
                    for a in alias:
                        add_node_to_node_dict(nodes_alias2id[n_type], a, i)
                else:
                    add_node_to_node_dict(nodes_alias2id[n_type], alias, i)
    return nodes_alias2id


def create_node_dict_mag(skb: SKB, nodes_alias2id: dict[str, dict[str,int|list[int]]]):
    for i in range(skb.num_nodes()):
        node = skb.node_info[i]
        n_type = node["type"]
        if 'title' in node:
            add_node_to_node_dict(nodes_alias2id[n_type], node['title'], i)
        elif 'DisplayName' in node and node['DisplayName'] != -1 and node['DisplayName'] != "-1":
            add_node_to_node_dict(nodes_alias2id[n_type], node['DisplayName'], i)
    return nodes_alias2id


def create_node_dict_amazon(skb: SKB, nodes_alias2id: dict[str, dict[str,int|list[int]]]):
    for i in range(skb.num_nodes()):
        node = skb.node_info[i]
        if 'title' in node:
            n_type = "product"
            name = node['title']
        elif 'brand_name' in node:
            n_type = "brand"
            name = node['brand_name']
        elif 'category_name' in node:
            n_type = "category"
            name = node['category_name']
        else:
            n_type = "color"
            name = node['color_name']

        add_node_to_node_dict(nodes_alias2id[n_type], name, i)
    return nodes_alias2id


def create_node_dict(skb: SKB, dataset: str):
    """
    Args:
        skb: semi-structured knowledge base
        dataset: name of dataset and semi-structured knowledge base

    Returns:
        a dictionary assigning node IDs to node aliases for each node type
        a dictionary assigning node IDs to node aliases for every node type
    """
    nodes_alias2id = {}
    for n_type in skb.node_type_lst():
        nodes_alias2id[n_type] = {}

    if dataset == 'prime':
        nodes_alias2id = create_node_dict_prime(skb, nodes_alias2id)
    elif dataset == 'mag':
        nodes_alias2id = create_node_dict_mag(skb, nodes_alias2id)
    elif dataset == 'amazon':
        nodes_alias2id = create_node_dict_amazon(skb, nodes_alias2id)
    else:
        raise ValueError(f"dataset name should be in ['prime', 'mag,', 'amazon'], but '{dataset}' is given")

    nodes_alias2id_unknown_type = {}
    for n_type in skb.node_type_lst():
        for n_alias, node_ids in nodes_alias2id[n_type].items():
            if isinstance(node_ids, list):
                for node_id in node_ids:
                    add_node_to_node_dict(nodes_alias2id_unknown_type, n_alias, node_id)
            else:
                add_node_to_node_dict(nodes_alias2id_unknown_type, n_alias, node_ids)
    return nodes_alias2id, nodes_alias2id_unknown_type





class SKBbridge:
    def __init__(self, settings: Settings, data_split: str, llm_bridge: LlmBridge,
                 skb: SKB = None, emb_dir: Path = None):
        name = settings.dataset_name
        if name not in ['prime', 'mag', 'amazon']:
            raise ValueError(f"Dataset {name} not found. It should be in ['prime', 'mag,', 'amazon']")

        self.settings = settings
        self.skb = skb
        self.llm_bridge = llm_bridge

        self.nodes_alias2id, self.nodes_alias2id_unknown_type = create_node_dict(self.skb, name)

        if name == 'prime' or name == 'mag' or name == 'amazon':
            self.is_directed = False
        else:
            self.is_directed = False

        self.node_ids_by_type = {}
        for n_type in skb.node_type_lst():
            self.node_ids_by_type[n_type] = skb.get_node_ids_by_type(n_type)

        if emb_dir is None:
            self.vss = None
        else:
            self.vss = VSS(skb, emb_dir, data_split, settings.get("emb_model"), self.node_ids_by_type,
                           settings.get("offline_mode"))

    def expected_answers(self, answer_ids: list[int], separator: str = ", ") -> str:
        out = ""
        if self.settings.dataset_name == 'prime':
            out += separator.join([self.skb[aid].name for aid in answer_ids])
        elif self.settings.dataset_name == 'mag':
            for aid in answer_ids:
                if "title" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['title'] + separator
                elif "DisplayName" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['DisplayName'] + separator
                else:
                    out += f"Answer has no name (!): {self.skb.node_info[aid]}{separator}"
        elif self.settings.dataset_name == 'amazon':
            for aid in answer_ids:
                if "title" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['title'] + separator
                elif "brand_name" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['brand_name'] + separator
                elif "category_name" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['category_name'] + separator
                elif "color_name" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['color_name'] + separator
                else:
                    out += f"Answer has no name (!): {self.skb.node_info[aid]}{separator}"
        else:
            raise NotImplementedError("unknown dataset")
        return out

    def find_closest_nodes_w_cutoff(self, target_name: str, target_type: str = None,
                           logger: Logger = None, enable_vss: bool = False, cutoff_vss: float = None, llm_activation: bool = False,
                           query_for_llm_activation: str = None, step3_result:Step3Result=None) -> list[int]:
        """
        Searches for nodes of a given or unknown type with an equal or similar alias to the target name
        Args:
            target_name: search string
            target_type: node type
            k: maximum number of node IDs to return
            m: maximum number of entities to be used for llm activation
            enable_difflib: enable diff_lib string similarity search
            enable_vss: use vss instead of diff_lib to find close nodes
            cutoff_difflib: cutoff value between 0 and 1 used for difflib string simility search
            cutoff_vss: cutoff value between 0 and 1 used for VSS

        Returns:
            list of pairs of node ID and similarity score to target_name in descending order of similarity
        """
        if target_type is None:
            node_dict = self.nodes_alias2id_unknown_type
        else:
            node_dict = self.nodes_alias2id[target_type]

        nodes_found = []
        nodes_direct_match = []
        num_found_key_match = 0

        if self.settings.get("k") > 0:
            k = self.settings.get("k")
        else:
            if enable_vss and cutoff_vss is None:
                raise ValueError(
                    "Invalid combination of limit_to_k_answers=False, enable_vss=True and cutoff_vss=None.")
            k = len(node_dict)

        # search for direct key matches in node list
        if target_name.lower() in node_dict:
            node_ids = node_dict[target_name.lower()]
            if isinstance(node_ids, list):
                nodes_direct_match = node_ids
            else:
                nodes_direct_match = [node_ids]
            nodes_found = nodes_direct_match
        num_found_key_match += len(nodes_found)
        print(f"Nodes with matching alias for {target_name.lower()} directly found in database: {nodes_found}.")
        if num_found_key_match > 0 and step3_result is not None:
            step3_result.num_key_matches += 1
            step3_result.num_key_matches_candidates += num_found_key_match

        # VSS, if it is enabled. And if there are not enough direct matches found already or llm_activation is enabled
        if enable_vss and (llm_activation or len(nodes_found) < k):
            if step3_result is not None:
                step3_result.num_constants_w_vss += 1
            node_types_to_consider = self.settings.get("nodes_to_consider")
            vss_nodes_found, vss_scores = self.vss.get_top_k_nodes(search_str=target_name, k=k, node_type=target_type,
                                                       node_id_mask=None, cutoff=cutoff_vss,
                                                       node_types_to_consider=node_types_to_consider,
                                                       query_id=None)
            nodes_found = [x for x in vss_nodes_found if x not in nodes_direct_match]
            nodes_found = nodes_direct_match + nodes_found
            nodes_found = nodes_found[:k]

            if step3_result is not None:
                step3_result.num_vss_candidates += len(nodes_found) - num_found_key_match

            if llm_activation and len(nodes_found) > 1:
                    if step3_result is not None:
                        step3_result.num_constants_w_llm_activation += 1
                    try:
                        node_descriptions = ""
                        for idx, node_id in enumerate(nodes_found):
                            node_descriptions += f"ENTITY_ID: {idx}, {self.skb.get_doc_info(node_id, add_rel=False, compact=False)}\n\n"
                        nodes_found = self.cull_nodes(nodes_found, query_for_llm_activation,
                                                                 target_name, node_descriptions, logger)
                    except RuntimeError as e:
                        node_descriptions = ""
                        for idx, node_id in enumerate(nodes_found):
                            node_descriptions += f"ENTITY_ID: {idx}, {self.skb.get_doc_info(node_id, add_rel=False, compact=True)}\n\n"

                        splits = 1
                        while splits < len(nodes_found):
                            try:
                                nodes_found = []
                                for i in range(splits):
                                    new_nodes_found = self.cull_nodes(nodes_found[i::splits], query_for_llm_activation,
                                                                     target_name, node_descriptions, logger)
                                    nodes_found.extend(new_nodes_found)
                                break
                            except RuntimeError as e:
                                splits *= 2

                    logger.log(f"Remaining nodes after LLM activation: {nodes_found}.")
            if step3_result is not None:
                step3_result.num_w_llm_activation_candidates += len(nodes_found)

        return nodes_found

    def cull_nodes(self, node_ids: list, query: str, target_name: str, node_descriptions: str,
                       logger: Logger) -> list[int]:
            prompt = self.settings.prompts["llm_activation"]
            prompt = prompt.replace('{type_of_kb}', self.settings.get("type_of_kb"))
            prompt = prompt.replace('{query}', query)
            prompt = prompt.replace('{target_name}', target_name)
            prompt = prompt.replace('{node_descriptions}', node_descriptions)
            [answer], _ = self.llm_bridge.ask_llm_batch([prompt])
            try:
                node_ids = [node_ids[int(num.strip())] for num in answer.split(',')]
            except ValueError:
                logger.log(f"Invalid answer, can't convert it to list of integers: {answer=}. "
                           f"Continuing with all nodes found, ignoring failed llm activation.")
            except KeyError:
                logger.log(f"Invalid answer, index of answer is not valid: {answer=}. "
                           f"Continuing with all nodes found, ignoring failed llm activation.")
            except IndexError:
                logger.log(f"Invalid answer, index of answer is not valid: {answer=}. "
                           f"Continuing with all nodes found, ignoring failed llm activation.")
            return node_ids

    def entity_ids2name(self, ids: list[int] | set[int], n=float("inf")) -> str:
        n = int(min(n, len(ids)))

        if not isinstance(ids, list):
            ids = list(ids)
        out = ", ".join([self.entity_id2name(idx) for idx in ids[:n]])
        if len(ids) > n:
            out += ", ..."
        return out

    def entity_id2name(self, idx: int):
        if self.settings.dataset_name == 'prime':
            return self.skb.node_info[idx]['name']
        if self.settings.dataset_name == 'mag':
            node = self.skb.node_info[idx]
            if 'title' in node:
                return node['title']
            elif 'DisplayName' in node and node['DisplayName'] != -1 and node['DisplayName'] != "-1":
                return node['DisplayName']
            else:
                return f"node without name. id: {idx}"
        elif self.settings.dataset_name == 'amazon':
            node = self.skb.node_info[idx]
            if 'title' in node:
                return node['title']
            elif 'brand_name' in node:
                return node['brand_name']
            elif "category_name" in node:
                return node['category_name']
            elif "color_name" in node:
                return node['color_name']

        raise NotImplementedError(f"Not implemented for dataset {self.settings.dataset_name}")

    def nodes2str(self, node_ids: int | list[str]) -> str:
        if isinstance(node_ids, list) or isinstance(node_ids, set):
            out = []
            for node_id in node_ids:
                out.append(self.skb.get_doc_info(node_id, add_rel=False, compact=False))
            return out
        else:
            return self.skb.get_doc_info(node_ids, add_rel=False, compact=False)

    def ground_triplets(self, triplets: list[Triplet], atomics: dict[str, TripletEnd], logger: Logger,
                        target_variable: TripletEnd, ignore_edge_labels: bool) -> {}:
        skb = self.skb

        earlier_sum_of_all_candidates = -1
        new_sum_of_all_candidates = 0

        while new_sum_of_all_candidates != earlier_sum_of_all_candidates:
            earlier_sum_of_all_candidates = new_sum_of_all_candidates
            new_sum_of_all_candidates = 0
            for triplet in triplets:
                h = triplet.h
                if ignore_edge_labels:
                    e = "*"
                else:
                    e = triplet.e

                t = triplet.t

                if h.candidates is None and t.candidates is None:
                    continue

                if h.candidates is None:
                    if h.node_type in self.node_ids_by_type:
                        h.candidates = self.node_ids_by_type[h.node_type].copy()
                    else:
                        h.candidates = list(self.skb.node_info.keys()).copy()
                if t.candidates is None:
                    if t.node_type in self.node_ids_by_type:
                        t.candidates = self.node_ids_by_type[t.node_type].copy()
                    else:
                        t.candidates = list(self.skb.node_info.keys()).copy()

                h_before = len(h.candidates)
                t_before = len(t.candidates)

                if len(h.candidates) < 10000:
                    candidates = set()
                    for h_candidate in h.candidates:
                        candidates = candidates.union(set(skb.get_neighbor_nodes(h_candidate, e)))
                    t.intersection_update(candidates)
                    logger.log(
                        f"Found {len(candidates)} candidates for head {h.get_uid()} ({h.node_type}).")
                else:
                    logger.log(f"{triplet=}: Too many candidates ({len(h.candidates)}) for triplet head to search "
                               f"for all their neighbors.")
                if len(t.candidates) < 10000:
                    candidates = set()
                    for t_candidate in t.candidates:
                        candidates = candidates.union(set(skb.get_neighbor_nodes(t_candidate, e)))
                    h.intersection_update(candidates)
                    logger.log(
                        f"Found {len(candidates)} candidates for tail {t.get_uid()} ({t.node_type}).")
                else:
                    logger.log(f"{triplet=}: Too many candidates ({len(t.candidates)}) for triplet tail to search "
                               f"for all their neighbors.")
                logger.log(
                    f"Grounded triplet: {h.get_uid()} ({h.node_type}) [{len(h.candidates)}/{h_before} candidates]-> "
                    f"{e} -> {t.get_uid()} ({t.node_type}) [{len(t.candidates)}/{t_before} candidates].")
            for atomic in atomics.values():
                if atomic.candidates is not None:
                    new_sum_of_all_candidates += len(atomic.candidates)
            if target_variable.candidates is not None and len(target_variable.candidates) == 0:
                logger.log("No candidates for target variable.")
                break
        return atomics
