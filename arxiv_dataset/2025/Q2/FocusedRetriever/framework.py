from pathlib import Path
import shutil

from stark_qa import load_skb

from llm_reranker import LLAMAReranker
from stark_main_offline.load_qa import load_qa
from stark_qa.skb import SKB

from llm_bridge import LlmBridge
from logger import Logger, Step2aResult, Step3Result, Step2bResult, Step4bResult, Step8Result, Step7Result, \
    Step4aResult, Step5Incl7Result, Step6Result
from skb_bridge import SKBbridge
from settings import Settings
from triplet import TripletEnd, Triplet


class Framework:
    def __init__(self, experiment_name: str, dataset_name: str, data_split: str, llm_model: str = None, skb: SKB = None,
                 enable_vss: bool = True, emb_model: str = None, configs_path: str = None):
        # plausibility checks
        valid_dataset_names = ['prime', 'mag', 'amazon']
        if dataset_name not in valid_dataset_names:
            raise ValueError(f"Dataset {dataset_name} not found. It should be in {valid_dataset_names}")

        valid_data_splits = ["train", "val", "test", "human_generated_eval"]
        if data_split not in valid_data_splits:
            raise ValueError(f"Data split {data_split} not found. "
                             f"It should be in {valid_data_splits}")

        # load settings
        self.settings = Settings(dataset_name, llm_model=llm_model, emb_model=emb_model, configs_path=configs_path)

        llm_model = self.settings.get("llm_model")
        configs = self.settings.configs

        # load logger
        if experiment_name is None:
            experiment_name = configs.get("experiment_name")
        self.logger = Logger(Path(__file__).parent / configs["output_path"] / dataset_name / data_split / llm_model / experiment_name)
        # copy config file to results file
        shutil.copy(configs_path, Path(__file__).parent / configs[
            "output_path"] / dataset_name / data_split / llm_model / experiment_name)

        # load llm bridge
        self.llm_bridge = LlmBridge(llm_model, configs_path, self.logger)

        # load SKB bridge including SKB, embeddings, and embedding client
        if skb is None:
            skb = load_skb(name=dataset_name, download_processed=True, root=self.settings.get("skb_path"))
        emb_dir = None
        if enable_vss:
            emb_dir = Path(__file__).parent / configs["embeddings_path"] / dataset_name
        self.skb_b = SKBbridge(settings=self.settings, data_split=data_split, llm_bridge=self.llm_bridge,
                               skb=skb, emb_dir=emb_dir)

        # load data
        self.eval_data = load_qa(name=dataset_name, human_generated_eval=data_split=="human_generated_eval",
                                 root=self.settings.get("qa_path"))


        #if data_split != "human_generated_eval":
        #    self.eval_data = self.eval_data.get_subset(data_split)
        # load reranker
        self.reranker = LLAMAReranker(skb, self.llm_bridge, self.skb_b.vss, self.settings, self.logger)

        print(f"Size of test dataset: {len(self.eval_data)} QA-pairs")


    def step6_get_target_type(self, question: str) -> str:
        """
        Prompts an LLM to return the type of entities that are searched for in a given question.
        :param question: question in natural language
        :return: Type of searched entity or entities
        """

        candidate_types = self.skb_b.skb.candidate_types

        # If only one candidate type is available return it immediately and skip the prompting.
        if len(candidate_types) == 1:
            return candidate_types[0]
        else:
            llm_query = self.settings.prompts['ask_for_target_type']
            llm_query = llm_query.replace("{query}", question)
            llm_query = llm_query.replace("{type_of_kb}", self.settings.get("type_of_kb"))
            llm_query = llm_query.replace("{candidate_types}", f"{candidate_types}")

            [target_type], _ = self.llm_bridge.ask_llm_batch([llm_query])
            target_type = target_type.strip("'\" \n[]")
            return target_type

    def save_and_validate_step2a(self, target_type: str, ground_truths: list[int]) -> Step2aResult:
        candidate_types = self.skb_b.skb.node_type_lst() # self.skb_b.skb.candidate_types
        ground_truth_node_type = self.skb_b.skb.get_node_type_by_id(ground_truths[0])

        is_invalid = False
        is_incorrect = False

        if target_type not in candidate_types:
            is_invalid = 1
            self.logger.log(f"Target type {target_type} is not a valid target type."
                            f"It must be in {candidate_types}.")
        elif target_type != ground_truth_node_type:
            is_incorrect = 1

        r = Step2aResult(target_type, is_invalid, is_incorrect, ground_truth_node_type)
        return r

    def save_and_validate_step6(self, target_type: str, ground_truths: list[int]) -> Step6Result:
        candidate_types = self.skb_b.skb.node_type_lst() # self.skb_b.skb.candidate_types
        ground_truth_node_type = self.skb_b.skb.get_node_type_by_id(ground_truths[0])

        is_invalid = False
        is_incorrect = False

        if target_type not in candidate_types:
            is_invalid = 1
            self.logger.log(f"Target type {target_type} is not a valid target type."
                            f"It must be in {candidate_types}.")
        elif target_type != ground_truth_node_type:
            is_incorrect = 1

        r = Step6Result(target_type, is_invalid, is_incorrect, ground_truth_node_type)
        return r

    def step2_get_constants(self,query: str, target_type: str) -> Step2bResult:
        settings = self.settings
        logger = self.logger

        # prompting:
        llm_query = (settings.prompts['ask_for_constants'] + settings.prompts['ask_for_further_symbol_terms_example'])
        llm_query = llm_query.replace("{query}", query)
        llm_query = llm_query.replace("{type_of_kb}", settings.get("type_of_kb"))
        llm_query = llm_query.replace("{nodes_types}", str(self.skb_b.skb.node_type_lst()))
        llm_query = llm_query.replace("{target_type}", f"{target_type}")

        [symbols], _ = self.llm_bridge.ask_llm_batch([llm_query])
        symbols = symbols.split("|")

        # parsing:
        valid_symbols = {}
        num_invalid_constants = 0

        for term in symbols:
            term = term.split("::")
            if len(term) == 2:
                name = term[0].strip("'\" ")
                n_type = term[1].strip("'\" ")
                if n_type in self.skb_b.skb.node_type_lst():
                    if n_type in self.settings.get("nodes_to_ignore"):
                        logger.log(f"Ignoring {term} because node type is flagged to be ignored for entity search {n_type}.")
                    else:
                        triplet_end = TripletEnd(name, n_type, is_constant=True)
                        valid_symbols[triplet_end.get_uid()] = triplet_end
                else:
                    logger.log(f"Node type {n_type} not found in knowledge base. Skipping constant.")
                    num_invalid_constants += 1
            else:
                logger.log(f"invalid constant structure. It needs to be a name followed by two colons and a node type, but is: '{term}'")
                num_invalid_constants += 1
        return Step2bResult(valid_symbols, num_invalid_constants)


    def step3_entity_search(self, valid_symbols: dict[str, TripletEnd], query: str, target_var: TripletEnd,
                            ignore_labels: bool) -> Step3Result:
        r = Step3Result()

        invalid_symbols = []

        for symbol in valid_symbols.values():
            candidates = set()
            for property_name in symbol.properties:
                property_val = symbol.properties[property_name]
                if property_name in self.settings.configs.get("avail_node_properties").keys():
                    property_name = self.settings.configs.get("avail_node_properties")[property_name]
                if property_name != "title" and property_name != "name":
                    if property_val[0] == "<" or property_val[0] == ">":
                        new_candidates = []
                        if symbol.node_type is None:
                            node_ids = self.skb_b.nodes_alias2id_unknown_type
                        else:
                            node_ids = self.skb_b.node_ids_by_type[symbol.node_type]
                        for c in node_ids:
                            if property_val[0] == "<":
                                if isinstance(self.skb_b.skb.node_info[c][property_name], int):
                                    if self.skb_b.skb.node_info[c][property_name] < int(property_val[1:5]):
                                        new_candidates.append(c)
                                elif self.skb_b.skb.node_info[c][property_name] < property_val[1:]:
                                    new_candidates.append(c)
                            if property_val[0] == ">":
                                if isinstance(self.skb_b.skb.node_info[c][property_name], int):
                                    if self.skb_b.skb.node_info[c][property_name] > int(property_val[1:5]):
                                        new_candidates.append(c)
                                elif self.skb_b.skb.node_info[c][property_name] > property_val[1:]:
                                    new_candidates.append(c)
                    else:
                        new_candidates = self.skb_b.skb.get_node_ids_by_value(
                            symbol.node_type, property_name, property_val)
                        try:
                            new_candidates += self.skb_b.skb.get_node_ids_by_value(
                                symbol.node_type, property_name, int(property_val))
                        except ValueError:
                            pass
                    new_candidates = set(new_candidates)

                    if len(new_candidates) > 0:
                        if len(candidates) == 0:
                            candidates = new_candidates
                        else:
                            candidates.intersection_update(new_candidates)
                    self.logger.log(f"Number of nodes with matching alias for {property_name} found in database: {len(candidates)}.")

            if "title" in symbol.properties:
                target_name = symbol.properties["title"]   # OLD: symbol.name
            elif "name" in symbol.properties:
                target_name = symbol.properties["name"]
            if "title" in symbol.properties or "name" in symbol.properties:
                if ignore_labels and symbol != target_var:
                    target_type = None
                else:
                    target_type = symbol.node_type
                candidates_sorted = self.skb_b.find_closest_nodes_w_cutoff(
                    target_name=target_name,
                    target_type=target_type,
                    logger=self.logger,
                    enable_vss=self.settings.get("vss_cutoff") < 1.0,
                    cutoff_vss=self.settings.get("vss_cutoff"),
                    llm_activation=self.settings.get("llm_activation"),
                    query_for_llm_activation=query,
                    step3_result=r
                )
                if len(candidates) == 0:
                    candidates = candidates_sorted
                else:
                    candidates = [x for x in candidates_sorted if x in candidates]

            if symbol.is_constant:
                if len(candidates) == 0:
                    self.logger.log(f"No nodes for constant {symbol.name}::{symbol.node_type} found in knowledge base. "
                               f"Removing it.")
                    r.num_zero_candidates += 1
                    invalid_symbols.append(f"{symbol.name}::{symbol.node_type}")
                else:
                    self.logger.log(f"Entities found for {symbol.name}::{symbol.node_type}:"
                               f"{len(candidates) > 10 =}, {list(candidates)[:10]=},\n"
                                    f"candidate names: {self.skb_b.entity_ids2name(candidates, n=10)}")
                    symbol.candidates = candidates

        for symbol_key in invalid_symbols:
            valid_symbols.pop(symbol_key)

        r.valid_symbols = valid_symbols
        return r



    def step4b_grounding(self, triplets: list[Triplet], target_variable: TripletEnd,
                        ignore_node_labels: bool, ignore_edge_labels: bool) -> Step4bResult:
        logger = self.logger

        edges_to_consider = list(self.settings.get("edge_type_long2short").values())
        nodes_to_consider = list(self.settings.get("nodes_to_consider"))

        num_variables_without_candidates = 0
        num_variable_candidates = 0

        new_symbols = {}


        filtered_triplets = []
        for triplet in triplets:
            if not ignore_edge_labels and triplet.e not in edges_to_consider:
                continue
            if not ignore_node_labels:
                if triplet.h.is_constant and triplet.h.node_type not in nodes_to_consider:
                    continue
                if triplet.t.is_constant and triplet.t.node_type not in nodes_to_consider:
                    continue
            filtered_triplets.append(triplet)
            new_symbols[triplet.h.get_uid()] = triplet.h
            new_symbols[triplet.t.get_uid()] = triplet.t
        triplets = filtered_triplets
        symbols = new_symbols

        if target_variable.get_uid() not in symbols:
            self.logger.log(f"Target variable is not used! Jumping to backup solution (VSS + LLM Reranker).")
            return Step4bResult(set(), 0, 0, "", skipped=True)
        if len(triplets) == 0:
            self.logger.log(f"No triplets used. Jumping to backup solution (VSS + LLM Reranker).")
            return Step4bResult(set(), 0, 0, "", skipped=True)

        symbols = self.skb_b.ground_triplets(triplets, symbols, logger, target_variable, ignore_edge_labels)

        logger.log(f"Candidates for symbol terms:")
        for symbol_uid in symbols.keys():
            if symbols[symbol_uid].candidates is None:
                logger.log(f"{symbol_uid}: Variable not used.")
            else:
                num_cands = len(symbols[symbol_uid].candidates)
                limit = 50
                if num_cands > limit:
                    logger.log(f"{symbol_uid}: More than {limit} ({num_cands}) candidates found.")
                else:
                    logger.log(f"{symbol_uid}: {symbols[symbol_uid].candidates}")
                if not symbols[symbol_uid].is_constant:
                    if num_cands == 0:
                        num_variables_without_candidates += 1
                    num_variable_candidates += num_cands

        answer = target_variable.candidates
        if answer is None:
            answer = set()
        logger.log(f"{len(answer)=}\n10 answers from the candidates set:\n"
                  f"{self.skb_b.entity_ids2name(answer, 10)}\n\n")
        return Step4bResult(answer, num_variables_without_candidates, num_variable_candidates, target_variable.node_type, skipped=False)

    def validate_step4b(self, r: Step4bResult, ground_truths: list[int]):
        # Stats
        for gt in ground_truths:
            if gt in r.answer_ids:
                r.num_true_pos_in_prefilter += 1

        r.num_target_candidates = len(r.answer_ids)
        r.num_false_pos_in_prefilter = r.num_target_candidates - r.num_true_pos_in_prefilter

        r.recall = r.num_true_pos_in_prefilter / len(ground_truths)
        if r.num_true_pos_in_prefilter == 0:
            r.precision = 0.0
        else:
            r.precision = r.num_true_pos_in_prefilter / r.num_target_candidates



        self.logger.log(f"\nStep 6:\n Number of true positives: {r.num_true_pos_in_prefilter},"
                        f" number of false positives: {r.num_false_pos_in_prefilter}")

        if len(r.answer_ids) == len(self.skb_b.skb.candidate_ids):
            r.answer_ids = set()
            self.logger.log(f"All candidates are target variable candidates. Hence, I am removing them now to save memory.")



    def vss(self, step6_result: Step4bResult, query: str, query_id: int, target_type: str | None,
                  step00result: Step4aResult):
        top_k_node_ids, vss_scores = self.skb_b.vss.get_top_k_nodes(search_str=query, k=self.skb_b.skb.num_candidates,
                                                         node_type=target_type, logger=self.logger,
                                                         node_id_mask=step6_result.answer_ids,
                                                        complement_with_non_masked_ids=True,
                                                         query_id=query_id,
                                                         node_types_to_consider=self.settings.get("nodes_to_consider"))
        if step00result is not None:
            new_order = []
            # Iterate through each set in x
            for current_set in step00result.answers:
                # Iterate through y and add elements to z if they are in the current set
                for element in top_k_node_ids:
                    if element in current_set:
                        new_order.append(element)

            top_hits_vss = new_order
        else:
            top_hits_vss = top_k_node_ids

        return Step7Result(top_hits_vss[:self.settings.get("k_target_variable")],
                           vss_scores[:self.settings.get("k_target_variable")])


    def step8_llm_reranker(self, step5_incl_7_result: Step5Incl7Result, node_id_mask: set[int], query: str):

        top_hits_vss = step5_incl_7_result.vss_top_hits
        top_hits = self.reranker.rerank(top_hits_vss, query, node_id_mask=node_id_mask)

        self.logger.log(f"Results (IDs): {top_hits=}")
        top_hits_str = str([self.skb_b.entity_id2name(x) for x in top_hits])
        self.logger.log(f"Results (aliases): {top_hits_str}")
        return Step8Result(top_hits, top_hits_str)

    def validate_step5(self, step5_result: Step5Incl7Result, ground_truths: list[int]):
        step5_result.ground_truths = ground_truths

    def validate_step7(self, step7_result: Step7Result, ground_truths: list[int]):
        step7_result.ground_truths = ground_truths

    def validate_step8(self, step8_result: Step8Result, ground_truths: list[int]):
        step8_result.ground_truth_str = self.skb_b.entity_ids2name(ground_truths, 10)
        step8_result.ground_truths = ground_truths