import openai
import torch
from typing import Union, List
from logger import Logger
from settings import Settings
from vss import VSS


import functools #This is for the custom sorting using python's built-in sorting methods

class LLAMAReranker:
    def __init__(self,
                 skb,
                 llm_bridge,
                 vss: VSS,
                 settings: Settings,
                 logger: Logger,
                 sim_weight: float = 0.1):
        """
        Initializes the LLMReranker model.

        Args:
            kb (SemiStruct): Knowledge base.
            llm_model : The actual LLM model.
            tokenizer : The actual tokenizer.
            emb_model_name (str): Embedding model name.
            sim_weight (float): Weight for similarity score.
        """
        self.skb = skb
        self.parent_vss = vss
        self.logger = logger
        self.settings = settings
        self.llm_bridge = llm_bridge
        self.sim_weight = sim_weight
        self.reranking_method = self.settings.get("reranking_method")
        self.max_k = settings.get("k_target_variable")

        self.add_rel = settings.configs["llm"]["add_rel"]
        self.compact_docs = settings.configs["llm"]["compact_docs"]




    def method0_reranking(self, top_k_node_ids: list[int], query: Union[str, List[str]], node_id_mask: set,
                          add_rel: bool) -> (list[int]):
        """
        Forward pass to compute predictions for the given query using LLM reranking.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list, None]): Query index (optional).

        Returns:

        """
        cand_len = len(top_k_node_ids)

        pred_dict = {}

        prompts = []
        for idx, node_id in enumerate(top_k_node_ids):
            doc_info = self.skb.get_doc_info(node_id, add_rel=add_rel, compact=self.compact_docs)
            node_type = self.skb.get_node_type_by_id(node_id)

            prompts.append(
                f'Examine if a {node_type} '
                f'satisfies a given query and assign a score from 0.0 to 1.0. '
                f'If the {node_type} does not satisfy the query, the score should be 0.0. '
                f'If there exists explicit and strong evidence supporting that {node_type} '
                f'satisfies the query, the score should be 1.0. If partial evidence or weak '
                f'evidence exists, the score should be between 0.0 and 1.0.\n'
                f'Here is the query:\n\"{query}\"\n'
                f'Here is the information about the {node_type}:\n' +
                doc_info + '\n\n' +
                f'Please score the {node_type} based on how well it satisfies the query. '
                f'ONLY output the floating point score WITHOUT anything else. '
                f'Output: The numeric score of this {node_type} is: '
            )


        answers, _ = self.llm_bridge.ask_llm_batch(prompts, chat_logs=None)
        for idx, node_id in enumerate(top_k_node_ids):
            try:
                llm_score = float(answers[idx])
            except TypeError:
                if answers[idx] is None:
                    if add_rel:
                        raise RuntimeError()
                    else:
                        llm_score = 0.5
            except ValueError:
                llm_score = 0.5
            sim_score = (cand_len - idx) / cand_len
            score = llm_score + self.sim_weight * sim_score

            # prefer nodes that have been in node_id_mask, i.e. that have been prefiltered
            if node_id_mask is not None:
                score /= 2
                if idx < len(node_id_mask):
                    score += 0.5
            pred_dict[node_id] = score

        node_scores = torch.FloatTensor(list(pred_dict.values()))
        top_k_idx = torch.topk(node_scores, min(self.max_k, len(node_scores)), dim=-1, largest=True, sorted=True
                               ).indices.tolist()
        top_k_node_ids = [list(pred_dict.keys())[i] for i in top_k_idx]

        return top_k_node_ids


    def method1_reranking(self, top_k_node_ids: list[int],
                                       query: Union[str, List[str]],
                                       node_id_mask: set ) -> (list[int]):
        
        """
        Forward pass to compute predictions for the given query using LLM reranking.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list, None]): Query index (optional).

        Returns:
            a ordered list by how well the elements satsify the given query.
        """

        def method1_for_list_of_nodes(node_ids_to_rerank, query):
            
            if len(node_ids_to_rerank) == 0:
                return []

            possible_answers = "\n"
            
            for node_id in node_ids_to_rerank:
                
                doc_info = self.skb.get_doc_info(node_id, add_rel=self.add_rel, compact=self.compact_docs)
                possible_answers += str(node_id) + " " + doc_info + "\n"

            prompt = (
                    f'The rows of the following list consist of an ID number, a type and a corresponding descriptive text:\n'
                    f'{possible_answers} \n'
                    f'Please sort this list in descending order according to how well the elements can be considered as '
                    f'answers to the following query: \n'
                    f'{query} \n'
                    f'Please make absolutely sure that the element which satisfies the query best is the first element in your order. '
                    f'Return ONLY the corresponding ID numbers separated by commas in the asked order.'
            )

            output, _ = self.llm_bridge.ask_llm_batch([prompt], chat_logs=None)

            try:

                answer = [int(node_id_str.strip()) for node_id_str in output[0].split(",")]
                
                answer = list(dict.fromkeys(answer)) #Remove duplicate Node_ids

                
                sorted_IDs = [node_id for node_id in answer if node_id in node_ids_to_rerank] #remove invented IDs
                invented_ids = len(answer) - len(sorted_IDs)
                print("LLM has invented: " ,invented_ids , " node IDs in it's answer." )
                missing_ids = len(node_ids_to_rerank) - len(sorted_IDs)
                print("LLM out does not contain ", missing_ids, " IDs from the input.")

            except:
                sorted_IDs = []
                print("LLM output contains elements that cannot be cast to integer.")
                print("Erroneous LLM output: ", output[0])

            sorted_IDs += [node_id for node_id in node_ids_to_rerank if node_id not in sorted_IDs]

            return sorted_IDs

        
        to_rerank = top_k_node_ids

        answer = method1_for_list_of_nodes(to_rerank, query)

        answer_prioritize_prefiltered = [x for x in answer if x in node_id_mask]
        answer_prioritize_prefiltered += [x for x in answer if x not in node_id_mask]

        return answer_prioritize_prefiltered


    def pairwise_comparison(self, node1_id : int, node2_id : int, query : str, add_rel: bool):
        """
        Function to compare two nodes in a SKB by how good they satisfy a query.

        Args:
            node1_id: ID of the first node
            node2_id: ID of the second node
            query: Query 
        Returns:
            {-1,0,1} depending on:
            -1 if node2_id satisfies the given query better.
            0 if the LLM output cannot be cast to a node_ID (in many cases the LLM outputs neither) or it is none of the given node IDs or if the two node_ids are identical.
            1 if node1_id satisfies the given query better.
        """

        if self.reranking_method == 3:
            print("Comparison for: ",query, "with nodes: ", node1_id," and: ", node2_id)

        if node1_id == node2_id:    #make sure that the comparison is reflective.
            return 0
        
        node_type_1 = self.skb.get_node_type_by_id(node1_id)
        node_type_2 = self.skb.get_node_type_by_id(node2_id)


        doc_info_1 = self.skb.get_doc_info(node1_id, add_rel=add_rel, compact=self.compact_docs)
        doc_info_2 = self.skb.get_doc_info(node2_id, add_rel=add_rel, compact=self.compact_docs)

        prompt = (
            f'The following two elements consist of an ID number, a type and a corresponding descriptive text:\n \n'
            f'{node1_id}, {node_type_1}, {doc_info_1}. \n'
            f'{node2_id}, {node_type_2}, {doc_info_2}. \n\n'
            f'Find out which of the elements satisfies the following query better: \n'
            f'{query} \n'
            f'Return ONLY the corresponding ID number which corresponds to the element that satisfies '
            f'the given query best. Nothing else.'
        )


        answer, _ = self.llm_bridge.ask_llm_batch([prompt], chat_logs=None)
        answer = answer[0]
        if isinstance(answer, str):
            answer = answer.replace("'","").replace('"','').strip()
        if answer == "A":
            answer = node1_id
        elif answer == "B":
            answer = node2_id

        try:
            answer = int(answer)
        except:
            print("LLM output cannot be cast to int.")
            print("Erroneous LLM output: ", answer)
            return 0    #we then assume the elements to be equally bad/good - often output is neither satisfies the query?
        if answer == node1_id:
            return 1
        elif answer == node2_id:
            return -1
        else:
            print("LLM output is neither of the given node IDs")
            print("Erroneous LLM output: ", answer)
            return 0


    def method2_reranking(self, top_k_node_ids: list[int],
                                       query: Union[str, List[str]],
                                       node_id_mask: set) -> (list[int]):
        """
        Forward pass to compute predictions for the given query using LLM reranking.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list, None]): Query index (optional).

        Returns:
            a ordered list by how well the elements satsify the given query.
        """
        
        to_rerank = top_k_node_ids

        try: # set add_rel = False, if prompt gets too long
            answer = sorted(to_rerank, key = functools.cmp_to_key(lambda node1_id, node2_id : self.pairwise_comparison(
                node1_id, node2_id, query = query, add_rel=self.add_rel)), reverse = True)
        except RuntimeError:
            answer = sorted(to_rerank, key=functools.cmp_to_key(lambda node1_id, node2_id: self.pairwise_comparison(
                node1_id, node2_id, query=query, add_rel=False)), reverse=True)

        return answer


    def rerank(self, top_k_node_ids: list[int], query: Union[str, List[str]], node_id_mask: set) -> (list[int]):
        

        top_k_node_ids = top_k_node_ids[:self.max_k]

        if self.reranking_method == 0:
            try:  # set add_rel = False, if prompt gets too long
                sorted_node_ids = self.method0_reranking(top_k_node_ids, query, node_id_mask, add_rel=self.add_rel)
            except RuntimeError:
                sorted_node_ids = self.method0_reranking(top_k_node_ids, query, node_id_mask, add_rel=False)
            except openai.BadRequestError:
                sorted_node_ids = self.method0_reranking(top_k_node_ids, query, node_id_mask, add_rel=False)

        elif self.reranking_method == 1:
            sorted_node_ids = self.method1_reranking(top_k_node_ids, query, node_id_mask)
            
        elif self.reranking_method == 2:
            sorted_node_ids = self.method2_reranking(top_k_node_ids, query, node_id_mask)

        else:
            raise(NotImplementedError("Reranking_method_not_specified!"))

        return sorted_node_ids
