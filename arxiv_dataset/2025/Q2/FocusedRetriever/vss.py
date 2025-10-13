import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from openai import OpenAI

from stark_qa.skb import SKB
from logger import Logger


def load_emb_model(offline_mode: bool, model_name: str):
    if offline_mode:
        return None
    else:
        if model_name == "text-embedding-ada-002":
            load_dotenv()
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Invalid embedding model name: {model_name}.")


class VSS:
    def __init__(self,
                 skb: SKB,
                 emb_dir: Path,
                 data_split: str,
                 emb_model_name: str,
                 node_ids_by_type: dict,
                 offline_mode: bool

                 ):
        """
        Initializes the VSS + LLMReranker model.
        Loads all embeddings in access efficient tensors.
        Args:
            skb (SemiStruct): Knowledge base.
            emb_dir (Path): Path to directory with all embedding.
            data_split (str): ["train", "val", "test", "human_generated_eval"] Data split mode determining which
                query embeddings to load.
            emb_model_name (str): Embedding model name.
            node_ids_by_type (dict): Dictionary grouping list of node IDs by their node types.
            offline_mode (bool): Whether to internet access is not available.
        """
        self.skb = skb
        self.candidate_ids = skb.candidate_ids
        self.query_emb_dict = {}
        self.emb_model_name = emb_model_name
        self.emb_client = load_emb_model(offline_mode, emb_model_name)

        self.node_ids_by_type = node_ids_by_type

        emb_dir /= emb_model_name

        # loading several embeddings:
        # 1) queries: questions from test sets (lazy loading)
        # 2) candidates: vertices in SKB that are answer candidates
        # 2b) all nodes instead of candidates
        # 3) non-candidate nodes: other vertices in SKB
        # 4) entities: other strings representing entities that have been searched for already


        if data_split == "human_generated_eval":
            self.query_emb_dir = emb_dir / "query_human_generated_eval"
        else:
            self.query_emb_dir = emb_dir / "query"
        query_emb_dict_path = self.query_emb_dir / 'query_emb_dict.pt'

        # 1) query embeddings
        self.query_emb_dict = {}
        if query_emb_dict_path.exists():
            print(f'Loading query embeddings from {query_emb_dict_path}.')
            self.query_emb_dict = torch.load(query_emb_dict_path)
        #     if len(self.query_emb_dict[0]) == 1:
        #         for i in range(len(self.query_emb_dict)):
        #             self.query_emb_dict[i] = self.query_emb_dict[i].reshape(-1)
        #         torch.save(self.query_emb_dict, query_emb_dict_path)

        # 2) candidate embeddings
        # 3) non-candidate embeddings
        nodes_emb_dir = emb_dir / "nodes"

        self.node_emb_dict = {}
        for node_type in skb.node_type_lst():
            node_emb_path = nodes_emb_dir / f'{node_type.replace("/", "")}_embeddings.pt'
            self.node_emb_dict[node_type] = torch.load(node_emb_path)
            assert len(self.node_emb_dict[node_type]) == len(self.node_ids_by_type[node_type]), \
                (f"number of node embeddings ({len(self.node_emb_dict[node_type])}) does not match number of nodes "
                 f"in the SKB ({len(self.node_ids_by_type[node_type])}). {node_type=}.")
        print(f'Loaded embeddings of nodes from {nodes_emb_dir}!')

        # 4) 'open' string embeddings
        entities_emb_dir = emb_dir / "entities"
        entities_emb_dir.mkdir(parents=True, exist_ok=True)
        self.entity_emb_path = entities_emb_dir / 'entity_emb_dict.pt'
        if self.entity_emb_path.exists():
            self.entity_emb_dict = torch.load(self.entity_emb_path)
        else:
            self.entity_emb_dict = {}
        print(f'Loaded {len(self.entity_emb_dict)} entity embeddings from {self.entity_emb_path}!')


    def get_embedding(self, query: str, model: str):
        emb = self.emb_client.embeddings.create(input=query, model=model)
        return torch.FloatTensor(emb.data[0].embedding)

    def compute_similarities(self,
                             query_emb: torch.Tensor,
                             node_type: str,
                             node_id_mask: list[int] | set[int],
                             node_ids_to_exclude: list[int] | set[int] = []) -> dict:
        """
        Forward pass to compute similarity scores for the given query.

        Args:
            node_type: Type of nodes to be returned.
            node_id_mask: A list or set of node IDs to be considered.
            node_ids_to_exclude: A list or set of node IDs to be NOT considered.

        Returns:
            pred_dict (dict): A dictionary of node ids and their corresponding similarity scores.
        """
        similarity = torch.matmul(self.node_emb_dict[node_type], query_emb)

        node_ids = self.node_ids_by_type[node_type]
        score_dict = {node_ids[i]: similarity[i] for i in range(len(similarity))}

        # filter score dict by masks
        if node_id_mask is not None:
            filtered_score_dict = {}
            for node_id in node_id_mask:
                if node_id in score_dict:
                    filtered_score_dict[node_id] = score_dict[node_id]
            score_dict = filtered_score_dict
        if len(node_ids_to_exclude) > 0:
            for node_id in node_ids_to_exclude:
                if node_id in score_dict.keys():
                    score_dict.pop(node_id)
        return score_dict

    def get_query_emb(self,
                      query: str,
                      query_id: int,
                      emb_model: str = None) -> torch.Tensor:
        """
        Retrieves or computes the embedding for the given query or entity.

        Args:
            query (str): Query string.
            query_id (int): Query index.
            emb_model (str): Embedding model to use.

        Returns:
            query_emb (torch.Tensor): Query embedding.
        """
        if emb_model is None:
            emb_model = self.emb_model_name

        # loading embedding of free text (entity embedding) not question embedding from dataset
        if query_id is None:
            # load embedding from cache if available
            if query in self.entity_emb_dict:
                query_emb = self.entity_emb_dict[query]
                # print(f'Entity embedding loaded from {self.entity_emb_path}')
            # retrieve embedding if it is not in the cache
            else:
                print(f"vss.166: {query=}")
                query_emb = self.get_embedding(query, model=emb_model)
                self.entity_emb_dict[query] = query_emb
                torch.save(self.entity_emb_dict, self.entity_emb_path)
                print(f'Entity embedding for {query} saved to {self.entity_emb_path}.')

        # return preloaded query embedding
        elif query_id in self.query_emb_dict:
            query_emb = self.query_emb_dict[query_id]
        else:
            # load single query embedding from single file
            query_emb_dir = self.query_emb_dir / 'query_embs'
            if not query_emb_dir.exists():
                query_emb_dir.mkdir()
            query_emb_dict_path = query_emb_dir / f'query_{query_id}.pt'
            if query_emb_dict_path.exists():
                query_emb = torch.load(query_emb_dict_path).reshape(-1)
                print(f'Query embedding loaded from {query_emb_dict_path}')
            else:
                query_emb = self.get_embedding(query, model=emb_model)
                torch.save(query_emb, query_emb_dict_path)
                print(f'Query embedding saved to {query_emb_dict_path}')
        return query_emb

    def get_top_k_nodes(self, search_str: str, k: int, node_type: str, logger: Logger = None,
                        node_id_mask: list[int] | set[int] = None, complement_with_non_masked_ids=False,
                        query_id: int = None, node_ids_to_exclude: list[int] | set[int] = [],
                        node_types_to_consider: list[str] = [],
                        cutoff=0.0) -> tuple[list, list]:
        """
        Searches for the k nodes with the highest cosine similarity to a search string.
        Args:
            search_str (str): search string
            k (int): number of nodes to return
            logger (Logger): optional
            node_id_mask (list[int] | set[int] = None): list of nodes to prefer
                fill_with_non_masked_ids (bool) : Whether to include any node if the sum of node_id_mask and
                second_node_id_mask is smaller than k.
            query_id (int): ID of a query in the dataset. If it is not None, it replaces the search string
                answer the query.

        Returns:
            list: the k closest nodes
        """
        if cutoff is None:
            cutoff = 0.0
        query_emb = self.get_query_emb(search_str, query_id)
        if node_type is None:
            score_dict = {}
            for n_type in node_types_to_consider:
                score_dict.update(
                    self.compute_similarities(query_emb=query_emb, node_type=n_type,
                                              node_id_mask=node_id_mask, node_ids_to_exclude=node_ids_to_exclude))
        else:
            score_dict = self.compute_similarities(query_emb=query_emb, node_type=node_type,
                                                   node_id_mask=node_id_mask, node_ids_to_exclude=node_ids_to_exclude)

        # Remove nodes whose similarity is below cutoff boundary
        if cutoff > 0.0:
            filtered_dict = {}
            for key in score_dict:
                if score_dict[key] >= cutoff:
                    filtered_dict[key] = score_dict[key]
            score_dict = filtered_dict

        # Get top k node IDs based on their similarity
        node_scores = list(score_dict.values())
        top_k_idx = torch.topk(torch.FloatTensor(node_scores), min(k, len(node_scores)), dim=-1, largest=True,
                               sorted=True).indices.tolist()
        # Convert score_dict.keys() to a tensor for efficient indexing
        vss_scores = torch.tensor(node_scores)[top_k_idx].tolist()
        keys_tensor = torch.tensor(list(score_dict.keys()), dtype=torch.long)

        # Use advanced indexing to get the top-k node IDs
        top_k_node_ids = keys_tensor[top_k_idx].tolist()

        # for target variable
        if complement_with_non_masked_ids and node_id_mask is not None and len(node_id_mask) < k:
            additional_node_ids, additional_vss_scores = self.get_top_k_nodes(search_str, k - len(node_id_mask), node_type,
                                                                              logger=logger,
                                                       complement_with_non_masked_ids=False, query_id=query_id,
                                                       node_ids_to_exclude=top_k_node_ids,
                                                       cutoff=cutoff,
                                                       node_types_to_consider=node_types_to_consider)

            top_k_node_ids += additional_node_ids
            vss_scores += additional_vss_scores
            if logger is not None:
                logger.log(f"VSS: Added further answers to candidate list. New list (top10): {top_k_node_ids[:10]}")
            else:
                print(f"VSS: Added further answers to candidate list. New list (top10): {top_k_node_ids[:10]}")

        return top_k_node_ids, vss_scores
