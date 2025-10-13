import ast
from argparse import ArgumentParser, Namespace

import tqdm.auto
import regex as re

from framework import Framework
from logger import Step1Result, Step2bResult, Step3Result, Step4bResult, Step4aResult
from triplet import TripletEnd, Triplet


def parse_args() -> Namespace:
    """
    Parses and returns console arguments.
    :returns: parsed console arguments
    """
    parser = ArgumentParser()

    # Experiment name
    parser.add_argument("--experiment_name", default=None)

    # Eval. data
    parser.add_argument("--dataset", choices=['amazon', 'prime', 'mag'])
    parser.add_argument("--split", choices=["train", "val", "test", "human_generated_eval"])
    parser.add_argument("--question_ids", default="-1", type=str)     # "-1" to select all

    # Models
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--emb_model", default=None)

    #Optional
    parser.add_argument("--steps", nargs="+", type=int, default=[1,2,3,4,5,6,7,8])
    parser.add_argument('--ignore_node_labels', dest='ignore_node_labels', action='store_true')
    parser.add_argument('--consider_node_labels', dest='ignore_node_labels', action='store_false')
    parser.add_argument('--ignore_edge_labels', dest='ignore_edge_labels', action='store_true')
    parser.add_argument('--consider_edge_labels', dest='ignore_edge_labels', action='store_false')
    parser.set_defaults(ignore_node_labels=False)
    parser.set_defaults(ignore_edge_labels=False)

    parser.add_argument("--config_file_path", type=str, default=None)

    return parser.parse_args()



def parse_conditions_from_cypher(cypher_query: str, triplet_ends: dict[str, TripletEnd],
                                 properties_dict: dict[str,str]):
    equals_pattern = re.compile(r'(\w+)\.(\w+)\s*(?:(=(?:~?)|CONTAINS|<|>)\s*(?:[\'"]([^\'"]*)[\'"]|(\w+))|IN\s*(\[[^\]]*\]))')

    atoms = re.findall(equals_pattern, cypher_query)
    for atom in atoms:
        var_name, var_property, var_equality, var_value, var_value_word, var_values = atom
        var_value += var_value_word
        if var_equality == "<" or var_equality == ">":
            var_value = var_equality + var_value
        try:
            var_values = ast.literal_eval(var_values)
            if var_value != "":
                var_values.append(var_value.replace("(?i)", ""))
        except (ValueError, SyntaxError):
            var_values = [var_value.replace("(?i)", "")]

        if var_name in triplet_ends:

            triplet_ends[var_name].is_constant = True

            if var_property in properties_dict:
                var_property = properties_dict[var_property]
                if var_property in triplet_ends[var_name].properties:
                    triplet_ends[var_name].properties[var_property] += "; " + "; ".join(var_values)
                else:
                    triplet_ends[var_name].properties[var_property] = "; ".join(var_values)


def parse_cypher_to_triplets(cypher_query: str, rel_dict: dict[str, str], properties_dict: dict[str,str],
                             node_type_list: list[str], symbols: dict[str,TripletEnd]=None) -> list[Triplet]:
    # Regular expression to match patterns like (HEAD)-[RELATION]->(TAIL) or (HEAD)<-[RELATION]-(TAIL)
    if symbols is None:
        symbols = {}

    pattern_left = r'\(([^)]+)\)<-\[[^:\]]*\:?([^\]]+)\]-\(([^)]+)\)'
    pattern_right = r'\(([^)]+)\)-\[[^:\]]*\:?([^\]]+)\]->?\(([^)]+)\)'
    pattern_symbol_w_property_only = r'\(\w+\:?[^\s\)]+\s*\{\w+\s*\:\s*[\'"][^\'"]+[\'"]\}\)'

    node_pattern = (r'(?:(\w*)\:([^\s\)]+)\s*(?:\{(\w+)\s*\:\s*[\'"]([^\'"]*)[\'"]\})?)|'
                    r'(?:(\w+))')

    # Convert matches to list of triplets
    triplets = []

    def add_symbol(node: str, idx: int):
        node_matches = re.findall(node_pattern, node.strip())
        if len(node_matches) != 1:
            return None
        var_name, n_type, n_property, n_property_value, var_name_only = node_matches[0]
        if n_type not in node_type_list:
            n_type = None

        if var_name_only != "":
            var_name = var_name_only

        if var_name == "":
            var_name = "c" + str(idx)

        if var_name in symbols:
            symbol = symbols[var_name]
        else:
            symbol = TripletEnd(var_name, n_type, is_constant=False)
        if symbol.node_type is None:
            symbol.node_type = n_type
        if n_property in properties_dict:
            symbol.is_constant = True
            if n_property not in symbol.properties:
                symbol.properties[properties_dict[n_property]] = n_property_value
            else:
                if n_property_value not in symbol.properties[n_property]:
                    symbol.properties[properties_dict[n_property]] += "; " + n_property_value
        return symbol

    def add_triplet(tail, relation, head):
        head = add_symbol(head, len(symbols))
        tail = add_symbol(tail, len(symbols) + 1)
        if head is None or tail is None:
            return None
        symbols[head.name] = head
        symbols[tail.name] = tail
        if relation in rel_dict:
            relation = rel_dict[relation.split(":")[-1]]
        return Triplet(head, relation, tail)

    # Find all matches in the Cypher query
    for match in re.findall(pattern_left, cypher_query, overlapped=True):
        tail, relation, head = match[2].strip(), match[1].strip(), match[0].strip()
        triplet = add_triplet(head, relation, tail)
        if triplet is not None:
            triplets.append(triplet)
    for match in re.findall(pattern_right, cypher_query, overlapped=True):
        tail, relation, head = match[0].strip(), match[1].strip(), match[2].strip()
        triplet = add_triplet(head, relation, tail)
        if triplet is not None:
            triplets.append(triplet)
    symbols_w_props = re.findall(pattern_symbol_w_property_only, cypher_query)
    for symbol in symbols_w_props:
        symbol = add_symbol(symbol, len(symbols))
        if symbol is not None:
            symbols[symbol.name] = symbol
    return triplets


def extract_from_cipher(cypher_str: str, rel_dict: dict[str,str], properties_dict: dict[str, str],
                        node_type_list: list[str]) -> (str, list[str, TripletEnd], list[str, Triplet], TripletEnd, str):
    cypher_str = cypher_str.strip("´`\n ;")
    cypher_str_split = cypher_str.split("RETURN")
    if len(cypher_str_split) != 2:
        return None, None, None, None, "ERROR: Cypher string contains no or too many RETURN operations."
    match_part, return_part = cypher_str_split
    match_part = match_part.split("AS")[0]  # remove AS-part of query
    if len(match_part.split("WHERE")) > 1:
        where_part = match_part.split("WHERE")[1]
    else:
        where_part = None
    match_part = match_part.split("WHERE")[0]

    symbols = {}
    triplets = parse_cypher_to_triplets(match_part, rel_dict, properties_dict, node_type_list, symbols)
    if where_part is not None:
        triplets_add = parse_cypher_to_triplets(where_part, rel_dict, properties_dict, node_type_list, symbols)
        triplets.extend(triplets_add)
        parse_conditions_from_cypher(where_part, symbols, properties_dict)

    target_var_name = return_part.split(".")[0].strip()
    target_type_pattern = re.compile(r'\b' + target_var_name + r'\b:([^\s)]+)')
    target_type = re.findall(target_type_pattern, cypher_str)
    if len(target_type) == 0:
        return None, None, None, None, "ERROR: No target type in Cypher string found. Using None."
    target_type = target_type[0]
    warnings = ""
    if target_var_name not in symbols:
        symbols[target_var_name] = TripletEnd(target_var_name, target_type, is_constant=False)
        warnings += "WARNING: Target variable not in triplets."
    else:
        symbols[target_var_name].node_type = target_type

    t_variable = symbols[target_var_name]
    symbols_w_uid = {}
    for symbol in symbols.values():
        #if symbol.node_type is not None:
        symbols_w_uid[symbol.get_uid()] = symbol

    return target_type, symbols_w_uid, triplets, t_variable, warnings

def extract_from_cipher2(cypher_str: str, rel_dict: dict[str,str], properties_dict: dict[str, str],
                        node_type_list: list[str]) -> (str, list[str, TripletEnd], list[str, Triplet], TripletEnd, str):
    cypher_str = cypher_str.strip("´`\n ;")
    cypher_str_split = cypher_str.split("RETURN")
    if len(cypher_str_split) != 2:
        return None, None, None, None, "ERROR: Cypher string contains no or too many RETURN operations."
    match_part, return_part = cypher_str_split

    symbols = {}
    triplets = parse_cypher_to_triplets(match_part, rel_dict, properties_dict, node_type_list, symbols)
    parse_conditions_from_cypher(match_part, symbols, properties_dict)

    target_var_name = return_part.split(".")[0].strip()
    target_type_pattern = re.compile(r'\b' + target_var_name + r'\b:([^\s)]+)')
    target_type = re.findall(target_type_pattern, cypher_str)
    if len(target_type) == 0:
        return None, None, None, None, "ERROR: No target type in Cypher string found. Using None."
    target_type = target_type[0]
    warnings = ""
    if target_var_name not in symbols:
        symbols[target_var_name] = TripletEnd(target_var_name, target_type, is_constant=False)
        warnings += "WARNING: Target variable not in triplets."
    else:
        symbols[target_var_name].node_type = target_type

    t_variable = symbols[target_var_name]
    symbols_w_uid = {}
    for symbol in symbols.values():
        #if symbol.node_type is not None:
        symbols_w_uid[symbol.get_uid()] = symbol

    return target_type, symbols_w_uid, triplets, t_variable, warnings


def qa_pair2str(experiment, q_id: int) -> str:
    query, _, answer_ids, _ = experiment.eval_data[q_id]

    out = f"\n++++++++++ question nr {q_id} ++++++++++++++\n"
    out += query + "\nAnswers:\n"
    expected_answers = experiment.skb_b.expected_answers(answer_ids, separator=" OR ")
    out += expected_answers[:-3]
    out += f"\n++++++++++ end of question nr {q_id} ++++++++++++++\n\n"
    return out


def main(question_ids: str, dataset_name: str, experiment_name: str, data_split: str, llm_model: str,
         emb_model: str, steps: list[str], ignore_node_labels: bool, ignore_edge_labels: bool, configs_path: str = None):
    """
    Evaluations FocusedRetriever on a given STARK dataset and given question IDs.
    Args:
        question_ids: set to -1 to select all available question IDs.
        experiment_name:
        dataset_name:
        data_split:
        :param ignore_edge_labels:
        :param ignore_node_labels:
        :param configs_path:
        :param steps:
        :param data_split:
        :param experiment_name:
        :param dataset_name:
        :param question_ids:
        :param emb_model:
        :param llm_model:
    """
    framework = Framework(experiment_name, dataset_name, data_split, llm_model=llm_model, enable_vss=True,
                          emb_model=emb_model, configs_path=configs_path)
    if question_ids == "-1":
        if data_split == "human_generated_eval":
            question_ids = framework.eval_data.indices
        elif data_split == "test":
            question_ids = framework.eval_data.split_indices[data_split].reshape(-1).tolist()
            question_ids = question_ids[:int(len(question_ids) * 0.1)]
        elif data_split == "val":
            question_ids = framework.eval_data.split_indices[data_split].reshape(-1).tolist()
            question_ids = question_ids[:int(len(question_ids) * 0.1)]
        else:
            raise ValueError()
    elif "[" and "]" in question_ids:
            question_ids = ast.literal_eval(question_ids)
    elif "-" in question_ids:
        start, end = question_ids.split("-")
        question_ids = range(int(start), int(end) + 1)
    else:
        question_ids = [int(question_ids)]


    logger = framework.logger

    # Load already saved results from files
    logger.load_step1()
    logger.load_step2a()
    logger.load_step2b()
    logger.load_step3()
    logger.load_step4a()
    logger.load_step4b()
    logger.load_step5_incl_7()
    logger.load_step6()
    logger.load_step7()
    logger.load_step8()

    alpha = framework.settings.get("alpha")

    for question_id in tqdm.auto.tqdm(question_ids):
        query, _, ground_truths, _ = framework.eval_data[question_id]
        logger.log(qa_pair2str(framework, question_id))

        if 1 in steps and alpha > 0:  # DERIVE_CYPHER_QUERY
            if question_id not in logger.step1_results:
                nodes_to_consider = str(framework.skb_b.skb.node_type_lst()).replace("'","")
                edges_to_consider = str(list(framework.settings.configs.get("edge_type_long2short").keys())).replace("'","")
                properties_to_consider = str(list(framework.settings.configs.get("avail_node_properties").keys())).replace("'","")
                prompt = (f'Return a Cypher query derived from the given query Q. But follow my requirements precisely! '
                          f'Use a very basic, short Cypher syntax. Contentwise, omit everything that cannot be captured exactly '
                          f'with one of the given, available labels. Omit quantifications. Do not use OR or |.'
                          f'Format dates as YYYY-MM-DD. ' 
                          'Only return one solution. '
                          f'Q: {query}\n\nAvailable node labels: {nodes_to_consider}\n\n'
                          f'Available properties: {properties_to_consider}\n\n'
                          f'Available relation labels: {edges_to_consider}\n\n'
                          f'Available keywords: [MATCH, WHERE, RETURN, AND, OR]\n\n'
                          "Example: MATCH (p:pathway)-[:interacts_with]->(g:gene/protein) WHERE g.name = 'IGF1' RETURN p.title")
                [cypher_str], _ = framework.llm_bridge.ask_llm_batch([prompt])
                rel_dict = framework.settings.configs.get("edge_type_long2short")
                properties_dict = framework.settings.configs.get("avail_node_properties")
                target_type, symbols, triplets, target_variable, error_message = (
                    extract_from_cipher2(cypher_str, rel_dict, properties_dict, framework.skb_b.skb.node_type_lst()))

                if target_type is not None:
                    target_variable.properties["title"] = query
                logger.log(error_message)
                step1_result = Step1Result(cypher_str, target_type, triplets, symbols, target_variable, error_message)
                logger.save_step1(question_id, query, step1_result)

        if 2 in steps and alpha > 0: # REGEX: Regular expressions - derive target node type + properties of constants + triplets incl. variables
            # get target node type
            if question_id not in logger.step2a_results:
                step1_result = logger.get_step1_result(question_id)
                step2a_result = framework.save_and_validate_step2a(step1_result.target_type, ground_truths)
                logger.save_step2a(question_id, step2a_result)

            # get triplets and symbols
            if not question_id in logger.step2b_results:
                step1_result = logger.get_step1_result(question_id)
                step2a_result = logger.get_step2a_result(question_id)
                if step2a_result.is_invalid:
                    step2b_result = Step2bResult({}, 0, skipped=True)
                else:
                    num_valid_constants = 0
                    num_invalid_constants = 0
                    constants = {}
                    for symbol in step1_result.symbols.values():
                        if symbol.is_constant:
                            num_valid_constants += 1
                            constants[symbol.get_uid()] = symbol
                    step2b_result = Step2bResult(constants, num_invalid_constants, step2a_result.is_invalid)
                logger.save_step2b(question_id, step2b_result)
        if (3 in steps or 4 in steps) and alpha > 0:   # VSS: Rank constant candidates
            if not question_id in logger.step3_results and question_id not in logger.step4b_results:
                step1_result = logger.get_step1_result(question_id)
                step2b_result = logger.get_step2b_result(question_id)
                if step2b_result.skipped or step2b_result.num_valid_constants == 0:
                    step3_result = Step3Result(skipped=True)
                    step3_result.valid_symbols = {}
                else:
                    step3_result = framework.step3_entity_search(step1_result.symbols, query,
                                                                 step1_result.target_variable, ignore_node_labels)
                # uses a lot of storage:
                # logger.save_step3(question_id, step3_result)
                # instead, only save to RAM
                logger.step3_results[question_id] = step3_result

        if 4 in steps and alpha > 0:   # Set joins by intersection: Ground triplets
            if not question_id in logger.step4b_results:
                step1_result = logger.get_step1_result(question_id)
                step3_result = logger.get_step3_result(question_id)
                if step3_result.skipped or len(step1_result.triplets) == 0 or len(step1_result.symbols) == 0:
                    step4a_result = Step4aResult([set()], skipped=True)
                    dummy_step4b_result = Step4bResult(set(), 0, 0, "", skipped=True)
                else:
                    answers = [set[int]()]
                    candidate_clones = {}
                    for a in step3_result.valid_symbols.values():
                        candidate_clones[a.get_uid()] = a.candidates

                    k = 1
                    dummy_step4b_result = Step4bResult(set(), 0, 0, "", skipped=False)
                    while len(answers[-1]) < framework.settings.configs["k_target_variable"] and k < 10000:# and k < framework.settings.configs["k_target_variable"]: #framework.settings.configs["k"]:
                        for a in step1_result.symbols.values():
                            if a.is_constant:
                                if type(candidate_clones[a.get_uid()]) is list:
                                    a.candidates = set(candidate_clones[a.get_uid()][:k])
                                else:
                                    a.candidates = candidate_clones[a.get_uid()]
                            else:
                                a.candidates = None
                        step1_result.target_variable.candidates = candidate_clones[step1_result.target_variable.get_uid()]
                        if isinstance(step1_result.target_variable.candidates, list):
                            step1_result.target_variable.candidates = set(step1_result.target_variable.candidates)
                        dummy_step4b_result = framework.step4b_grounding(step1_result.triplets, step1_result.target_variable,
                                                                         ignore_node_labels, ignore_edge_labels)
                        if dummy_step4b_result.skipped:
                            answers.append(set[int]())
                            break
                        answers.append(dummy_step4b_result.answer_ids)
                        k = int (k * 1.5 + 0.5)

                    framework.validate_step4b(dummy_step4b_result, ground_truths)
                    answers[-1] = dummy_step4b_result.answer_ids.copy()

                    for i in range(len(answers) - 1, 0, -1):
                        answers[i] = set(answers[i]) - set(answers[i-1])
                    step4a_result = Step4aResult(answers[1:])
                logger.save_step4a(question_id, step4a_result)
                logger.save_step4b(question_id, dummy_step4b_result)

        if 6 in steps:   # LLM: derive target node type
            if not question_id in logger.step6_results:
                target_type = framework.step6_get_target_type(query)
                step6_result = framework.save_and_validate_step6(target_type, ground_truths)
                logger.save_step6(question_id, step6_result)

        if 7 in steps:   # Pure VSS + step 6: Rank all target candidates
            if not question_id in logger.step7_results:
                dummy_step4b_result = Step4bResult(set(), 0, 0, "", skipped=True)
                target_type = logger.get_step6_result(question_id).target_type
                step7_result = framework.vss(dummy_step4b_result, query, question_id, target_type, None)
                framework.validate_step7(step7_result, ground_truths)
                logger.save_step7(question_id, step7_result)

        if 5 in steps and alpha > 0:   # VSS: Rank target candidates
            if not question_id in logger.step5_incl_7_results:
                step1_result = logger.get_step1_result(question_id)
                step4a_result = logger.get_step4a_result(question_id)
                step4b_result = logger.get_step4b_result(question_id)
                step7_result = logger.get_step7_result(question_id)

                if step4b_result.skipped or len(step4b_result.answer_ids) == 0:
                    fallback = True
                    logger.log("ERROR. First steps failed. Now using backup method.")
                else:
                    fallback = False

                if fallback:
                    target_type = logger.get_step2a_result(question_id).target_type
                    if target_type is None:
                        step6_fallback = False
                    else:
                        step6_fallback = True
                    step5_result = step7_result
                    step5_result.step6_fallback = step6_fallback
                    step5_result.fallback_solution = fallback
                else:
                    target_type = step4b_result.target_type

                    if "title" in step1_result.target_variable.properties:
                        search_str = query + " " + step1_result.target_variable.properties["title"] #
                    else:
                        search_str = query
                    step5_result = framework.vss(step4b_result, search_str, question_id, target_type,
                                                 step4a_result)

                    answers_combined = step5_result.vss_top_hits[:alpha]
                    vss_scores_combined = step5_result.vss_scores[:alpha]
                    j = 0
                    while len(answers_combined) < framework.settings.configs["k_target_variable"]:
                        if step5_result.vss_top_hits[j] not in answers_combined:
                            answers_combined.append(step5_result.vss_top_hits[j])
                            vss_scores_combined.append(step5_result.vss_scores[j])
                        j += 1
                    step5_result.vss_top_hits = answers_combined
                    step5_result.vss_scores = vss_scores_combined

                framework.validate_step5(step5_result, ground_truths)
                logger.save_step5_incl_7(question_id, step5_result)

        if 8 in steps:  # LLM: rerank top-k candidates
            if not question_id in logger.step8_results:
                if alpha > 0:
                    step1_result = logger.get_step1_result(question_id)
                    dummy_step4b_result = logger.get_step4b_result(question_id)
                    step5_incl_7_result = logger.get_step5_incl_7_result(question_id)

                    step8_result = framework.step8_llm_reranker(step5_incl_7_result, dummy_step4b_result.answer_ids, query)
                else:
                    step7_result = logger.get_step7_result(question_id)
                    step8_result = framework.step8_llm_reranker(step7_result, set(), query)
                framework.validate_step8(step8_result, ground_truths)
                logger.save_step8(question_id, step8_result, query)
    print('Done.')


if __name__ == '__main__':
    args = parse_args()
    main(question_ids=args.question_ids, dataset_name=args.dataset, experiment_name=args.experiment_name,
         data_split=args.split, steps=args.steps, llm_model=args.llm_model, emb_model=args.emb_model,
         ignore_node_labels=args.ignore_node_labels, ignore_edge_labels=args.ignore_edge_labels,
         configs_path=args.config_file_path)
