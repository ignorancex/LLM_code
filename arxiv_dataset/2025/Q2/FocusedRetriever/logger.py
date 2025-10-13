import ast
import csv
from pathlib import Path

#csv.field_size_limit(sys.maxsize)

from triplet import TripletEnd, Triplet


class Step1Result:
    def __init__(self, cypher_str: str, target_type: str, triplets: list[Triplet], symbols: dict[str,TripletEnd],
                 target_variable: TripletEnd, error_message: str):
        self.cypher_str = cypher_str
        self.target_type = target_type
        self.triplets = triplets
        self.symbols = symbols
        self.target_variable = target_variable
        self.error_message = error_message

class Step2aResult:
    def __init__(self, target_type: str, is_invalid: bool, is_incorrect: bool,
                 ground_truth: str):
        self.target_type = target_type
        self.is_invalid = is_invalid
        self.is_incorrect = is_incorrect
        self.ground_truth = ground_truth

class Step2bResult:
    def __init__(self, symbols: dict[str, TripletEnd], num_invalid_constants: int, skipped: bool=False):
        self.symbols = symbols
        self.num_valid_constants = len(symbols)
        self.num_invalid_constants = num_invalid_constants
        self.skipped = skipped

class Step3Result:
    def __init__(self, skipped:bool=False):
        self.valid_symbols = None

        self.skipped = skipped

        self.num_key_matches = 0
        self.num_zero_candidates = 0

        self.num_constants_w_vss = 0
        self.num_constants_w_llm_activation = 0

        self.num_key_matches_candidates = 0
        self.num_vss_candidates = 0
        self.num_w_llm_activation_candidates = 0

class Step4aResult:
    def __init__(self, answers: list[set[int]], skipped: bool =False):
        self.answers = answers
        self.skipped = skipped


class Step4bResult:
    def __init__(self, answer_ids: set[int], num_variables_without_candidates: int,
                 num_variable_candidates: int, target_type: str, skipped:bool=False):
        if answer_ids is None:
            self.answer_ids = set()
        else:
            self.answer_ids = answer_ids
        self.num_variables_without_candidates = num_variables_without_candidates
        self.num_variable_candidates = num_variable_candidates
        self.skipped = skipped
        self.num_true_pos_in_prefilter = 0
        self.num_false_pos_in_prefilter = 0
        self.precision = 0
        self.recall = 0
        self.num_target_candidates = 0
        self.target_type = target_type

class Step5Incl7Result:
    def __init__(self, vss_top_hits: list[int], vss_scores: list, fallback_solution: bool = False, step6_fallback: bool = False):
        self.vss_top_hits = vss_top_hits
        self.fallback_solution = fallback_solution
        self.step6_fallback = step6_fallback
        self.ground_truths = None
        self.vss_scores = vss_scores

class Step6Result:
    def __init__(self, target_type: str, is_invalid: bool, is_incorrect: bool,
                 ground_truth: str):
        self.target_type = target_type
        self.is_invalid = is_invalid
        self.is_incorrect = is_incorrect
        self.ground_truth = ground_truth

class Step7Result:
    def __init__(self, vss_top_hits: list[int], vss_scores: list, fallback_solution: bool = False, step6_fallback: bool = False):
        self.vss_top_hits = vss_top_hits
        self.fallback_solution = fallback_solution
        self.step6_fallback = step6_fallback
        self.ground_truths = None
        self.vss_scores = vss_scores

class Step8Result:
    def __init__(self, answer_ids: list[int], answer_str: str, fallback_solution: bool = False, step6_fallback: bool = False):
        self.final_answer_str = answer_str
        self.answer_ids = answer_ids
        self.fallback_solution = fallback_solution
        self.step6_fallback = step6_fallback
        self.ground_truth_str = None
        self.ground_truths = None

def load_symbols_from_str(skipped: bool, symbols_str: str, only_uids: bool=False) -> dict[str, TripletEnd]:
    if skipped:
        return {}
    symbols = {}
    if symbols_str == "":
        return symbols
    for symbol in symbols_str.split("<<<>>>"):
        uid = symbol.split(", self.is_constant=")[0]
        name, node_type = uid.split("::")
        if node_type == "None":
            node_type = None
        if only_uids:
            is_constant = True
            candidates = None
        else:
            is_constant = symbol.split(", self.is_constant=")[1]
            candidates = is_constant.split(", self.candidates=")[1]
            candidates = ast.literal_eval(candidates)

            is_constant = is_constant.split(", self.properties=")[0]
            is_constant = is_constant == "True"

            properties = symbol.split(", self.properties=")[1].split(", self.candidates=")[0]
            properties = ast.literal_eval(properties)
        triplet_end = TripletEnd(name, node_type, is_constant, candidates)
        if not only_uids:
            triplet_end.properties = properties
        symbols[triplet_end.get_uid()] = triplet_end
    return symbols

def symbols_to_str(skipped: bool, symbols: dict[str, TripletEnd], only_uids: bool=False) -> str:
    if skipped:
        return ""
    elif only_uids:
        return "<<<>>>".join([x.get_uid() for x in list(symbols.values())])
    else:
        return "<<<>>>".join([str(x) for x in list(symbols.values())])

def triplets_to_str(triplets: list[Triplet]):
        if triplets is None or len(triplets) == 0:
            return ''
        else:
            return "<<<>>>".join([str(t) for t in triplets])

def str_to_triplets(skipped: bool, triplet_str: str, symbols: dict[str,TripletEnd]):
    triplets = []
    if not skipped and triplet_str != "":
        triplets_str = triplet_str.split("<<<>>>")

        for triplet in triplets_str:
            triplet = triplet.split(" -> ")
            try:
                h = symbols[triplet[0]]
                e = triplet[1]
                r = symbols[triplet[2]]
                triplets.append(Triplet(h, e, r))
            except KeyError:
                print("STOP")
    return triplets

class Logger:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.qa_stats = {}

        self.step1_results = {}
        self.step2a_results = {}
        self.step2b_results = {}
        self.step3_results = {}
        self.step4a_results = {}
        self.step4b_results = {}
        self.step5_incl_7_results = {}
        self.step6_results = {}
        self.step7_results = {}
        self.step8_results = {}

    def get_step1_result(self, question_id) -> Step1Result:
        return self.step1_results[question_id]

    def get_step2a_result(self, question_id) -> Step2aResult:
        return self.step2a_results[question_id]

    def get_step2b_result(self, question_id) -> Step2bResult:
        return self.step2b_results[question_id]

    def get_step3_result(self, question_id) -> Step3Result:
        return self.step3_results[question_id]

    def get_step4a_result(self, question_id) -> Step4aResult:
        return self.step4a_results[question_id]

    def get_step4b_result(self, question_id) -> Step4bResult:
        return self.step4b_results[question_id]

    def get_step5_incl_7_result(self, question_id) -> Step5Incl7Result:
        return self.step5_incl_7_results[question_id]

    def get_step6_result(self, question_id) -> Step6Result:
        return self.step6_results[question_id]

    def get_step7_result(self, question_id) -> Step7Result:
        return self.step7_results[question_id]

    def get_step8_result(self, question_id) -> Step8Result:
        return self.step8_results[question_id]

    def log(self, text: str, print_to_console: bool = True):
        if print_to_console:
            print(text)

        # Open the log_file file in append mode ('a')
        with (open(self.output_path / "log.txt", 'a', encoding='utf-8', errors='replace') as log_file):
            log_file.write(text + "\n")


    def save_step1(self, question_id: int, query:str, r: Step1Result):
        self.step1_results[question_id] = r
        symbols = symbols_to_str("ERROR:" in r.error_message, r.symbols, only_uids=False)
        triplet = triplets_to_str(r.triplets)
        target_variable = "" if r.target_variable is None else r.target_variable.get_uid()

        # save to file
        file_path = self.output_path / "step1.csv"
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,query,cypher_str,target_type,symbols,triplets,target_variable,errors\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id,query, r.cypher_str,r.target_type,symbols,triplet, target_variable, r.error_message])

    def load_step1(self):
        file_path = self.output_path / "step1.csv"

        if file_path.exists():
            self.log("Step1 results loaded.")
        else:
            self.log("Step1 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                symbols = load_symbols_from_str(False, row["symbols"])
                if row["target_variable"] == "":
                    target_variable = None
                else:
                    if row["target_variable"] in symbols:
                        target_variable = symbols[row["target_variable"]]
                    else:
                        row["target_variable"] = None

                triplets = str_to_triplets(False, row["triplets"], symbols)

                r = Step1Result(row["cypher_str"], row["target_type"], triplets, symbols, target_variable, row["errors"])

                self.step1_results[int(row["q_id"])] = r

    def save_step4a(self, question_id: int, r: Step4aResult):
        self.step4a_results[question_id] = r

        # save to file
        file_path = self.output_path / "step4a.csv"
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,answers,skipped\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id,r.answers, r.skipped])

    def load_step4a(self):
        file_path = self.output_path / "step4a.csv"

        if file_path.exists():
            self.log("Step4a results loaded.")
        else:
            self.log("Step4a results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                r = Step4aResult(ast.literal_eval(row['answers']), row['skipped'])
                self.step4a_results[int(row["q_id"])] = r


    def save_step2a(self, question_id: int, r: Step2aResult):
        self.step2a_results[question_id] = r

        # save to file
        file_path = self.output_path / "step2a.csv"
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,target_type,ground_truth,invalid,incorrect\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id,r.target_type,r.ground_truth,int(r.is_invalid),int(r.is_incorrect)])



    def load_step2a(self):
        file_path = self.output_path / "step2a.csv"

        if file_path.exists():
            self.log("Step2a results loaded.")
        else:
            self.log("Step2a results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                r = Step2aResult(row['target_type'], bool(int(row["invalid"])), bool(int(row["incorrect"])),
                                 row["ground_truth"])
                self.step2a_results[int(row["q_id"])] = r

    def save_step2b(self, question_id: int, r: Step2bResult):
        self.step2b_results[question_id] = r

        # save to file
        file_path = self.output_path / "step2b.csv"
        if not file_path.exists():
            with open(file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,constants,num_valid_constants,num_invalid_constants,skipped\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id,symbols_to_str(r.skipped, r.symbols, only_uids=True),
                             r.num_valid_constants, r.num_invalid_constants, int(r.skipped)])

    def load_step2b(self):
        file_path = self.output_path / "step2b.csv"

        if file_path.exists():
            self.log("Step2b results loaded.")
        else:
            self.log("Step2b results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                # Parse the constants back into a list of TripletEnd
                skipped = bool(int(row["skipped"]))
                constants = load_symbols_from_str(skipped, row["constants"], only_uids=True)

                r = Step2bResult(constants, int(row["num_invalid_constants"]), skipped)
                self.step2b_results[int(row["q_id"])] = r

    def save_step3(self, question_id: int, r: Step3Result):
        self.step3_results[question_id] = r

        constants = symbols_to_str(r.skipped, r.valid_symbols)

        # save to file
        file_path = self.output_path / "step3.csv"
        if not file_path.exists():
            with open(file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,key_matches,zero_candidates,constants_w_vss,constants_w_llm_activation,"
                    "key_matches_candidates,vss_candidates,w_llm_activation_candidates,skipped,constants\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id,r.num_key_matches,r.num_zero_candidates,r.num_constants_w_vss,
                             r.num_constants_w_llm_activation, r.num_key_matches_candidates, r.num_vss_candidates,
                             r.num_w_llm_activation_candidates, int(r.skipped),constants])

    def load_step3(self):
        file_path = self.output_path / "step3.csv"

        if file_path.exists():
            self.log("Step3 results loaded.")
        else:
            self.log("Step3 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                r = Step3Result()
                r.skipped = bool(int(row["skipped"]))

                r.num_key_matches = row["key_matches"]
                r.num_zero_candidates = row["zero_candidates"]

                r.num_constants_w_vss = row["constants_w_vss"]
                r.num_constants_w_llm_activation = row["constants_w_llm_activation"]

                r.num_key_matches_candidates = row["key_matches_candidates"]
                r.num_vss_candidates = row["vss_candidates"]
                r.num_w_llm_activation_candidates = row["w_llm_activation_candidates"]

                # Parse the constants back into a list of TripletEnd
                r.valid_symbols = load_symbols_from_str(r.skipped, row["constants"])
                self.step3_results[int(row["q_id"])] = r


    def save_step5_incl_7(self, question_id: int, r: Step5Incl7Result):
        self.step5_incl_7_results[question_id] = r
        answers_vss = r.vss_top_hits
        vss_scores = r.vss_scores

        hit_1_vss, hit_5_vss, hit_20_vss, hit_40_vss, hit_50_vss = 0, 0, 0, 0, 0
        reciprocal_rank_vss, reciprocal_rank_20_vss = 0.0, 0.0
        hits = 0
        for i in range(len(answers_vss)):
            if answers_vss[i] in r.ground_truths:
                hits += 1
                if i < 1:
                    hit_1_vss = 1
                if i < 5:
                    hit_5_vss = 1
                if i < 20:
                    hit_20_vss = 1
                if i < 40:
                    hit_40_vss = 1
                if i < 50:
                    hit_50_vss = 1
                if reciprocal_rank_vss <= 0.0:
                    reciprocal_rank_vss = 1.0 / (i + 1)
                if reciprocal_rank_20_vss <= 0.0 and i < 20:
                    reciprocal_rank_20_vss = 1.0 / (i + 1)
        recall_20_vss = hits / min(20, len(r.ground_truths))

        res_file_path = self.output_path / "step5_incl_7.csv"
        if not res_file_path.exists():
            with open(res_file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,hit_1_vss,hit_5_vss,hit_20_vss,hit_40_vss,hit_50_vss,reci_rank_vss,reci_rank_20_vss,recall_20_vss,fallback,step6_fallback,"
                    "answers_vss,vss_scores\n")

        with open(res_file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, hit_1_vss, hit_5_vss, hit_20_vss, hit_40_vss, hit_50_vss, reciprocal_rank_vss,
                             reciprocal_rank_20_vss, recall_20_vss, int(r.fallback_solution), int(r.step6_fallback),
                             answers_vss, vss_scores])

    def load_step5_incl_7(self):
        file_path = self.output_path / "step5_incl_7.csv"

        if file_path.exists():
            self.log("Step5 results loaded.")
        else:
            self.log("Step5 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                skipped = bool(int(row["fallback"]))
                vss_top_hits = ast.literal_eval(row["answers_vss"])
                vss_scores = ast.literal_eval(row["vss_scores"])
                r = Step7Result(vss_top_hits, vss_scores, skipped, bool(int(row["step6_fallback"])))

                self.step7_results[int(row["q_id"])] = r



    def save_step4b(self, question_id: int, r: Step4bResult):
        self.step4b_results[question_id] = r

        # save to file
        file_path = self.output_path / "step4b.csv"
        if not file_path.exists():
            with (open(file_path, 'x', encoding='utf-8', newline='') as result_file):
                result_file.write("q_id,variables_without_candidates,variable_candidates,true_pos,false_pos,"
                                  "precision,recall,target_candidates,skipped,answer,target_type\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id,r.num_variables_without_candidates,r.num_variable_candidates,
                            r.num_true_pos_in_prefilter,r.num_false_pos_in_prefilter, r.precision, r.recall,
                            r.num_target_candidates,int(r.skipped),r.answer_ids,r.target_type])


    def load_step4b(self):
        file_path = self.output_path / "step4b.csv"

        if file_path.exists():
            self.log("Step4b results loaded.")
        else:
            self.log("Step4b results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                skipped = bool(int(row["skipped"]))
                if skipped:
                    answer_ids = set()
                else:
                    answer_ids = ast.literal_eval(row["answer"])
                r = Step4bResult(answer_ids, int(row["variables_without_candidates"]),
                                 int(row["variable_candidates"]), row["target_type"], skipped)

                r.num_true_pos_in_prefilter = int(row["true_pos"])
                r.num_false_pos_in_prefilter = int(row["false_pos"])
                r.num_target_candidates = int(row["target_candidates"])
                self.step4b_results[int(row["q_id"])] = r


    def save_step6(self, question_id: int, r: Step6Result):
        self.step6_results[question_id] = r

        # save to file
        file_path = self.output_path / "step6.csv"
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,target_type,ground_truth,invalid,incorrect\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id,r.target_type,r.ground_truth,int(r.is_invalid),int(r.is_incorrect)])



    def load_step6(self):
        file_path = self.output_path / "step6.csv"

        if file_path.exists():
            self.log("Step6 results loaded.")
        else:
            self.log("Step6 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                r = Step6Result(row['target_type'], bool(int(row["invalid"])), bool(int(row["incorrect"])),
                                row["ground_truth"])
                self.step6_results[int(row["q_id"])] = r

    def save_step7(self, question_id: int, r: Step7Result):
        self.step7_results[question_id] = r
        answers_vss = r.vss_top_hits
        vss_scores = r.vss_scores

        hit_1_vss, hit_5_vss, hit_20_vss, hit_40_vss, hit_50_vss = 0, 0, 0, 0, 0
        reciprocal_rank_vss, reciprocal_rank_20_vss = 0.0, 0.0
        hits = 0
        for i in range(len(answers_vss)):
            if answers_vss[i] in r.ground_truths:
                hits += 1
                if i < 1:
                    hit_1_vss = 1
                if i < 5:
                    hit_5_vss = 1
                if i < 20:
                    hit_20_vss = 1
                if i < 40:
                    hit_40_vss = 1
                if i < 50:
                    hit_50_vss = 1
                if reciprocal_rank_vss <= 0.0:
                    reciprocal_rank_vss = 1.0 / (i + 1)
                if reciprocal_rank_20_vss <= 0.0 and i < 20:
                    reciprocal_rank_20_vss = 1.0 / (i + 1)
        recall_20_vss = hits / min(20, len(r.ground_truths))

        res_file_path = self.output_path / "step7.csv"
        if not res_file_path.exists():
            with open(res_file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,hit_1_vss,hit_5_vss,hit_20_vss,hit_40_vss,hit_50_vss,reci_rank_vss,reci_rank_20_vss,recall_20_vss,fallback,step6_fallback,"
                    "answers_vss,vss_scores\n")

        with open(res_file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, hit_1_vss, hit_5_vss, hit_20_vss, hit_40_vss, hit_50_vss, reciprocal_rank_vss,
                             reciprocal_rank_20_vss, recall_20_vss, int(r.fallback_solution), int(r.step6_fallback),
                             answers_vss, vss_scores])

    def load_step7(self):
        file_path = self.output_path / "step7.csv"

        if file_path.exists():
            self.log("Step7 results loaded.")
        else:
            self.log("Step7 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                skipped = bool(int(row["fallback"]))
                vss_top_hits = ast.literal_eval(row["answers_vss"])
                vss_scores = ast.literal_eval(row["vss_scores"])
                r = Step7Result(vss_top_hits, vss_scores, skipped, bool(int(row["step6_fallback"])))

                self.step7_results[int(row["q_id"])] = r


    def save_step8(self, question_id: int, r: Step8Result, question: str):
        self.step8_results[question_id] = r
        answers = r.answer_ids

        hit_1, hit_5, hit_20, hit_40, hit_50 = 0, 0, 0, 0, 0
        reciprocal_rank, reciprocal_rank_20 = 0.0, 0.0
        hits = 0
        for i in range(len(answers)):
            if answers[i] in r.ground_truths:
                hits += 1
                if i < 1:
                    hit_1 = 1
                if i < 5:
                    hit_5 = 1
                if i < 20:
                    hit_20 = 1
                if i < 40:
                    hit_40 = 1
                if i < 50:
                    hit_50 = 1
                if reciprocal_rank <= 0.0:
                    reciprocal_rank = 1.0 / (i + 1)
                if reciprocal_rank_20 <= 0.0 and i < 20:
                    reciprocal_rank_20 = 1.0 / (i + 1)
        recall_20 = hits / min(20, len(r.ground_truths))

        res_file_path = self.output_path / "step8.csv"
        if not res_file_path.exists():
            with open(res_file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,question,final_answer,ground_truth,hit_1,hit_5,hit_20,hit_40,hit_50,reci_rank,reci_rank_20,recall_20,"
                    "fallback,step6_fallback,answers\n")

        with open(res_file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, question, r.final_answer_str, r.ground_truth_str, hit_1, hit_5, hit_20, hit_40, hit_50,
                             reciprocal_rank, reciprocal_rank_20, recall_20, int(r.fallback_solution), int(r.step6_fallback), answers])

    def load_step8(self):
        file_path = self.output_path / "step8.csv"

        if file_path.exists():
            self.log("Step8 results loaded.")
        else:
            self.log("Step8 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                skipped = bool(int(row["fallback"]))
                answer_ids = ast.literal_eval(row["answers"])
                r = Step8Result(answer_ids, row["final_answer"], skipped, bool(int(row["step6_fallback"])))

                self.step8_results[int(row["q_id"])] = r