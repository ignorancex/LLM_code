from transformers import AutoTokenizer
from tqdm.notebook import tqdm
import torch
import pandas as pd
import json
import argparse

from .data_processing.pipline import process_data, load_data
from .ser.ser_eval import extract_evidence_tfidf_qatc
from .ser.qatc_model import QATCForQuestionAnswering
from .tvc.tvc_eval import classify_claim
from .tvc.model import ClaimModelForClassification

class SemViQAPipeline:
    def __init__(self, model_evidence_QA, model_bc, model_tc, thres_evidence=0.5, length_ratio_threshold=0.6, is_qatc_faster=False, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thres_evidence = thres_evidence
        self.length_ratio_threshold = length_ratio_threshold
        self.is_qatc_faster = is_qatc_faster
        
        # Load models
        self.tokenizer_QA = AutoTokenizer.from_pretrained(model_evidence_QA)
        self.model_evidence_QA = QATCForQuestionAnswering.from_pretrained(model_evidence_QA)
        
        self.tokenizer_classify = AutoTokenizer.from_pretrained(model_tc)
        self.model_tc = ClaimModelForClassification.from_pretrained(model_tc)
        self.model_bc = ClaimModelForClassification.from_pretrained(model_bc, num_labels=2)
        
    def predict(self, claim, context, return_evidence_only=False):
        evidence = extract_evidence_tfidf_qatc(
            claim, context, self.model_evidence_QA, self.tokenizer_QA, self.device, confidence_threshold=self.thres_evidence, length_ratio_threshold=self.length_ratio_threshold, is_qatc_faster=self.is_qatc_faster
        )
        
        if return_evidence_only:
            return {"evidence": evidence}
        
        verdict = "NEI"
        prob3class, pred_tc = classify_claim(claim, evidence, self.model_tc, self.tokenizer_classify, self.device)
        
        if pred_tc != 0:
            prob2class, pred_bc = classify_claim(claim, evidence, self.model_bc, self.tokenizer_classify, self.device)
            verdict = "SUPPORTED" if pred_bc == 0 else "REFUTED" if prob2class > prob3class else ["NEI", "SUPPORTED", "REFUTED"][pred_tc]
        
        return {"verdict": verdict, "evidence": evidence}

    def process_batch(self, data_path, output_path, return_evidence_only=False):
        data = pd.read_csv(data_path) if "csv" in data_path else pd.read_json(data_path).T
        print(f'Load data: {len(data)} samples')
        
        data["id"] = data.index
        test_data = load_data(data)
        
        results = {}
        for idx, item in tqdm(test_data.items()):
            results[str(idx)] = self.predict(item[0]['claim'], item[0]['context'], return_evidence_only)
        
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
        
        print(pd.DataFrame(results).T.verdict.value_counts() if not return_evidence_only else "Evidence extraction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/test.json", help="Path to data")
    parser.add_argument("--output_path", type=str, default="output.json", help="Path to output")
    parser.add_argument("--model_evidence_QA", type=str, default="QATC", help="Model evidence QA")  
    parser.add_argument("--model_bc", type=str, default="bc", help="Model 2 class") 
    parser.add_argument("--model_tc", type=str, default="tc", help="Model 3 class") 
    parser.add_argument("--thres_evidence", type=float, default=0.5, help="Threshold evidence")
    parser.add_argument("--return_evidence_only", action="store_true", help="Only extract evidence without classification")
    args = parser.parse_args()

    semviqa = SemViQAPipeline(args.model_evidence_QA, args.model_bc, args.model_tc, args.thres_evidence)
    semviqa.process_batch(args.data_path, args.output_path, args.return_evidence_only)
