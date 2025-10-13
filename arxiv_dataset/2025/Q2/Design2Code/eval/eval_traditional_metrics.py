import json, inspect
import argparse
import src.metrics as metrics
from src.metrics.screenshot_capture import *
from tqdm import tqdm 


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark script")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the result json file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results_path = args.results_path

    with open(results_path, 'r') as file:
        results = json.load(file)

    metrics_objects = {}
    metrics_scores = {}

    for name, obj in inspect.getmembers(metrics, inspect.isclass):
        if obj.__module__.startswith('src.metrics') and not obj.__module__.startswith('src.metrics.base_metrics'):  
            metrics_objects[f"{obj.__module__}/{name}"] = obj()
            metrics_scores[f"{obj.__module__}/{name}"] = 0

    for result in tqdm(results['results'], desc="Evaluating"):
        predicted = result["Predicted"]
        ground_truth = result["Ground Truth"]

        ref_img_path = "./graphics/reference.png"
        hyp_img_path = "./graphics/hypothesis.png"

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()

            take_screenshot_from_html_content(page, ground_truth, ref_img_path, do_it_again=True)
            take_screenshot_from_html_content(page, predicted, hyp_img_path, do_it_again=True)
            

        
        for metric_name, metric_object in metrics_objects.items():
            if metric_name.startswith("src.metrics.visual_metrics"):
                metrics_scores[metric_name] += metric_object(ref_img_path, hyp_img_path)
            else:
                metrics_scores[metric_name] += metric_object(ground_truth, predicted)

 
    
    for metric_name, metric_object in metrics_scores.items():
        metrics_scores[metric_name]/=len(results['results'])
    
    # Get the directory of the results_path
    directory = os.path.dirname(results_path)
    
    # Define the new file path
    total_benchmark_path = os.path.join(directory, 'total_results_ws_gemini.json')
    

    with open(total_benchmark_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_scores, f, ensure_ascii=False, indent=4)