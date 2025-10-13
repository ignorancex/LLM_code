from transformers import AutoProcessor
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import time
import argparse
import ray
import torch

class COCOEvaluator:
    def __init__(self, args):
        self.args = args

        self.model_args = {
            "model_path": args.model_path,
            "gpu_memory_utilization": 0.7
        }
        self.coco_gt = COCO(self.args.anno_dir)
        # coco set, you can customize the question for different tasks
        # self.question="Please output bbox coordinates and names of person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush."
        self.question = "Locate every item from the category list in the image and output the coordinates in JSON format. The category set includes person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush."
        self.img_ids = self.coco_gt.getImgIds()[:self.args.sample_num]
        self.images=[os.path.join(self.args.image_dir, x['file_name']) for x in self.coco_gt.loadImgs(self.img_ids)]
        self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.coco_gt.dataset['categories']}

    def evaluate_dataset(self):
        ray.init()
        try:
            tasks = []
            chunk_size = len(self.images) // self.gpu_num

            for i, gpu_id in enumerate(range(self.gpu_num)):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != self.gpu_num-1 else len(self.images)
                chunk = {
                    "images": self.images[start_idx:end_idx],
                    "img_ids": self.img_ids[start_idx:end_idx]
                }
                task = self._infer_on_single_gpu.remote(
                    gpu_id, chunk, self.model_args, self.question, self.cat_name_to_id
                )
                tasks.append(task)

            results = []
            while tasks:
                done, tasks = ray.wait(tasks)
                results.extend(ray.get(done))
            results = sum(results, [])

            return self._run_coco_eval(results)
        finally:
            ray.shutdown()

    @ray.remote(num_gpus=1)
    def _infer_on_single_gpu(gpu_id, chunk, model_args, question, cat_name_to_id):
        from vllm import LLM, SamplingParams
        from qwen_vl_utils import process_vision_info
        
        llm = LLM(
            model=model_args["model_path"],
            gpu_memory_utilization=model_args["gpu_memory_utilization"],
            tensor_parallel_size=1,
        )
        
        sampling_params = SamplingParams(
            temperature=0,
            # repetition_penalty=1.05,
            max_tokens=4096,
            skip_special_tokens=True,
            # frequency_penalty=0.1,
        )

        processor = AutoProcessor.from_pretrained(model_args["model_path"])
        
        messages = []
        for img_path in chunk["images"]:
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"{question}"}
                ]
            }])

        results = []
        for i in tqdm(range(0, len(messages), args.batch_size)):
            batch = messages[i:i+args.batch_size]
            batch_img_ids = chunk["img_ids"][i:i+args.batch_size]
            text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
            
            # Process visual input
            image_inputs, _ = process_vision_info(batch)
            inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt, image in zip(text, image_inputs)]
            
            outputs = llm.generate(
                inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            outputs_decoded = [o.outputs[0].text for o in outputs]
            results.extend(COCOEvaluator._parse_output(outputs_decoded, batch_img_ids, cat_name_to_id))

        return results
    
    @staticmethod
    def _parse_output(outputs, img_ids, cat_name_to_id):
        results = []
        for o, img_id in zip(outputs, img_ids):
            try:
                if '<answer>' in o:
                    o = o.split('<answer>')[1].split('</answer>')[0]
                o = json.loads(o.replace("```json", "").replace("```", ""))
                for inst in o:
                    label = cat_name_to_id.get(inst["label"], None)
                    if label is None:
                        continue
                    x1, y1, x2, y2 = inst["bbox_2d"]
                    results.append({
                        "image_id": img_id,
                        "category_id": label,
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "score": 1.0
                    })
            except Exception as e:
                continue
        return results

    def _run_coco_eval(self, results):
        output_path = os.path.join('logs', os.path.basename(self.args.model_path), 'detection')
        os.makedirs(output_path, exist_ok=True)
        temp_file = os.path.join(output_path, f"temp_results_{time.time()}.json")
        with open(temp_file, 'w') as f:
            json.dump(results, f)
        
        coco_dt = self.coco_gt.loadRes(temp_file)
        
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        result = {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
            "AR@1": coco_eval.stats[6],
            "AR@10": coco_eval.stats[7],
            "AR@100": coco_eval.stats[8],
            "AR_small": coco_eval.stats[9],
            "AR_medium": coco_eval.stats[10],
            "AR_large": coco_eval.stats[11],
        }
        
        
        with open(os.path.join(output_path, 'coco_eval_results.json'), 'w') as f:
            json.dump(result, f)
        print(f'result_file: {temp_file}')
        return result

def main(args):
    evaluator = COCOEvaluator(args)
    result = evaluator.evaluate_dataset()
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Grounding Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--anno_dir", type=str, default=None, help="Data root directory")
    parser.add_argument("--image_dir", type=str, default=None, help="Image root directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (for debugging)")
    args = parser.parse_args()
    main(args)