from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import torch
from eval_utils import get_device

class lmdeploy_model:
    def __init__(self, args):
        if "AWQ" in args.model_path:
            model = pipeline(args.model_path, 
                            backend_config=TurbomindEngineConfig(session_len=32768, model_format='awq'), 
                            log_level='ERROR')
        else:
            model = pipeline(args.model_path, 
                            backend_config=TurbomindEngineConfig(session_len=32768), 
                            log_level='ERROR')
        self.model = model
        self.args = args

    def inference(self, prompt_batch, image_path_batch):
        image_batch = [load_image(image_path) for image_path in image_path_batch]
        model_input_prompts = [(prompt, image) for prompt, image in zip(prompt_batch, image_batch)]
        with torch.no_grad():
            response_batch = self.model(model_input_prompts)

        response_batch = [response.text.strip() for response in response_batch]
        
        return response_batch

        

