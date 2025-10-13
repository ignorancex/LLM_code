from models.HuatuoGPTmodel.cli import HuatuoChatbot

class HuatuoGPT:
    def __init__(self, args):
        model = HuatuoChatbot(args.model_path)
        self.model = model
        self.args = args

    def inference(self, prompt_batch, image_path_batch):
        output_batch = []
        for prompt, image_path in zip(prompt_batch, image_path_batch):
            response = self.model.inference(prompt, image_path)
            if type(response) == list:
                response = response[0]
            output_batch.append(response)
        return output_batch