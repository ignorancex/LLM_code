import torch
from tqdm import tqdm
from PIL import Image

class Engine(): 

	def __init__(self, args):
		self.args = args

	def get_output(self, model, dataloader, collectables):

		output = {k:[] for k in collectables}
		output["model_output"] = []

		model = model.eval()
		with torch.no_grad():

			for _, batch in enumerate(tqdm(dataloader)):
				
				batch["images"] = [Image.fromarray(img.numpy()) for img in batch["images"]]
				batch["images"] = [[img] for img in batch["images"]]

				out = model(batch)
				if type(out) == torch.Tensor:
					out = out.cpu()

				output["model_output"].append(out)
				for k in collectables:
					output[k].append(batch[k])

		for k in output:
			if type(output[k][0]) == list:
				output[k] = [item for sublist in output[k] for item in sublist]

			elif type(output[k][0]) == torch.Tensor:
				output[k] = torch.cat(output[k], dim=0)

			else: 
				raise ValueError("Unsupported type in collectables")                

		return output
