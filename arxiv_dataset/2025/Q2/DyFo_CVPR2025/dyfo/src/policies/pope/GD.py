from .policy import QuestionSample as BaseQuestionSample
import shortuuid
import random
import aiohttp
import traceback

class GDQuestionSample(BaseQuestionSample):
    def __init__(self, row, args, round_idx=0):
        super().__init__(row, args, round_idx)
        
        # Vision expert API
        self.expert_ports = [1]  # Multiple expert ports, corresponding to port number +8000
        self.expert_ports = [port + 8000 for port in self.expert_ports]
        self.expert_base_url = "http://localhost:{}/predict"
        
    async def get_expert_boxes(self, image, text):
        """Call vision expert to get boxes"""
        try:
            # Randomly select expert
            port = random.choice(self.expert_ports)
            
            expert_url = self.expert_base_url.format(port)
            timeout = aiohttp.ClientTimeout(total=10000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    expert_url,
                    json={
                        "image": image,  # image is already base64 string
                        "text": text
                    }
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        print(f"Vision expert API returned error status: {response.status}")
                        print(f"Error message: {error_text}")
                        print(f"Request URL: {expert_url}")
                        print(f"Request text: {text}")
                        return None
        except Exception as e:
            print(f"Error occurred when calling vision expert: {str(e)}")
            print(f"Request URL: {expert_url}")
            print(f"Request text: {text}")
            print(f"Exception stack: {traceback.format_exc()}")
            return None

    async def _process(self):
        # Get vision expert feedback
        expert_result = await self.get_expert_boxes(self.image, self.row['question'])
        
        # Determine answer based on whether boxes exist
        if expert_result and expert_result['boxes']:
            final_answer = 'Yes'
        else:
            final_answer = 'No'
            
        return {
            "question_id": self.row['index'],
            "round_id": self.round_idx,
            "prompt": self.row['question'],
            "text": final_answer,
            "answer_id": shortuuid.uuid(),
            "model_id": self.args.model_path,
            "answer": self.row['answer'],
            "metadata": {
                "expert_result": expert_result if expert_result else {}
            }
        }
