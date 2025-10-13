from abc import ABC, abstractmethod
from utils import get_openai_clients_and_models
import shortuuid
import base64
import io
from PIL import Image
import random

class QuestionSample(ABC):
    
    def __init__(self, row, args, round_idx=0):
        self.row = row
        self.args = args
        self.round_idx = round_idx
        self.image = row['image']  # base64
        self.question = row['question']
        self.answer = row['answer']
        self.category = row['category']
        # Instance-level clients and models
        self.clients, self.models = get_openai_clients_and_models(self.args.model_path)
        self.current_client_idx = 0

    async def generate(self, prompt, image, max_tokens=1024):
        # Rotate client selection
        # Randomly select client and model
        idx = random.randint(0, len(self.clients) - 1)
        client = self.clients[idx]
        model = self.models[idx]

        # Process image scaling
        image_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Get target size from args
        target_size = (self.args.image_size, self.args.image_size)
        
        # Only scale if image is larger than target size
        if img.width > target_size[0] or img.height > target_size[1]:
            # Calculate scaling ratio
            ratio = min(target_size[0]/img.width, target_size[1]/img.height)
            new_size = (int(img.width*ratio), int(img.height*ratio))
            
            # Use bilinear interpolation for scaling
            img = img.resize(new_size, Image.Resampling.BILINEAR)
            
            # Convert back to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            processed_image = base64.b64encode(buffered.getvalue()).decode()
        else:
            processed_image = image

        chat_completion = await client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{processed_image}"
                        }
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    }
                ]
            }],
            model=model,
            max_tokens=max_tokens,
            temperature=self.args.temperature if self.args.temperature > 0 else 0.0,
            extra_body={
                "add_generation_prompt": True
            }
        )
        
        result = chat_completion.choices[0].message.content
        return result

    async def process(self):
        try:
            return await self._process()
        except Exception as e:
            import traceback
            print(f"Error occurred while processing sample: {e}")
            traceback.print_exc()  # Print full error stack trace
            
            return {
                "question_id": self.row['index'],
                "round_id": self.round_idx,
                "prompt": "",
                "text": "",  # No longer default to returning 'A'
                "answer_id": shortuuid.uuid(),
                "model_id": self.args.model_path,
                "answer": self.row['answer'],
                "metadata": {"error": str(e)}
            }

    @abstractmethod
    async def _process(self):
        """Abstract method that must be implemented by subclasses"""
        pass