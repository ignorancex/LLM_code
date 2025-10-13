from .policy import QuestionSample as BaseQuestionSample
import shortuuid

class CleanQuestionSample(BaseQuestionSample):
    def __init__(self, row, args, round_idx=0):
        super().__init__(row, args, round_idx)

    async def get_final_answer(self):
        # Add prompt to only answer Yes/No
        # prompt = self.question + " Please answer this question with one word."
        prompt = self.question
        answer = await self.generate(prompt, self.image)
        
        # Check if answer contains Yes or No
        if 'yes' in answer.lower():
            return 'Yes', prompt, answer
        return 'No', prompt, answer  # Default to No

    async def _process(self):
        final_answer, prompt, full_answer = await self.get_final_answer()
        
        return {
            "question_id": self.row['index'],
            "round_id": self.round_idx,
            "prompt": prompt,
            "text": final_answer,
            "answer_id": shortuuid.uuid(),
            "model_id": self.args.model_path,
            "answer": self.row['answer'],
            "metadata": {
                "full_answer": full_answer
            }
        }
