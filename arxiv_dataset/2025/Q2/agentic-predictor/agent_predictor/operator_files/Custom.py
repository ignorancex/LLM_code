class Custom(Operator):
    def __init__(self, llm: LLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
        