import dspy
from langchain.chat_models import ChatOpenAI
from utils.remote_llm import RemoteAPILLM
import subprocess

class code_agent(dspy.Tool):
    """A tool for generating and executing Python code"""
    
    def __init__(self, model_name=None):
        # Define the function that will be called
        def generate_code_func(query: str, context: str = "") -> str:
            return self.generate_code(query, context)
            
        super().__init__(
            func=generate_code_func,
            name="code_agent",
            desc="Generate and execute Python code based on a query",
            # input_variable="query",  # Add input_variable
            args={
                "query": str,
                "context": str
            }
            # arg_desc={
            #     "query": "The code generation query",
            #     "context": "Optional context for code generation"
            # }
        )
        self.model_name = model_name
        # Initialize the appropriate model
        if model_name in ['gpt-4o', 'claude-3.5-sonnet']:
            self.llm = RemoteAPILLM(model_name=model_name)
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.7, streaming=True)

    def generate_code(self, query: str, context: str = "") -> str:
        """Generate code based on query and context"""
        # Generate code
        result = self.llm.invoke("Given the Context: {}\n\n Write a code snippet in Python for the given Problem. Make sure it can be run as a script and directly output the result. OUTPUT JUST CODE SNIPPET AND NOTHING ELSE. Problem:{}".format(context, query))
        result = result.content

        # Clean up code if it's in markdown format
        if "```python" in result:
            result = result[result.find("```python") + 9:result.rfind("```")].strip()
        elif "```" in result:
            result = result[result.find("```") + 3:result.rfind("```")].strip()
        
        # Write and execute code
        path = "scripts/tools/temp.py"
        with open(path, "w") as file:
            file.write(result)
        
        try:
            result = subprocess.run(['python', path], capture_output=True, text=True, timeout=10)
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Code execution timed out after 10 seconds"
    
    def __call__(self, query: str, context: str = "") -> str:
        """Make the class callable with the same interface as generate_code"""
        return self.generate_code(query, context)

# """test code"""
# openai =ChatOpenAI(model_name="gpt-4o", streaming=True)
# tools=[generate_code]

# # memory = ConversationBufferMemory(memory_key="chat_history")
# agent_chain=initialize_agent(tools, openai, verbose=True)
# res = agent_chain.run({'input':'Generate code to check are following numbers are palindrome: 123, 121, 12321, 12345, 123454321.'})
# print(res)