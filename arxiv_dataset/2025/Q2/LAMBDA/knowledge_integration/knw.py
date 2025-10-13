import inspect
import textwrap


KNW_INJECTION = {}

class knw:
    def __init__(self):
        self.name = 'knowledge_integration'
        self.description = 'Integrate knowledge into the LLM.'
        self.core_function = 'core_function'
        self.runnable_function = None
        self.mode = 'full'
        self.method_code = {}


    def get_core_function(self):
        """
        Core function of the knowledge integration.
        """
        function_name = self.core_function
        if function_name:
            core_function = getattr(self, function_name, None)
            return textwrap.dedent(core_function())
        else:
            return "None code is provided."

    def get_runnable_function(self):
        """
        Runnable function of the knowledge integration.
        """
        function_name = self.runnable_function
        if function_name:
            runnable_function = getattr(self, function_name, None)
            rn_code = textwrap.dedent(runnable_function())
            return rn_code
        else:
            return ''

    def get_all_code(self):
        return self.get_runnable_function() + self.get_core_function()