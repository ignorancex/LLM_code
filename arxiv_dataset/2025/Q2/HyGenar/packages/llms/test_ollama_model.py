from unittest import TestCase

from packages.llms.ollama_model import OllamaModel


class TestOllama(TestCase):
    def setUp(self):
        import warnings
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
    def test_predict(self):
        model = OllamaModel()
        model.reconfig({
            "model": "gemma:7b-instruct",
            "temperature": 0,
            "max_tokens": 20,
        })
        message = "Hello"
        content = model.chat(message)
        self.assertEqual("Hello! 👋 It's great to hear from you. What can I do for you today?",content)