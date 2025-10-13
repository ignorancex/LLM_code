from unittest import TestCase

from packages.llms.openai_model import OpenAIModel


class TestOpenAIModel(TestCase):
    def setUp(self):
        import warnings
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
    def test_model(self):
        model = OpenAIModel()
        model.reconfig({
            "temperature": 0,
            "max_tokens": 20
        })
        message = "Hello, how are you?"
        content = model.chat(message)
        self.assertTrue(len(content) > 0)
        self.assertEqual("Hello! I'm just a computer program, so I don't have feelings, but I'm here",content)
