import pytest

from judgetuning.llm_client import OllamaCompletionClient


@pytest.mark.skip()
def test_ollama_completion_client():
    client = OllamaCompletionClient(max_pred_len=10)
    completions = client.complete_text(
        requests=[
            "Complete the following sequence. Just output a single number. 1,2,3,",
            "Complete the following sequence. Just output a single number. 1,2,3,4",
        ]
    )
    assert "4" in completions[0].completions[0]
    assert "5" in completions[1].completions[0]
