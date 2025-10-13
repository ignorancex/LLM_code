from judgetuning.llm_client import format_request

def test_format_request_gemma2():
    expected_request = (
        "<bos><start_of_turn>user\nYou are a helpful assistant.\nHello, how are you?<end_of_turn>"
        "<start_of_turn>model\n"
    )

    model = "gemma-2-"
    request = format_request(
        model=model,
        system_prompt="You are a helpful assistant.",
        message="Hello, how are you?",
    )

    print("got")
    print(request)
    print("expected")
    print(expected_request)

    assert request == expected_request


def test_format_request_gemma():
    expected_request = (
        "<start_of_turn>user\nYou are a helpful assistant.\nHello, how are you?<end_of_turn>"
        "<start_of_turn>model\n"
    )

    model = "gemma"
    request = format_request(
        model=model,
        system_prompt="You are a helpful assistant.",
        message="Hello, how are you?",
    )

    print("got")
    print(request)
    print("expected")
    print(expected_request)

    assert request == expected_request


def test_format_request_qwen():
    expected_request = (
        "<|im_start|>system\n"
        "you are an AI expert<|im_end|>\n"
        "<|im_start|>user\n"
        "what is bias?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    model = "Qwen2.5"
    request = format_request(
        model=model,
        system_prompt="you are an AI expert",
        message="what is bias?",
    )

    print("got")
    print(request)
    print("expected")
    print(expected_request)

    assert request == expected_request


def test_format_request_llama():
    expected_request = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant for "
        "travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>What can you "
        "help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    # taken from https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    request = format_request(
        model=model,
        system_prompt="You are a helpful AI assistant for travel tips and recommendations",
        message="What can you help me with?",
    )

    print("got")
    print(request)
    print("expected")
    print(expected_request)

    assert request == expected_request


def test_format_request_smol():
    expected_request = """\
<|im_start|>system
[System message]<|im_end|>
<|im_start|>user
[User message]<|im_end|>
<|im_start|>assistant
"""
    print(expected_request)
    # taken from https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    request = format_request(
        model=model,
        system_prompt="[System message]",
        message="[User message]",
    )
    for i, _ in enumerate(expected_request):
        if request[i] != expected_request[i]:
            print(i, request[i], expected_request[i])
    assert request == expected_request
