from mod.expert import Expert


def test_expert():
    model_id = "gpt2"
    expert = Expert(model_id)
    print(expert.config)
