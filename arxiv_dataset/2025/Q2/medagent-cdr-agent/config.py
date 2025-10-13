def OpenaiConfig(model):
    if model == 'gpt-3.5-turbo':
        config = {
            "model": "gpt-3.5-turbo",
            "api_key": "",
            "temperature": 0,
        }
    elif model == 'gpt-4':
        config = {
            "model": "gpt-4",
            "api_key": "",
            "temperature": 0,
        }
    elif model == 'gpt-4o':
        config = {
            "model": "gpt-4o",
            "api_key": "",
            "temperature": 0,
        }
    elif model == 'gpt-4o-mini':
        config = {
            "model": "gpt-4o-mini",
            "api_key": "",
            "temperature": 0,
        }

    return config
