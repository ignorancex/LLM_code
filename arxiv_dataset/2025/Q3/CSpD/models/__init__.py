from . import mar


def create_mar(model_name, **kwargs):
    return mar.__dict__[model_name](**kwargs)
