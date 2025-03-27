def get_model(network_global, client_id, args):
    name = args.model.lower()
    if "fppl" in name:
        from Models.fppl import Learner
    else:
        assert 0

    return Learner(network_global, client_id, args)
