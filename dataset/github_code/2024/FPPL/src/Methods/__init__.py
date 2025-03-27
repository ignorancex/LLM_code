def get_fed_method(args):
    name = args.method.lower()
    if "fppl" in name:
        from Methods.FPPL import Method
    else:
        assert 0

    return Method(args)
