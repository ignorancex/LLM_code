def get_local(args):
    name = args.method.lower()
    if "fppl" in name:
        from Locals.FPPLLocal import Local

    return Local(args)
