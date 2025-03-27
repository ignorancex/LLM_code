def get_server(args):
    name = args.method.lower()
    if "fppl" in name:
        from Servers.FPPLServer import Server

    return Server(args)
