import gdb

class PrintJetLocals(gdb.Command):
    """Print all local variables except the specified one."""

    def __init__(self):
        super(PrintJetLocals, self).__init__("print_jet_locals",
                                             gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        show_var = {
            'particles': '.size()',
            'jet'      : '',
            'njets_tot': '',
            'all_jets' : '.size()',
            'good_jets': '.size()',
        }
        frame = gdb.selected_frame()
        locals = frame.block()

        for symbol in locals:
            if symbol.is_variable and symbol.is_argument == False:
                if symbol.name in show_var.keys():
                    key = symbol.name
                    attribute = show_var[key]
                    value = gdb.parse_and_eval(f"{key}{attribute}")
                    print(f"{key}{attribute} = {value}")



PrintJetLocals()
