import gdb

class PrintEventLocals(gdb.Command):
    """Print all local variables except the specified one."""

    def __init__(self):
        super(PrintEventLocals, self).__init__("print_event_locals",
                                             gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        show_var = {
            'iev': '',
        }
        frame = gdb.selected_frame()
        locals = frame.block()

        for symbol in locals:
            if symbol.is_variable:
                if symbol.name in show_var.keys():
                    key = symbol.name
                    attribute = show_var[key]
                    value = gdb.parse_and_eval(f"{key}{attribute}")
                    print(f"{key}{attribute} = {value}")


PrintEventLocals()
