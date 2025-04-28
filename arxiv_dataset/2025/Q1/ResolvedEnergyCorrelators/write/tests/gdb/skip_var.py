import gdb

class PrintFilteredLocals(gdb.Command):
    """Print all local variables except the specified one."""

    def __init__(self):
        super(PrintFilteredLocals, self).__init__("print_filtered_locals",
                                                  gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        skip_var = arg.strip()
        frame = gdb.selected_frame()
        locals = frame.block()

        for symbol in locals:
            if symbol.is_variable and symbol.is_argument == False:
                if symbol.name == skip_var:
                    continue
                value = symbol.value(frame)
                print(f"{symbol.name} = {value}")

PrintFilteredLocals()

