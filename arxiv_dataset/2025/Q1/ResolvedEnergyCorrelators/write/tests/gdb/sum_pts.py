import gdb

class SumPseudoJetPts(gdb.Command):
    """Command to sum pt values of PseudoJet objects in a vector."""

    def __init__(self):
        super(SumPseudoJetPts, self).__init__("sum_pseudojets_pt", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        # Argument should be the name of the std::vector<PseudoJet>
        vector_name = arg.strip()
        try:
            # Access the vector
            vector = gdb.parse_and_eval(vector_name)
            size   = gdb.parse_and_eval(f'{vector_name}.size()')

            # Iterate through the vector elements
            total_pt = 0.0
            for i in range(size):
                pt = gdb.parse_and_eval(f'{vector_name}.at({i}).pt()')
                total_pt += float(pt)
            print(f"Total pt sum: {total_pt}")

        except gdb.error as e:
            print(f"Error: {e}")

# Register the command with GDB
SumPseudoJetPts()

