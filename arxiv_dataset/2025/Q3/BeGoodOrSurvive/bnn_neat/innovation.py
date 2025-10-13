class InnovationNumberGenerator:
    def __init__(self, start=0):
        self.current = start

    def get_next(self):
        innov_num = self.current
        self.current += 1
        return innov_num

# Instantiate it somewhere accessible
innovation_number_generator = InnovationNumberGenerator()

# Maintain a global innovation history
innovation_history = {}


def get_innovation_number(input_node_id, output_node_id):
    key = (input_node_id, output_node_id)
    if key not in innovation_history:
        # Assign a new innovation number
        innovation_history[key] = innovation_number_generator.get_next()
    return innovation_history[key]
