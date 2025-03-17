from dhnamlib.pylib.context import Environment

config = Environment(
    using_arg_candidate=False,
    constrained_decoding=False,

    num_prediction_beams=4,
)
