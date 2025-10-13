class Color:
    def __init__(self, fg_color_reference: str, bg_color_reference: str):
        if fg_color_reference is None or bg_color_reference is None:
            raise ValueError("Both fg_color_reference and bg_color_reference must not be None.")
        self.fg_color_reference = fg_color_reference
        self.bg_color_reference = bg_color_reference