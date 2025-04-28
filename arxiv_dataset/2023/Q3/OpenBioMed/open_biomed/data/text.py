from typing_extensions import Self

class Text:
    def __init__(self) -> None:
        self.str = None

    @classmethod
    def from_str(cls, sample: str) -> Self:
        text = cls()
        text.str = sample
        return text

    def __str__(self) -> str:
        return self.str