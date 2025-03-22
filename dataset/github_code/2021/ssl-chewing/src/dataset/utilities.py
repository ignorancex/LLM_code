def suggest_patience(epochs: int) -> int:
    """Current implementation: 10% of total epochs, but can't be less than 5."""
    assert isinstance(epochs, int)

    return max(5, round(.1 * epochs))
