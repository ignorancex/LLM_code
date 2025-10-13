class PatchExtractionFailedException(Exception):
    def __init__(self, message="Patch extraction failed."):
        self.message = message
        super().__init__(self.message)

