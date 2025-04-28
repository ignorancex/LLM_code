from crossasr.text import Text
class TranscriptionResult:
    def __init__(self, text: Text, transcriptions: dict, cases: dict):
        self.text = text
        self.transcriptions = transcriptions
        self.cases = cases

    def get_text(self):
        return self.text

    def get_transcriptions(self):
        return self.transcriptions

    def get_cases(self):
        return self.cases
        