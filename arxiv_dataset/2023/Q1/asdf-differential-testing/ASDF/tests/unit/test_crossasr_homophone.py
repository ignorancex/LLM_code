import sys
sys.path.append('../../')

from crossasr import TranscriptionResult
from crossasr import Text


import unittest
from examples.mutator.homophone import Homophone

class TestHomophone(unittest.TestCase):

    def setUp(self) -> None:
        self.mutator = Homophone()

    def tearDown(self) -> None:
        self.mutator.reset()

    def test_get_all_error_words(self):
        
        text = "hello world hello"
        transcriptions = {
            "deepspeech": "hello world hello",  # exactly the same
            "deepspeech2": "hi world hello",  # missed one hello
            "wav2letter": "hello wo hello",  # world transcription is incorrect
            "wit": "hello world",  # missed the last hello
            "wav2vec2": ""  # no output
        }
        cases = {
            "deepspeech": 2,
            "deepspeech2": 1,
            "wav2letter": 1,
            "wit": 1,
            "wav2vec2": 1,
        }
        expected_result = list(set(["hello", "world"]))

        self.mutator.save_transcription_result(TranscriptionResult('1', text, transcriptions, cases))
        all_error_words = self.mutator.get_all_error_words(text, transcriptions, cases)

        self.assertEqual(all_error_words, expected_result)

    def test_get_error_text_per_asr(self):
        text = "hello world hello"
        transcriptions = {
            "deepspeech": "hello world hello",  # exactly the same
            "deepspeech2": "hi world hello",  # missed one hello
            "wav2letter": "hello wo hello",  # world transcription is incorrect
            "wit": "hello world",  # missed the last hello
            "wav2vec2": ""  # no output
        }
        cases = {
            "deepspeech": 2,
            "deepspeech2": 1,
            "wav2letter": 1,
            "wit": 1,
            "wav2vec2": 1,
        }
        expected_result = {
            "deepspeech": [],
            "deepspeech2": ["hello"],
            "wav2letter": ["world"],
            "wit": ["hello"],
            "wav2vec2": ["hello", "world"]
        }

        self.mutator.save_transcription_result(TranscriptionResult('1', text, transcriptions, cases))
        asrs_error_words = self.mutator.get_error_words_per_asr(text, transcriptions, cases)

        self.assertEqual(asrs_error_words, expected_result)

    def test_generate_wrapped_mutated_sentences(self):
        filename = '1'
        text = "hello world hello"
        transcriptions = {
            "deepspeech": "hello world hello",  # exactly the same
            "deepspeech2": "hi world hello",  # missed one hello
            "wav2letter": "hello wo hello",  # world transcription is incorrect
            "wit": "hello world",  # missed the last hello
            "wav2vec2": ""  # no output
        }
        cases = {
            "deepspeech": 2,
            "deepspeech2": 1,
            "wav2letter": 1,
            "wit": 1,
            "wav2vec2": 1,
        }
        expected_result = [Text(f"m_{filename}", "hello whirled hello")]

        self.mutator.save_transcription_result(TranscriptionResult(filename, text, transcriptions, cases))
        mutated_sentences = self.mutator.generate_wrapped_mutated_sentences()

        self.assertEqual(mutated_sentences, expected_result)

if __name__ == "__main__":
    unittest.main()