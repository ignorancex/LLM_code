"""Helper class allowing users to record and play-back input across invocations.
"""
import os

RECORDINGS_DIR = "{}/recordings".format(os.environ.get("BUILD_WORKING_DIRECTORY", "."))

class ShadowInput:
    """Class for recording, saving, and re-playing user input.

    This is used in interactive.py to make replaying and debugging complex
    sequences of commands easier.

    Initialization: input_ = ShadowInput()

    Usage: replace all instances of input(...) with input_(...).

    NOTE: the string "__FINISH__" has special meaning in our saved recordings,
    so you should never expect users to input it.
    """
    def __init__(self):
        """Initializes a new ShadowInput.
        """
        # Used as a stack of recordings that we're currently reading from
        # (needed eg. if one recording then loads another recording).
        self.in_files = []
        # The user's input(s) since the last begin_recording().
        self.recording = []

    def __call__(self, prompt):
        """Prompt the user for input, then record and return it.

        This is meant to be a drop-in replacement for input(prompt).
        """
        print(prompt, end="")
        while self.in_files:
            line = self.in_files[-1].readline()
            # When __FINISH__ is encountered in a recording, we finish the
            # recording immediately. This is useful for debugging.
            if line == "" or line.strip() == "__FINISH__":
                # This file is finished, move on to the next one.
                self.in_files[-1].close()
                self.in_files.pop()
                continue
            # We read a line from the file successfully.
            line = line.strip()
            print(line, end="\n")
            break
        else:
            # There are no loaded recordings; read an input from the user and
            # record it.
            line = input()
            self.recording.append(line)
        return line

    def begin_recording(self):
        """Clears the recorded user input.
        """
        self.recording = []

    def scrub_last(self, n_inputs):
        """Removes the last @n_inputs lines of user input.

        Used eg. to remove the user input that led to a call to save_recording
        (so the recording doesn't try to save itself unnecessarily).
        """
        self.recording = self.recording[:-n_inputs]

    def load_recording(self, file_name):
        """Begins replaying a saved recording.
        """
        self.in_files.append(open(self.recording_path(file_name), "r"))

    def save_recording(self, file_name):
        """Saves the current recording.

        NOTE: This does *NOT* clear the recording history; if you
        save_recording again before calling begin_recording, the second
        recording will have all the contents of the first one.
        """
        with open(self.recording_path(file_name), "w") as out_file:
            out_file.write("\n".join(self.recording))

    @staticmethod
    def recording_path(file_name):
        """Helper to return the absolute path corresponding to @file_name.

        See RECORDINGS_DIR to set where recordings are stored/read from.
        """
        return "{}/{}".format(RECORDINGS_DIR, file_name)
