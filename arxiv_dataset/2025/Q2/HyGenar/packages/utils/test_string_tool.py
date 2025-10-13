from unittest import TestCase

from packages.utils.string_tool import split_string_with_escapes, unescape_string, escape_string, extract_code_block


class Test(TestCase):
    def test_no_escape_characters(self):
        # Test string with no escape characters
        s = "123"
        result = split_string_with_escapes(s)
        self.assertEqual(result, ["1","2","3"])

    def test_single_newline_escape(self):
        # Test string with a single newline escape character
        s = r"1\n2"
        result = split_string_with_escapes(s)
        self.assertEqual(result, ["1", r"\n","2"])

    def test_multiple_escape_characters(self):
        # Test string with multiple escape characters
        s = r"1\n2\t3"
        result = split_string_with_escapes(s)
        self.assertEqual(result, ["1", r"\n","2", r"\t", "3"])

    def test_escape_backslash(self):
        # Test string with backslash escape character
        s = r"A\\B"
        result = split_string_with_escapes(s)
        self.assertEqual(result, ["A", r"\\", "B"])

    def test_mixed_content(self):
        # Test string with letters, numbers, and escape characters
        s = r"Hello\nWorld\t!"
        result = split_string_with_escapes(s)
        self.assertEqual(result, ["H","e","l","l","o", r"\n", "W","o","r","l","d", r"\t", "!"])

    def test_empty_string(self):
        # Test empty string input
        s = ""
        result = split_string_with_escapes(s)
        self.assertEqual(result, [])

    def test_only_escape_characters(self):
        # Test string with only escape characters
        s = r"\n\t\\"
        result = split_string_with_escapes(s)
        self.assertEqual(result, [r"\n", r"\t", r"\\"])

    def test_unescape_string(self):
        s = r"1\n2\t3"
        result = unescape_string(s)
        self.assertEqual(result, "1\n2\t3")

    def test_escape_string(self):
        s = "1\n2\t3"
        result = escape_string(s)
        self.assertEqual(result, r"1\n2\t3")

    def test_extract_code_block(self):
        # Test code block with language specified
        s = "```python\nprint('Hello, World!')```"
        result = extract_code_block(s)
        self.assertEqual(result, "print('Hello, World!')")

        # Test code block with no language specified
        s = "```\nprint('Hello, World!')```"
        result = extract_code_block(s)
        self.assertEqual(result, "print('Hello, World!')")

        # Test code block with multiple lines
        s = "```python\nprint('Hello, World!')\nprint('1')```"
        result = extract_code_block(s)
        self.assertEqual(result, "print('Hello, World!')\nprint('1')")

        # Test code block with outside content
        s = "Hello\n```python\nprint('Hello, World!')```World"
        result = extract_code_block(s)
        self.assertEqual(result, "print('Hello, World!')")

        # Test code block with no new line on backticks
        s = "```print('Hello, World!')```"
        result = extract_code_block(s)
        self.assertEqual(result, "print('Hello, World!')")



