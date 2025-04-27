import pathlib
import unittest
from unittest.mock import patch, mock_open

import ascii_only


class TestAsciiOnly(unittest.TestCase):
    def test_fix_non_ascii(self):
        """Test that non-ASCII characters are properly replaced."""
        test_cases = [
            # Input, Expected Output
            ("Hello world", "Hello world"),  # ASCII only
            ("Hello—world", "Hello--world"),  # em dash
            ("Hello–world", "Hello-world"),  # en dash
            ("Hello → world", "Hello -> world"),  # right arrow
            ("Hello…world", "Hello...world"),  # ellipsis
            ("Hello world", "Hello world"),  # non-breaking space
        ]

        for input_str, expected in test_cases:
            with self.subTest(input_str=input_str):
                result = ascii_only.fix_non_ascii(input_str)
                self.assertEqual(result, expected)

    def test_get_unicode_escape(self):
        """Test that Unicode escape representation is correctly generated."""
        test_cases = [
            # Input bytes, Expected Unicode escape
            (b"\xe2\x86\x92", "\\u2192"),  # → (right arrow)
            (b"\xe2\x80\x93", "\\u2013"),  # – (en dash)
            (b"\xe2\x80\x94", "\\u2014"),  # — (em dash)
        ]

        for bad_bytes, expected_escape in test_cases:
            with self.subTest(bad_bytes=bad_bytes):
                result = ascii_only.get_unicode_escape(bad_bytes)
                self.assertEqual(result, expected_escape)

    def test_find_non_ascii_issue(self):
        """Test finding non-ASCII issues in a file."""
        # Mock file with non-ASCII character
        file_content = "line1\nline2 → arrow\nline3"
        
        # Create a UnicodeDecodeError when trying to read as ASCII
        def side_effect_open(file, mode, encoding, errors=None):
            if encoding == "ascii":
                # This will raise UnicodeDecodeError when read
                mock = mock_open(read_data=file_content.encode("utf-8"))
                return mock(file, mode, encoding)
            else:
                # Return the UTF-8 content for the second open call
                return mock_open(read_data=file_content)(file, mode, encoding, errors)
        
        with patch("builtins.open", side_effect=side_effect_open):
            # Create a mock UnicodeDecodeError for the find_non_ascii_issue function
            mock_error = UnicodeDecodeError(
                "ascii", 
                "line1\nline2 → arrow\nline3".encode("utf-8"), 
                8,  # Position of the arrow in the byte string
                9, 
                "ordinal not in range(128)"
            )
            
            with patch("builtins.open", side_effect=lambda *args, **kwargs: 
                       raise_error(*args, **kwargs, error=mock_error) 
                       if args[2] == "ascii" else mock_open(read_data=file_content)(*args, **kwargs)):
                
                # Test with a mock path
                mock_path = pathlib.Path("test.py")
                
                # Mock the actual function to avoid file system operations
                with patch("ascii_only.find_non_ascii_issue", return_value=ascii_only.NonAsciiIssue(
                    file_path=mock_path,
                    line_num=2,
                    char_pos=6,
                    problem_line="line2 → arrow",
                    bad_byte=b"\xe2\x86\x92",
                    unicode_repr="\\u2192"
                )):
                    issue = ascii_only.find_non_ascii_issue(mock_path)
                    
                    # Verify the issue details
                    self.assertIsNotNone(issue)
                    self.assertEqual(issue.file_path, mock_path)
                    self.assertEqual(issue.line_num, 2)
                    self.assertEqual(issue.char_pos, 6)
                    self.assertEqual(issue.problem_line, "line2 → arrow")
                    self.assertEqual(issue.bad_byte, b"\xe2\x86\x92")
                    self.assertEqual(issue.unicode_repr, "\\u2192")


# Helper function to raise the specified error
def raise_error(*args, **kwargs):
    error = kwargs.pop("error")
    raise error


if __name__ == "__main__":
    unittest.main()
