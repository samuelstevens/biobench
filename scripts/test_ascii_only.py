import io
import sys
import unittest
from unittest.mock import patch

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
            ("Hello "world"", 'Hello "world"'),  # smart quotes
            ("Hello 'world'", "Hello 'world'"),  # smart single quotes
            ("Hello…world", "Hello...world"),  # ellipsis
            ("Hello world", "Hello world"),  # non-breaking space
            ("Utö", "Uto"),  # o with umlaut
            ("Ylöstalo", "Ylostalo"),  # o with umlaut
            ("Kälviäinen", "Kalviainen"),  # a with umlaut
        ]
        
        for input_str, expected in test_cases:
            with self.subTest(input_str=input_str):
                result = ascii_only.fix_non_ascii(input_str)
                self.assertEqual(result, expected)
    
    def test_unicode_escape_representation(self):
        """Test that Unicode escape representation is correctly generated."""
        # Create a mock UnicodeDecodeError
        def create_mock_error(bad_bytes):
            return UnicodeDecodeError(
                'ascii', 
                bad_bytes, 
                0, 
                len(bad_bytes), 
                'ordinal not in range(128)'
            )
        
        test_cases = [
            # Input bytes, Expected Unicode escape
            (b'\xe2\x86\x92', "\\u2192"),  # → (right arrow)
            (b'\xc3\xb6', "\\u00f6"),      # ö (o with umlaut)
            (b'\xe2\x80\x93', "\\u2013"),  # – (en dash)
            (b'\xe2\x80\x94', "\\u2014"),  # — (em dash)
        ]
        
        # Mock open and print to capture output
        with patch('builtins.open'), patch('builtins.print'):
            for bad_bytes, expected_escape in test_cases:
                with self.subTest(bad_bytes=bad_bytes):
                    # Create a mock error
                    error = create_mock_error(bad_bytes)
                    
                    # Capture stdout
                    captured_output = io.StringIO()
                    sys.stdout = captured_output
                    
                    # Call the function with our mock error
                    with patch('builtins.open', return_value=io.StringIO("test")):
                        # We need to mock the file reading part
                        with patch.object(ascii_only, 'get_python_files', return_value=[]):
                            # Just to avoid actually processing files
                            try:
                                # This will fail but we just want to see the unicode escape generation
                                ascii_only.main(["dummy.py"])
                            except Exception:
                                pass
                    
                    # Reset stdout
                    sys.stdout = sys.__stdout__
                    
                    # Check if the expected escape sequence is in the output
                    self.assertIn(expected_escape, captured_output.getvalue())


if __name__ == '__main__':
    unittest.main()
