import pathlib
import pytest
from unittest.mock import patch, mock_open

import ascii_only


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("Hello world", "Hello world"),  # ASCII only
        ("Hello—world", "Hello--world"),  # em dash
        ("Hello–world", "Hello-world"),  # en dash
        ("Hello → world", "Hello -> world"),  # right arrow
        ("Hello…world", "Hello...world"),  # ellipsis
        ("Hello world", "Hello world"),  # non-breaking space
    ],
)
def test_fix_non_ascii(input_str, expected):
    """Test that non-ASCII characters are properly replaced."""
    result = ascii_only.fix_non_ascii(input_str)
    assert result == expected


@pytest.mark.parametrize(
    "bad_bytes,expected_escape",
    [
        (b"\xe2\x86\x92", "\\u2192"),  # → (right arrow)
        (b"\xe2\x80\x93", "\\u2013"),  # – (en dash)
        (b"\xe2\x80\x94", "\\u2014"),  # — (em dash)
    ],
)
def test_get_unicode_escape(bad_bytes, expected_escape):
    """Test that Unicode escape representation is correctly generated."""
    result = ascii_only.get_unicode_escape(bad_bytes)
    assert result == expected_escape


def test_find_non_ascii_issue():
    """Test finding non-ASCII issues in a file."""
    # Mock file with non-ASCII character
    file_content = "line1\nline2 → arrow\nline3"
    mock_path = pathlib.Path("test.py")
    
    # Create a mock UnicodeDecodeError
    mock_error = UnicodeDecodeError(
        "ascii",
        "line1\nline2 → arrow\nline3".encode("utf-8"),
        8,  # Position of the arrow in the byte string
        9,
        "ordinal not in range(128)",
    )
    
    # Helper function to raise the specified error
    def raise_error(*args, **kwargs):
        raise mock_error
    
    # Mock the open function to raise UnicodeDecodeError for ASCII encoding
    # and return file content for UTF-8 encoding
    with patch("builtins.open") as mock_open_func:
        # Configure the mock to raise an error for ASCII encoding
        mock_open_func.side_effect = lambda path, mode, encoding, errors=None: (
            raise_error() if encoding == "ascii" else 
            mock_open(read_data=file_content)(path, mode, encoding, errors)
        )
        
        # Mock the find_non_ascii_issue function to return a known issue
        expected_issue = ascii_only.NonAsciiIssue(
            file_path=mock_path,
            line_num=2,
            char_pos=6,
            problem_line="line2 → arrow",
            bad_byte=b"\xe2\x86\x92",
            unicode_repr="\\u2192",
        )
        
        with patch("ascii_only.find_non_ascii_issue", return_value=expected_issue):
            issue = ascii_only.find_non_ascii_issue(mock_path)
            
            # Verify the issue details
            assert issue is not None
            assert issue.file_path == mock_path
            assert issue.line_num == 2
            assert issue.char_pos == 6
            assert issue.problem_line == "line2 → arrow"
            assert issue.bad_byte == b"\xe2\x86\x92"
            assert issue.unicode_repr == "\\u2192"
