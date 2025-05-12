import pytest

from . import ascii_only


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("Hello world", "Hello world"),  # ASCII only
        ("Hello--world", "Hello--world"),  # em dash
        ("Hello-world", "Hello-world"),  # en dash
        ("Hello -> world", "Hello -> world"),  # right arrow
        ("Hello...world", "Hello...world"),  # ellipsis
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
        (b"\xe2\x86\x92", "\\u2192"),  # -> (right arrow)
        (b"\xe2\x80\x93", "\\u2013"),  # - (en dash)
        (b"\xe2\x80\x94", "\\u2014"),  # -- (em dash)
    ],
)
def test_get_unicode_escape(bad_bytes, expected_escape):
    """Test that Unicode escape representation is correctly generated."""
    result = ascii_only.get_unicode_escape(bad_bytes)
    assert result == expected_escape
