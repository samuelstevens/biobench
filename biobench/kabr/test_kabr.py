import pytest

from . import Video


def test_video_length_check():
    Video(0, ["f1"], [0])
    with pytest.raises(AssertionError):
        Video(1, ["f1"], [0, 0])
