from course_generator.utils import estimate_reading_minutes


def test_estimate_reading_minutes_short():
    assert estimate_reading_minutes("hello world") == 1


def test_estimate_reading_minutes_longer():
    text = "word " * 400
    assert estimate_reading_minutes(text, wpm=200) == 2
