from multilingual_sentiment_analysis.config import LANGUAGES
from multilingual_sentiment_analysis.preprocess import clean_text


def test_clean_text_preserves_non_ascii_text():
    assert clean_text("  Merci @airline — الخدمة ممتازة! #great  ") == "merci — الخدمة ممتازة! great"


def test_multilingual_configuration_covers_expected_languages():
    assert len(LANGUAGES) == 8
    assert {"arabic", "english", "french", "german", "hindi", "italian", "portuguese", "spanish"} == set(LANGUAGES)
