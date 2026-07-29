from unittest.mock import patch

import app


def test_single_analysis_matches_gradio_outputs():
    with patch.object(app, "predict", return_value={"label": "positive", "confidence": 0.987}):
        result = app.analyze_single("A great flight")
    assert result == ("🟢 POSITIVE", "98.7%")


def test_empty_single_analysis_matches_gradio_outputs():
    assert app.analyze_single("   ") == ("⚠️ Please enter some text", "")
