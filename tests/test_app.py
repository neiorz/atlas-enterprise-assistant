"""UI-boundary tests: every failure reaches the user as a fixable sentence.

NFR-D3 forbids stack traces in the UI, and a bare "something went wrong" is
only marginally better than one. These pin the two behaviours that make the
boundary useful: common failures explain *what to do*, and unknown failures
still never leak a traceback.
"""

from __future__ import annotations

from src.judge import get_judge  # noqa: F401 — proves imports stay key-free


def test_qdrant_lock_gets_an_actionable_message():
    """The single-process index lock is the most common integration hiccup."""
    from app import _friendly_error

    exc = RuntimeError(
        "Storage folder /tmp/x/data/qdrant_index is already accessed by "
        "another instance of Qdrant client."
    )
    msg = _friendly_error(exc)
    assert "another process" in msg
    assert "tests/evaluate.py" in msg, "should name the usual culprit"
    assert "Traceback" not in msg and 'File "' not in msg


def test_rate_limit_gets_a_wait_message():
    from app import _friendly_error

    class RateLimitError(Exception):
        pass

    msg = _friendly_error(RateLimitError("Error code: 429 - rate limit"))
    assert "free tier" in msg.lower()
    assert "wait" in msg.lower()


def test_unknown_failure_is_friendly_and_leaks_nothing():
    from app import _friendly_error

    msg = _friendly_error(ValueError("boom /home/user/secret/path.py line 42"))
    assert "ValueError" in msg  # enough to correlate with run_logs
    assert "Traceback" not in msg
    assert "secret/path.py" not in msg  # never echo internals back to the UI


def test_api_key_guard_accepts_either_provider():
    from app import _api_key_ready
    from src.settings import settings

    original_g, original_r = settings.google_api_key, settings.groq_api_key
    try:
        settings.google_api_key = ""
        settings.groq_api_key = ""
        assert _api_key_ready() is False
        settings.groq_api_key = "gskSomeRealKey123"
        assert _api_key_ready() is True, "a Groq key alone must be enough"
        settings.groq_api_key = ""
        settings.google_api_key = "your-google-api-key-here"
        assert _api_key_ready() is False, "the .env.example placeholder is not a key"
        settings.google_api_key = "AIzaRealLookingKey123"
        assert _api_key_ready() is True
    finally:
        settings.google_api_key, settings.groq_api_key = original_g, original_r
