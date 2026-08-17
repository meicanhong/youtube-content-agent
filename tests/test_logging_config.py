import logging

from youtube_content_agent.logging_config import configure_logging


def test_verbose_logging_does_not_expose_third_party_request_bodies() -> None:
    configure_logging(verbose=True)

    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("openai").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
