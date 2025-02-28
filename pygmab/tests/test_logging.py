import gmab
from pytest import CaptureFixture, LogCaptureFixture


def test_get_logger(caplog: LogCaptureFixture) -> None:
    logger = gmab.logging.get_logger("gmab.foo")

    logger.info("hello")
    assert "hello" in caplog.text  # Checks logging with a simple example

    logger.debug("bye")
    assert "bye" not in caplog.text  # DEBUG is not displayed per default


def test_set_level(caplog: LogCaptureFixture) -> None:
    logger = gmab.logging.get_logger("gmab.foo")

    gmab.logging.set_level(gmab.logging.DEBUG)
    logger.debug("debug_msg")
    assert "debug_msg" in caplog.text  # level is set to DEBUG

    gmab.logging.set_level(gmab.logging.CRITICAL)
    logger.error("error_msg")
    assert "error_msg" not in caplog.text  # level is set to CRITICAL


def test_disable(capsys: CaptureFixture) -> None:
    logger = gmab.logging.get_logger("gmab.foo")

    gmab.logging.disable()
    logger.info("hello")
    assert "hello" not in capsys.readouterr().err  # Logging is disabled

    gmab.logging.enable()
    logger.info("bye")
    assert "bye" in capsys.readouterr().err  # Logging is enabled
