import natex


def test_version():
    import re

    assert re.fullmatch(r"\d+\.\d+\.\d+(?:\.\w+)?", natex.__version__)


def test_kink_estimators_are_public():
    assert callable(natex.regression_kink)
    assert callable(natex.difference_in_kinks)
