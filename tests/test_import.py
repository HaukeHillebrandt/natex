import natex


def test_version():
    assert natex.__version__.startswith("0.1")


def test_kink_estimators_are_public():
    assert callable(natex.regression_kink)
    assert callable(natex.difference_in_kinks)
