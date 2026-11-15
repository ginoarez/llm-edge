from config_validation import validate_config


def test_valid():
    assert validate_config({"model": "x", "quant": "Q4"}) is True


def test_unknown_key():
    try:
        validate_config({"model": "x", "typo": 1})
        assert False
    except ValueError:
        assert True
