import pytest


def test_streamlit_app_importable():
    pytest.importorskip("streamlit")
    import importlib

    mod = importlib.import_module("streamlit_app")
    assert hasattr(mod, "st")
