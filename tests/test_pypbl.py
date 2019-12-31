from pypbl import __version__
from dunamai import Version


def test_version():
    assert __version__ == Version.from_git().serialize()
