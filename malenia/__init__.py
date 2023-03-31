__version__ = '0.0'
__all__ = [
    "check_condor_errors",
    "test",
    "results"
]

from malenia.util_checkers import check_condor_errors
from malenia.util_tests import test
from malenia.results import results