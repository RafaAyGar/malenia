import os
import malenia

def test():
    ## Results extraction tests
    results_test_path = malenia.__path__[0] + "/results_test.py"
    os.system(f"pytest {results_test_path}")