#!/usr/bin/env python3
from typing import List
from sim_module import Bug

def test_bug() -> None:
    bugA: Bug = Bug(12, ["C", "A", "C", "T", "T"]) # Bug with id 12, genome CACGT
    counta: int = bugA.base_composition("A")
    countc: int = bugA.base_composition("C")
    countg: int = bugA.base_composition("G")
    assert counta == 1, "Error on bug_test 1"
    assert countc == 2, "Error on bug_test 2"
    assert countg == 0, "Error on bug_test 3"
    assert bugA.get_id() == 12, "Error on test 4"


def test_bug_fitness() -> None:
    bugA: Bug = Bug(1, ["C", "A", "C", "T", "T"])
    bugB: Bug = Bug(2, ["C", "G", "C", "A", "T"])
    bugC: Bug = Bug(3, ["A", "A", "A", "A", "T"])

    assert bugA.fitness() == 10, "Error on bug_test_fitness 1"
    assert bugB.fitness() == 9, "Error on bug_test_fitness 1"
    assert bugC.fitness() == 3, "Error on bug_test_fitness 1"


def test_bug_reproduce() -> None:
    bugA: Bug = Bug(1, ["C", "A", "C", "T", "T", "C", "G", "G", "T"])
    bugB: Bug = bugA.reproduce(0) # reproduce with no mutation
    assert bugA.fitness() == bugB.fitness(), "Error with test_bug_reproduce 1"

    bugC: Bug = bugA.reproduce(1.0) # reproduce with lots of mutation
    # chances of equal fitness: very tiny (but not 0; is this a good automated test?)
    assert bugA.fitness() != bugC.fitness(), "Probable error with test_bug_reproduce 2"
    
    # NOTE: the above line used to have a typo, it used to be:
    # assert bugA.fitness() != bugB.fitness(), "Probable error with test_bug_reproduce 2"
    # which certainly isn't correct ;) [who tests the tests?!?]


test_bug()
test_bug_fitness()
test_bug_reproduce()
print("tests completed successfully!")
