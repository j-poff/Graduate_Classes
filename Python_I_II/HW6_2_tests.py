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

test_bug()
test_bug_fitness()
print("tests completed successfully!")