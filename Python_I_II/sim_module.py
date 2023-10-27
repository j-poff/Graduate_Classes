import random
#!/usr/bin/env python3
from typing import List, IO
import random
import io

class Bug:
    def __init__(self, id: int, genome: List[str]) -> None:
        self.id: int = id
        self.genome: List[str] = genome
        
    def get_id(self) -> int:
        return(self.id)
        
    def base_composition(self, base: str) -> int:
        counter: int = 0
        for i in self.genome:
            if i == base:
                counter += 1
        return counter
    
    def fitness(self) -> int:
        score: int = 0
        for i in self.genome:
            if i == "C" or i == "G":
                score += 2
            elif i == "T":
                score += 3
        return score
    
    def reproduce(self, mutation_prob: float) -> 'Bug':
        assert mutation_prob >= 0 and mutation_prob <= 1, "Bug reproduce error: mutation_prob not between 0 and 1" 
        new_genome: List[str] = []
        gene_list: List[str] = ["A", "C", "G", "T"]
        for i in self.genome:
            if random.uniform(0,1) < mutation_prob:
                new_genome.append(random.choice(gene_list))
            else:
                new_genome.append(i)
    
        new_id: int = random.randint(0, 1000000)
        offspring: Bug = Bug(new_id, new_genome)
        return offspring

def bug_fitness(b: Bug) -> float:
    result = b.fitness()
    return result

class Population:
    def __init__(self, text: str) -> None:
        self.text: str = text
        gene_text: IO = io.open(self.text, "r")
        genes: List = []
        for item in gene_text:
            item_stripped: str = item.strip()
            item_list: List = item.split(" ")
            bug1: Bug = Bug(item_list[0],item_list[1])
            genes.append(bug1)
        self.bugs: List[Bug] = genes
        
    def get_size(self) -> int:
        bug_list: List[Bug] = self.bugs
        count: int = len(bug_list)

        return count

    def mean_fitness(self) -> float:
        bug_list = self.bugs
        fitness_list: List[int] = []
        for item in bug_list:
            fitness_list.append(item.fitness())
        avg_fitness: float = sum(fitness_list) / len(fitness_list)

        return avg_fitness
    
    def grow(self, n: int, mutation_prob: float) -> None:
        bug_list = self.bugs
        for i in range(0,n):
            picked = random.choice(bug_list)
            new_one = picked.reproduce(mutation_prob)
            self.bugs.append(new_one)
            
    def cull_to_size(self, n: int):
        assert n <= len(self.bugs), "Cannot reduce population to less than 1"
        self.bugs.sort(key = bug_fitness)
        self.bugs.reverse()
        self.bugs = self.bugs[0:n]
        
        
#%%
popA: Population = Population("genomes_test.txt")
aaa = popA.cull_to_size(1)

def test_population() -> None:
    popA: Population = Population("genomes_test.txt")
    popA_size: int = popA.get_size()
    assert popA_size == 3, "Error on test_population 1"

test_population()
print("tests completed successfully!")

def test_population_mean() -> None:
     popA: Population = Population("genomes_test.txt")
     popA_mean: float = popA.mean_fitness()
     known_mean: float = (4*2 + 1*2 + 4*3) / 3     # 4 Cs, 1 G, 4 Ts (computed by hand for genomes_test.txt)
 
     # assert that the two calculations are equal (within an amount allowing for roundoff error)
     assert abs(popA_mean - known_mean) < 0.000001, "Error on test_population 1"
  

test_population_mean()
print("tests completed successfully!")

def test_population_grow() -> None:
    popA: Population = Population("genomes_test.txt")
    popA.grow(5, 0.1)         # grow by 5 individuals, with mutation probability 0.1
    assert popA.get_size() == 8, "Error on test_population_grow 1"   

  
test_population_grow()
print("tests completed successfully!")

def test_bug_fitness_func() -> None:
    bugA: Bug = Bug(1, ["C", "A", "C", "T", "T"])
  
    # check to see if our function computes the same fitness as the bug itself does
    assert bug_fitness(bugA) == bugA.fitness(), "Error with bug fitness function"


test_bug_fitness_func()
print("tests completed successfully!")

def test_population_cull() -> None:     
    popA: Population = Population("genomes_test.txt")     
    popA.grow(7, 0.1)         # grow by 7 individuals, with mutation probability 0.1
 
    ave_fitness_precull: float = popA.mean_fitness()    # average fitness before culling
 
    popA.cull_to_size(5)      # shrink down to 5 individuals
 
    ave_fitness_postcull: float = popA.mean_fitness()   # average fitness after culling
 
    # the new population should be of size 5, and have a higher average fitness
    assert popA.get_size() == 5, "Error on test_population_cull 1"
    assert ave_fitness_postcull >= ave_fitness_precull, "Error on test_population_cull 2"


test_population_cull()
print("tests completed successfully!")

pop: Population = Population("genomes.txt")

i: int
for i in range(0, 100):
   pop.grow(10, 0.1)  # grow by 10 individuals, 10% mutation rate
   pop.cull_to_size(100) # cull back down to 100
   # print the generation number, and mean fitness
   print(str(i) + "\t" + str(pop.mean_fitness()))