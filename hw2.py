import pandas as pd
import math
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import statistics

#%% Create Distance Matrix
city_map = pd.read_csv('hw2.csv',header=None,names=['X', 'Y'])
city_distance_matrix = []
for index, row in city_map.iterrows():
    distances_row = []
    xy_coor_1 = (city_map.loc[index,'X'], city_map.loc[index,'Y'])
    for index1, row1 in city_map.iterrows():
        xy_coor_2 = (city_map.loc[index1,'X'], city_map.loc[index1,'Y'])
        distance = math.sqrt((xy_coor_2[0] - xy_coor_1[0]) ** 2 + 
                             (xy_coor_2[1] - xy_coor_1[1]) ** 2)
        distances_row.append(distance)
    city_distance_matrix.append(distances_row)
city_distance_matrix= np.array(city_distance_matrix)

#%% Functions 

def get_distance(sequence,city_distance_matrix):
    total_distance = 0
    for i in range(len(sequence)):
        city_1 = sequence[i]
        if i+1 == len(sequence):
            city_2 = sequence[0] # go back to first city
        else:
            city_2 = sequence[i+1]
        total_distance += city_distance_matrix[city_1,city_2]
        
    return total_distance

def generate_random_solution():
    sequence = [i for i in range(25)]
    random.shuffle(sequence)
    return sequence

def mutate_solution(solution,mutation_number=1):
    number = len(solution)
    for _ in range(mutation_number):
        # Pick adjacent indicies to switch (mutate)
        mutate_index_1 = random.randint(0,number-1)
        if mutate_index_1 == 0:
            mutate_index_2 = 1
        elif mutate_index_1 == number-1:
            mutate_index_2 = number-2
        else:
            if random.random() > 0.5: # chance to pick either side
                mutate_index_2 = mutate_index_1 + 1
            else:
                mutate_index_2 = mutate_index_1 - 1
        # Store city number
        city_1 = solution[mutate_index_1]
        city_2 = solution[mutate_index_2]
        # Switch adjacent city order
        solution[mutate_index_1] = city_2
        solution[mutate_index_2] = city_1
    return solution

def pick_k_states(solution_space):
    num_k = int(len(solution_space)/2) #count / 2
    # Calculate probabilities with softmax
    distances = np.array([solution[1] for solution in solution_space])
    max_distance = np.max(distances)  # Normalize to prevent overflow
    e_distances = np.exp(-distances + max_distance)
    probabilities = e_distances / np.sum(e_distances) # softmax
    
    selected_indicies = np.random.choice(len(solution_space),num_k,
                                         replace=False,p=list(probabilities))
    selected_solutions = [solution_space[i] for i in selected_indicies]
    return selected_solutions

def plot_path(df,order,title='City Path'):
    plt.figure(figsize=(6, 6))
    plt.plot(df['X'], df['Y'], marker='o', markersize=8, linestyle='', 
             color='b')
    
    # Loop through to connect points in order
    for i in range(len(order) - 1):
        start_idx = order[i]
        end_idx = order[i + 1]
        start_point = df.iloc[start_idx]
        end_point = df.iloc[end_idx]
        plt.plot([start_point['X'], end_point['X']], [start_point['Y'], 
                                                      end_point['Y']], 'r-')
    
    # Connect the last point to the starting point
    start_idx = order[-1]
    end_idx = order[0]
    start_point = df.iloc[start_idx]
    end_point = df.iloc[end_idx]
    plt.plot([start_point['X'], end_point['X']], [start_point['Y'], 
                                                  end_point['Y']], 'r-')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.show()
    return

def simulated_annealing(city_distance_matrix,iterations,alpha=.97,
                        intial_temp=None,plot_degrading_prob=True):
    ## Calculate intial temp
    if intial_temp == None: 
        # Method from DOI 10.1023/B:COAP.0000044187.23143.bd
        average_difference = []
        for _ in range(10):
            sol1 = generate_random_solution()
            sol1 = get_distance(sol1,city_distance_matrix)
            sol2 = generate_random_solution()
            sol2 = get_distance(sol2,city_distance_matrix)
            average_difference.append(abs(sol2-sol1))
        avg = sum(average_difference)/ len(average_difference)
        intial_temp = avg / math.log(2)
        #print(intial_temp)
        
    ## Intialize time and values
    start_time = time.time()
    current_solution = generate_random_solution()
    current_solution_distance = get_distance(current_solution,
                                             city_distance_matrix)
    probs = [] # For plotting later
    degrading_temp = intial_temp # Set intial temp
    probability_to_accept_anyway = 1 # For intial graphing purposes
    
    ## Do simulated annealing optimization
    for iteration in range(iterations): # Loop through iterations
        new_solution = generate_random_solution() # New solution
        new_solution_distance = get_distance(new_solution,city_distance_matrix)
        delta_e = new_solution_distance - current_solution_distance
        if delta_e < 0: # If new solution is better, accept.
            current_solution = new_solution
            current_solution_distance = new_solution_distance
            probs.append(probability_to_accept_anyway)
        else: # If not better, calculate probability to accept based on
              # temp and how bad the new solution is
            probability_to_accept_anyway = math.exp(-delta_e/degrading_temp)
            # The equation above was given in class
            probs.append(probability_to_accept_anyway) # graphing
            if random.uniform(0, 1) < probability_to_accept_anyway:
                # see if we should randomly accept bad solution
                current_solution = new_solution
                current_solution_distance = new_solution_distance
            
        degrading_temp = degrading_temp * alpha # update temperature
    
    ## Calculate end time and plot if wanted
    end_time = time.time() - start_time
    if plot_degrading_prob:
        iteration_list = [i for i in range(iterations)]
        plt.scatter(iteration_list, probs)
        plt.xlabel('Iterations')
        plt.ylabel('Probability to Accept')
        plt.title('Degrading Probabilities')
    return current_solution, current_solution_distance, end_time

def evolutionary_algorithm(city_distance_matrix,iterations,k_states,
                           mutation_number=1):
    
    ## Intialize time and values
    start_time = time.time()
    k_solutions = []
    for i in range(k_states):
        k_solution = generate_random_solution()
        k_distance = get_distance(k_solution, city_distance_matrix)
        k_solutions.append([k_solution,k_distance])

    ## Mutate solutions, pick from best, mutate again.
    for _ in range(iterations):
        # Mutate solutions
        mutated_k_solutions = []
        for solution in k_solutions:
            new_mutated_solution = mutate_solution(solution[0],mutation_number)
            new_mutated_distance = get_distance(new_mutated_solution,
                                                city_distance_matrix)
            mutated_k_solutions.append([new_mutated_solution,
                                        new_mutated_distance])
        k2_solutions = k_solutions + mutated_k_solutions
        # New solutions picked, smaller distances more likely to be chosen
        k_solutions = pick_k_states(k2_solutions) 
        
    end_time = time.time() - start_time # End time
    ## Return best solution and distance
    final_distance = [dis[1] for dis in k_solutions]
    best_solution = k_solutions[final_distance.index(min(final_distance))][0]
    return best_solution, min(final_distance), end_time

def pop_stochastic_beam_search(city_distance_matrix,iterations,k_states):
    ## Intialize time and values
    start_time = time.time()
    k_solutions = []
    for i in range(k_states):
        k_solution = generate_random_solution()
        k_distance = get_distance(k_solution, city_distance_matrix)
        k_solutions.append([k_solution,k_distance])
        
    ## New solutions, pick from best, generate new ones again. 
    for _ in range(iterations):
        # Generate new solutions
        new_k_solutions = []
        for solution in k_solutions:
            new_random_solution = generate_random_solution()
            new_random_distance = get_distance(new_random_solution,
                                                city_distance_matrix)
            new_k_solutions.append([new_random_solution,new_random_distance])
        k2_solutions = k_solutions + new_k_solutions
        # New solutions picked, smaller distances more likely to be chosen
        k_solutions = pick_k_states(k2_solutions) 
        
    end_time = time.time() - start_time # End time
    ## Return best solution and distance
    final_distance = [dis[1] for dis in k_solutions]
    best_solution = k_solutions[final_distance.index(min(final_distance))][0]
    return best_solution, min(final_distance), end_time

def solution_analysis(solutions,city_map,name):
    ## Simulated Annealing
    best_distances = [x[1] for x in solutions]
    best_map = solutions[best_distances.index(min(best_distances))][0]
    times = [x[2] for x in solutions]
    average = sum(best_distances) / len(best_distances)
    print(f'______{name}______')
    print(f'Best Distance: {min(best_distances):.2f}')
    print(f'Average Distance: {average:.2f}')
    print(f'Average Time: {sum(times) / len(times):.2f} seconds')
    print(f'Median Solution: {statistics.median(best_distances):.2f}')
    print(f'St Dev Solution: {statistics.stdev(best_distances):.2f}')
    print(f'Range Solution: {max(best_distances)-min(best_distances):.2f}')
    graph_title = name + ' Best Path'
    plot_path(city_map,best_map,graph_title)
    return
    
#%% Testing
sa_solutions = []
ea_solutions = []
pop_solutions = []
progress = 0

for _ in range(10):
    progress += 1
    print(f'Running {progress}/10 iteration')
    
    ### Simulated Annealing
    solution, distance, end_time= simulated_annealing(city_distance_matrix,
                                                    iterations=50000,
                                                    alpha=.999,
                                                    plot_degrading_prob=False)
    sa_solutions.append([solution,distance,end_time])
    
    ### Evolutionary Algorithm
    solution, distance, end_time = evolutionary_algorithm(city_distance_matrix,
                                                          iterations=50000,
                                                          k_states=50,
                                                          mutation_number=4) 
    ea_solutions.append([solution,distance,end_time])
    
    ### Stochastic Beam Search
    solution, distance, end_time = pop_stochastic_beam_search(city_distance_matrix,
                                                          iterations=50000,
                                                          k_states=50) 
    pop_solutions.append([solution,distance,end_time])
    
solution_analysis(sa_solutions,city_map,'Simulated Annealing')
solution_analysis(ea_solutions,city_map,'Evolutionary Algorithm')  
solution_analysis(pop_solutions,city_map,'Stochastic Beam Search')  

