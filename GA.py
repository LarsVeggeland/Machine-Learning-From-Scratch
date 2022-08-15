#---------- Imported libraires---------

from random import randint, randrange, random



#---------- Utility functions ----------

def func(vars : list) -> float:
    x, y = vars[0], vars[1]
    return 25 * (y + x**2)**2 + (1 + x)**2


def create_population(size : int, num_bits : int,) -> list:
    # The samples themselves are represented as a bitstring (list of 1's and 0's)
    return [[randint(0, 1) for bit in range(num_bits*2)] for bit_string in range(size)]


def scores(population : list) -> list:
    # Returns a list with the scores  for all samples in the population
    return [func(xy) for xy in population]


def decode_bitstrings(bitstrings : list, bounds : list) -> list:
    # As the samples are represented as bitstrings there is a need for converting 
    # the bitstrings to numerical values within the specified bounds
    decoded_bstr = []

    # The largest possible value represented as a bitstring, i.e., all 1's
    max_val = 2**int(len(bitstrings[0])/2)-1

    for bitstring in bitstrings:
        # Easier to work with strings when casting back to int using base=2
        bts = ''.join([str(s) for s in bitstring])

        # The first value (x) in the bitstring
        x = bounds[0] + (int(bts[:int(len(bts)/2)], base=2)/max_val) * (bounds[1] - bounds[0]) 

        # The second value (y) in the bitstring
        y = bounds[0] + (int(bts[int(len(bts)/2):], base=2)/max_val) * (bounds[1] - bounds[0]) 

        decoded_bstr.append([x, y])

    return decoded_bstr


def select_parents(population : list, scores : list, depth : int) -> list:

    # Selecting a random sample as initial placeholder
    ran_index = randint(0, len(population)-1)
    winner = population[ran_index]

    # Will iterate over a list of randomly chosen indecies
    # depth determines how many indecies to check
    for i in [randrange(0, len(population)) for i in range(depth)]:
        winner = population[i] if scores[i] < scores[ran_index] else winner
    
    return winner


def crossover(parent1 : list, parent2 : list, crossover_proba : float, crossover_index : int = None) -> list:
    child1, child2 = parent1.copy(), parent2.copy()

    # Chance crossover per gene
    if crossover_index is None:
        for i in range(len(child1)):
            if random() <= crossover_proba:
                temp = child1[i]
                child1[i] = child2[i]
                child2[i] = temp
        

    # One point crossover of chromosomes 
    else:
        if random() < crossover_proba:
            child1 = parent1[:crossover_index] + parent2[crossover_index:]
            child2 = parent2[:crossover_index] + parent1[crossover_index:]

    return [child1, child2]


def mutation(bitstring : list, mutation_proba : float) -> list:
    return [1 - i * (random() < mutation_proba) for i in bitstring]
    #if random() < mutation_proba:
    #    return [1 - bit for bit in bitstring]
    #return bitstring


def GA(size : int, vector_size : int, maxgens : int, crossover_proba : float, mutation_proba, bounds : list, crossover_index : int = None):

    # Get the initial randomly generated population where the samples are represented as bitstrings
    pop_as_bitstrings = create_population(size=size, num_bits=vector_size)

    # Convert the population to numerical values
    pop_as_vals = decode_bitstrings(pop_as_bitstrings, bounds)

    # Get the scores/fitness of the initial population
    pop_scores = scores(population=pop_as_vals)

    # Just placeholder values
    best_score = pop_scores[0]
    best_vals = pop_as_vals[0]

    for generation in range(maxgens):
        
        # Find the x, y value pair with the best fitness in the current generation
        for i in range(size):

            if pop_scores[i] < best_score:
                best_score = pop_scores[i]
                best_vals = pop_as_vals[i]
                
        # Pseudo randomly select the percieved best parents for the next generation 
        best_samples = [select_parents(pop_as_bitstrings, pop_scores, 5) for i in range(size)]
        next_generation = []

        # Apply chance crossover
        for i in range(0, size, 2):
            parent1, parent2 = best_samples[i], best_samples[i+1]

            # Apply chance mutation
            for child in crossover(parent1, parent2, crossover_proba, crossover_index=crossover_index):
                next_generation.append(mutation(child, mutation_proba))

        # We now have the next generation of x,y pairs
        pop_as_bitstrings = next_generation
        pop_as_vals = decode_bitstrings(pop_as_bitstrings, bounds)
        pop_scores = scores(population=pop_as_vals)


    return [best_vals, best_score]



# The remaining functions are not part of the algorithm. The code is already set up to be run and
# test all the different generation size, crossover probabilities, etc... configurations requested



def print_results():
    
    print("\nWith one point crossover applied it the rate of crossover_proba")

    for gensize in [4, 10]:
        print()
        file = open(f"gensize{gensize}WithIndex.csv", "a")
        file.write("size, gens, crossover, mutation, x, y, fitness\n")
        for total_gens in [100, 1000]:
            for crossover_proba in [0.25, 0.5]:
                for mutation_proba in [0.01, 0.05, 0.25]:
                    res = GA(gensize, 32, total_gens, crossover_proba, mutation_proba, [-10, 10], crossover_index=int(gensize/2))

                    print(f"size: {gensize} gens: {total_gens} crossover: {crossover_proba} mutation: {mutation_proba} x: {res[0][0]} y: {res[0][1]} fitness: {res[1]}")
                    file.write(f"{gensize}, {total_gens}, {crossover_proba}, {mutation_proba}, {res[0][0]}, {res[0][1]}, {res[1]}\n")
        file.close()


    print("\n\nWith crossover_proba applied to each gene for crossover")

    for gensize in [4, 10]:
        file = open(f"gensize{gensize}.csv", "a")
        file.write("size, gens, crossover, mutation, x, y, fitness\n")
        print()
        for total_gens in [100, 1000]:
            for crossover_proba in [0.25, 0.5]:
                for mutation_proba in [0.01, 0.05, 0.25]:
                    res = GA(gensize, 32, total_gens, crossover_proba, mutation_proba, [-10, 10])

                    print(f"size: {gensize} gens: {total_gens} crossover: {crossover_proba} mutation: {mutation_proba} x: {res[0][0]} y: {res[0][1]} fitness: {res[1]}")
                    file.write(f"{gensize}, {total_gens}, {crossover_proba}, {mutation_proba}, {res[0][0]}, {res[0][1]}, {res[1]}\n")
        file.close()



def write_samples_to_file(filename : str, param_to_tweak : int = 1):
    with open(filename, "a") as file:
        for i in range(4, 101, 4):

            # Adjusting the maximum number of generations
            if param_to_tweak == 1:
                avgerage_for_gen_size = sum([GA(10, 32, i, 0.25, 0.05, [-10, 10])[1] for j in range(10)])/10
                file.write(str([avgerage_for_gen_size, i]) + "\n")

            # Adjusting the size of each generation
            elif param_to_tweak == 2: 
                avgerage_for_gen_size = sum([GA(i, 32, 100, 0.25, 0.05, [-10, 10])[1] for j in range(10)])/10
                file.write(str([avgerage_for_gen_size, i]) + "\n")

            # Adjusting the number of genes per sample
            elif param_to_tweak == 3: 
                avgerage_for_gen_size = sum([GA(10, 32, 100, 0.25, 0.05, [-10, 10])[1] for j in range(10)])/10
                file.write(str([avgerage_for_gen_size, i*2]) + "\n")

            elif param_to_tweak == 4:
                avgerage_for_gen_size = sum([GA(10, 32, 100, i/100, 0.05, [-10, 10])[1] for j in range(10)])/10
                file.write(str([avgerage_for_gen_size, i]) + "\n")

            elif param_to_tweak == 5:
                avgerage_for_gen_size = sum([GA(10, 32, 100, 0.25, i/100, [-10, 10])[1] for j in range(10)])/10
                file.write(str([avgerage_for_gen_size, i]) + "\n")

            else: 
                avgerage_for_gen_size = sum([GA(10, 32, i, 0.25, 0.25, [-10, 10])[1] for j in range(10)])/10
                file.write(str([avgerage_for_gen_size, i]) + "\n")
                avgerage_for_gen_size = sum([GA(10, 32, i, 0.25, 0.25, [-10, 10], 16)[1] for j in range(10)])/10
                file.write(str([avgerage_for_gen_size, i]) + "\n")

#write_samples_to_file("GA_crossover_max_gens.txt", 6)


def compare_crossover_proba_convergence():
    with open("GA_crossover_convergence.txt", "a") as file:
        for i in range(1, 41):

            avr = sum([sum([GA(10, 32, i, 0.25, 0.25, [-10, 10])[1] for j in range(25)])/25 for k in range(20)])/20
            file.write(str([avr, i]) + "\n")
            avr = sum([sum([GA(10, 32, i, 0.5, 0.25, [-10, 10])[1] for j in range(25)])/25 for k in range(20)])/20
            file.write(str([avr, i]) + "\n")

print_results()