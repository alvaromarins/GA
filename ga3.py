import numpy as np
import random
import matplotlib.pyplot as plt
import textract

test_data = "data/test3.txt"
train_data = "data/train3.txt"

# Read the data from the data set txt file
def read_data(file_path):
    '''
    :param file_path:
    :return:
    '''
    text = textract.process(file_path)
    rows = text.decode("utf-8").split('\n')
    rows = rows[:-1]
    rows = [row.split(' ') for row in rows]
    return rows


# Generate a random binary encoded individual
def generate_solution(chromosome_length, feature):
    ind = [[sorted([random.random(),random.random()]) for i in range(feature)] + [random.choice([0,1])] for j in range(chromosome_length)]
    return ind


# Generate a random binary encoded individual with wildcards
def generate_solution_wildcards(chromosome_length, feature):
    '''
    :param chromosome_length:
    :param feature:
    :return:
    '''

    ind = [[random.choice(['#', sorted([random.random(), random.random()])]) for i in range(feature)] + [
        random.choice([0, 1])] for j in range(chromosome_length)]
    return ind


# Generate an initial population
def initialize_population(popSize, chromosome_length, feature):
    '''
    :param popSize:
    :param chromosome_length:
    :param feature:
    :return:
    '''

    # Cereate a list (population) by calling the generate_solution function, population size is user-defined
    population = [generate_solution_wildcards(chromosome_length, feature) for i in range(popSize)]
    return population


def calculate_fitness(data_set, individual):
    fitness = 0
    for element in data_set:
        match = True
        for rule in individual:
            if int(rule[-1]) != int(element[-1]):
                continue
            for j,feature_range in enumerate(rule[:-1]):
                if feature_range == "#":
                    continue
                if float(feature_range[0]) > float(element[j]) or float(feature_range[1]) <float(element[j]):
                    match = False
                    break
            if match == True :
              break
        if match:
            fitness += 1
    return fitness


# Select individuals by tournament selection
def tournament_selection(population, T, data_set):
    # Get population length
    popSize = len(population)

    # Empty list to append offsprings
    offspring = []
    for _ in range(popSize):
        # Randomly select individuals from the population, taking into consideration tournament size
        tournament = random.sample(population, T)
        # Compare which individual is the fittest
        fittest_individual = tournament[np.argmax(population_fitness(tournament, data_set))]
        offspring.append(fittest_individual)

    return offspring


# One-point crossover
def crossover(parent_1, parent_2):
    # Get the length of the chromosome
    chromosome_length = len(parent_1)

    # crossover point
    crossover_point = random.randint(1, chromosome_length - 2)

    # Create offsprings from crossover point and parents
    offspring_1 = np.row_stack((np.array(parent_1)[0:crossover_point], parent_2[crossover_point:]))
    offspring_2 = np.row_stack((np.array(parent_2)[0:crossover_point], parent_1[crossover_point:]))

    return offspring_1.tolist(), offspring_2.tolist()


# Mutation
def mutation(individual, feature):
    # randomly select the number of genes to mutate
    select_number_of_genes = random.randint(2, np.array(individual).shape[0])
    new_individual = np.copy(np.array(individual))

    # randomly select the indices to insert the new genes
    random_indices = random.sample(range(len(individual)), select_number_of_genes)

    for index in random_indices:
        # create a new random rule
        new_rule = [sorted([random.random(),random.random()]) for i in range(feature)] + [random.choice([0,1])]
        # the new individual is formed from a new random rule and selected in the random index
        new_individual[index, :] = new_rule

    return new_individual.tolist()


# Population size
def population_fitness(population, dataset):
    return np.array([calculate_fitness(dataset, individual) for individual in population])


def best_individual(population, fitness):
    best_fit = np.argmax(fitness)
    return population[best_fit]




def main():
    # User defined parameters
    CHROMOSOME_LENGTH = 5
    POP_SIZE = 200
    MAXIMUM_GEN = 100
    PROBABILITY_CROSSOVER = 0.7
    PROBABILITY_MUTATION = 0.25
    TOURNAMENT_SIZE = 3
    FEATURE = 6

    # Initialzize the population
    population = initialize_population(POP_SIZE, CHROMOSOME_LENGTH, FEATURE)
    # Read the data set
    training_data_set = read_data(train_data)
    # Calculate the fitness of the population
    fitness = population_fitness(population, training_data_set)

    # Get the best fitness in the population
    best_fitness = np.max(fitness)

    # Calculate mean, percetange and keep track of it
    mean_progress = []
    mean_fitness = np.mean(fitness)
    mean_progress.append(mean_fitness)
    percetange = best_fitness / 60 * 100

    print("population fitness % : ", percetange)
    print("population fitness score : ", best_fitness)

    # Append the best fitness to a list to keep track of it to plot it on the graph
    best_fitness_progress = []
    best_fitness_progress.append(best_fitness)

    current_pop = population

    # Loop the generations
    for generation in range(MAXIMUM_GEN):
        print("Generation: ", generation)
        # Create and empty population
        new_population = []
        clone_population = np.copy(current_pop)
        # Do tournament selection on the current population
        selected_pop = tournament_selection(clone_population.tolist(), TOURNAMENT_SIZE, training_data_set)

        for ind1, ind2 in zip(selected_pop[::2], selected_pop[1::2]):
            # Generate random number between 0 and 1
            random_number = random.random()
            # The chance of individual being crossovered based on the crossover probability
            if random_number < PROBABILITY_CROSSOVER:
                # Do crossover
                offspring_1, offspring_2 = crossover(ind1, ind2)
                # Append new offsprings to the new population
                new_population.append(offspring_1)
                new_population.append(offspring_2)

        # Loop the individuals in the new population
        for individual in new_population:
            # The probability of the individual being mutated based on mutation probability rate
            if random.random() < PROBABILITY_MUTATION:
                # Do mutation on the individual
                new_individual = mutation(individual)
                # Append individual to new appopulation
                new_population.append(new_individual)

        # Calculate the fitness of the new individual
        fitness = population_fitness(current_pop, training_data_set)
        # Calculate mean and percetange
        best_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)
        percetange = best_fitness / 60 * 100

        print("population fitness: ", percetange)
        print("population fitness score : ", best_fitness)

        # Append the result to the progress list
        best_fitness_progress.append(best_fitness)
        mean_progress.append(mean_fitness)

        if len(new_population) < POP_SIZE:
            args_fitness = np.argsort(fitness)
            new_population.extend(
                np.array(current_pop)[args_fitness[::-1]][:len(current_pop) - len(new_population)].tolist())
        current_pop = new_population

    # Read the train data
    test_data_set = read_data(train_data)
    # Find the best individual in the current population
    best_ind = best_individual(current_pop, fitness)
    # Calculate the fitness of the best individual from the training data using the test data
    best_fitness_test_data = calculate_fitness(test_data_set, best_ind)

    print("Best fitness in test data ", best_fitness_test_data)

    # Plot the best fitness score and the mean in the graph
    plt.plot(best_fitness_progress, c='b', marker="^", label="fitness test")
    plt.plot(mean_progress, c='r', marker="^", label="mean")
    plt.plot(best_fitness_test_data, c='g', marker="^", label="fitness training")
    plt.legend(loc=1)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()


if __name__ == '__main__':
    main()
