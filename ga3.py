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
    CHROMOSOME_LENGTH = 7
    POP_SIZE = 150
    MAXIMUM_GEN = 150
    PROBABILITY_CROSSOVER = 0.6
    PROBABILITY_MUTATION = 0.4
    TOURNAMENT_SIZE = 3
    FEATURE = 6

    population = initialize_population(POP_SIZE, CHROMOSOME_LENGTH, FEATURE)
    train_set = read_data(train_data)
    fitness = population_fitness(population, train_set)

    fitter_fitness = np.max(fitness)
    mean_progress = []
    mean_fitness = np.mean(fitness)
    mean_progress.append(mean_fitness)
    percentage = np.max(fitness) / 1000 * 100

    print("population fitness: ", percentage)
    print("population fitness score: ", fitter_fitness)

    best_fitness_progress = []

    best_fitness_progress.append(fitter_fitness)
    current_pop = population
    for generation in range(MAXIMUM_GEN):
        print("Generation: ", generation)
        new_population = []
        clone_population = np.copy(current_pop)
        selected_pop = tournament_selection(clone_population.tolist(), TOURNAMENT_SIZE, train_set)

        for ind1, ind2 in zip(selected_pop[::2], selected_pop[1::2]):
            random_number = random.random()
            if random_number < PROBABILITY_CROSSOVER:
                offspring_1, offspring_2 = crossover(ind1, ind2)
                new_population.append(offspring_1)
                new_population.append(offspring_2)

        for individual in new_population:
            if random.random() < PROBABILITY_MUTATION:
                new_individual = mutation(individual, FEATURE)
                new_population.append(new_individual)

        fitness = population_fitness(current_pop, train_set)

        fitter_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)
        percentage = np.max(fitness) / 1000 * 100
        print("population fitness (train data): ", percentage)
        print("population fitness score (train data): ", fitter_fitness)

        best_fitness_progress.append(fitter_fitness)
        mean_progress.append(mean_fitness)

        if len(new_population) < POP_SIZE:
            args_fitness = np.argsort(fitness)
            new_population.extend(
                np.array(current_pop)[args_fitness[::-1]][:len(current_pop) - len(new_population)].tolist())
        current_pop = new_population

    print("best individual")
    best_ind = best_individual(current_pop, fitness)
    print(best_ind)
    print("best fitness in test data")
    test = read_data(test_data)
    fit = calculate_fitness(test, best_ind)
    print(fit)


    plt.plot(best_fitness_progress, c='b', marker="^", label="training")
    plt.plot(fit, c='g', marker="^", label="test")
    plt.plot(mean_progress,  c='r', marker="^", label="mean")
    plt.legend(loc=1)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()


if __name__ == '__main__':
    main()
