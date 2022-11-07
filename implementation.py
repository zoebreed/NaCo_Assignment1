"""NACO assignment 22/23.

By Björn Keyser, Jimmy Oei, and Zoë Breed

This file contains the skeleton code required to solve the first part of the 
assignment for the NACO course 2022. 

You can test your algorithm by using the function `test_algorithm`. For passing A1,
your GA should be able to pass the checks in this function for dimension=100.

## Installing requirements
You are encouraged to use a virtual environment. Install the required dependencies 
(via the command line) with the following command:
    pip install ioh>=0.3.3
"""

import random
import shutil
import numpy as np

import ioh


class GeneticAlgorithm:
    """An implementation of the Genetic Algorithm."""

    def __init__(self, budget: int) -> None:
        """Construct a new GA object.

        Parameters
        ----------
        budget: int
            The maximum number objective function evaluations
            the GA is allowed to do when solving a problem.

        M: int
            The number of individuals in set/array

        Notes
        -----
        *   You can add more parameters to this constructor, which are specific to
            the GA or one of the (to be implemented) operators, such as a mutation rate.
        """
        # deze moet je kunnen aanpassen
        #offspring size, crossover rate
        self.mutation_rate = 0.5
        self.budget = budget

        #Je moet ook kunnen aanpassen welke mutatie je doet

        #Wat we nog kunnen toevoegen = offspring siZe

        self.pop_size = 1000

    def __call__(self, problem: ioh.problem.Integer) -> ioh.IntegerSolution:
        """Run the GA on a given problem instance.

        Parameters
        ----------
        problem: ioh.problem.Integer
            An integer problem, from the ioh package. This version of the GA
            should only work on binary/discrete search spaces.

        Notes
        -----
        *   This is the main body of you GA. You should implement all the logic
            for this search algorithm in this method. This does not mean that all
            the code needs to be in this method as one big block of code, you can
            use different methods you implement yourself.

        *   For now there is a random search process implemented here, which
            is a placeholder, and just to show you how to call the problem
            class.
        """

        # print(problem.log_info)
        # print(problem.meta_data)
        # print(problem.state)
        print("optimum:",(problem.optimum), problem.optimum.y)
        # print("n_variables", problem.meta_data.n_variables)

        #initialize
        pop = self.initialize_population(n=problem.meta_data.n_variables)

        #evaluate
        fitnesses = self.calculate_fitness(pop, problem)

        #terminate if optimum reached or budget exceeded
        terminated = False
        gen = 1

        #budget = 50k * dim
                #= 10 bits -> 50000 problem() calls
                # pop_size = 1000
                # -> 50 generations

    
        for e in range(self.budget - self.pop_size):
            if e % self.pop_size == 0:
                new_pop, new_fit = self.generate_population(fitnesses, pop, problem)
                pop, fitnesses = self.select_population(pop, new_pop, fitnesses, new_fit, problem)
                # fitnesses = self.calculate_fitness(pop, problem)
                gen = gen + 1

                if problem.state.current_best.y == problem.optimum.y:
                    # print(f"found")
                    break

        print(f"curr best: {problem.state.current_best}")
        

        # for evaluation in range(self.budget):
            #Create initial pop  --> check
            
            #Calculate fitness --> check 

            #Generate new population
                # Select 2 individuals --> check nu nog random
                # Crossover
                # Mutate

            # Check if termination criteria is satisfied

            # Do selection (-> new population)


            # for i in range(population_size):
            #   population.append(first_generation(toppings))
            # population = sorted(population, reverse = True, key = lambda r: r[-1])
            # population = initialize_population(population)

            # for i in range(num_generations):
            # if not terminated:
            #     R = generate_recipes(population_size, population)
            #     population = select_population(population, R)
            #     all_fitnesses = [(population[i][-1]) for i in range(population_size)



        return problem.state.current_best


    def generate_population(self, fitnesses, pop, problem):
        """ generates a new population

        Parameters
        ----------
        fitnesses: liste
            List with all of the fitnes values of the population
        
        pop:  list
            List containing the population
        
        problem: #
            ###

        Notes
        -----
        *   Sorts the old population and new population based upon their fitness.
            The upper half of each of the populations are then added together to create a new
            population. 
        """
        new_pop = []
        while len(new_pop) < self.pop_size:
            p1 = self.select_individual(fitnesses, pop)
            p2 = self.select_individual(fitnesses, pop)
            child = self.crossover(p1, p2)
            self.mutation(child)
            new_pop.append(child)

        new_fitnesses = self.calculate_fitness(new_pop, problem)
        return new_pop, new_fitnesses
        
    def select_population(self, pop, new_pop, fit, new_fit, problem):
        """ selects new population

        Parameters
        ----------
        #: ##
            ####

        Notes
        -----
        *   Sorts the old population and new population based upon their fitness.
            The upper half of each of the populations are then added together to create a new
            population. 
        """
        
        pop_old = [x for _,x in sorted(zip(fit, pop), reverse=True, key=lambda pair: pair[0])]
        pop_new = [x for _,x in sorted(zip(new_fit, new_pop), reverse=True, key=lambda pair: pair[0])]
        pop = pop_old[0:len(pop_old)//2] + pop_new[0:len(pop_new)//2]
        fitnesses = self.calculate_fitness(pop, problem)

        return pop, fitnesses
       
        
    def initialize_population(self, n):
        """Generates a randomly initialized population

        Parameters
        ----------
        n: int
            The dimensionality of the search space

        Notes
        -----
        *   A solution candidate set is created, consisting of M number of solution
            candidates of the form: solution candidate x_i = (x_1, x_2, ..., x_n) with
            i ∈ {1 . . . M}, which are all initialized randomly.
        """

        pop = []
        for _ in range(self.pop_size):
            pop.append(np.random.randint(0, 2, n))
        return pop

    def calculate_fitness(self, pop, problem):
        """ Calculates the fitness of a population

        Parameters
        ----------
        problem: ###
         

        pop: list
         List containing the population

        Notes
        -----
        *   
        """
        fitnesses = []
        for popi in pop:
                yi = problem(popi)
                fi = yi
                fitnesses.append(fi)
        
        return fitnesses


    def select_individual(self, fitnesses, pop):
        """ Implements selection of individu

        Parameters
        ---------- 
        fitnesses: list
         List with all of the fitnes values of the population

        pop: list
         List containing the population

        Notes
        -----
        *   Implements Roulette Wheel selection of individuals based on their fitness
        """

        sorted_fitpop = sorted(zip(fitnesses, pop), reverse=True, key=lambda pair: pair[0])

        sum_fitness = abs(int(sum(fitnesses)))
        f = random.randint(0, sum_fitness)
        for fit, popi in sorted_fitpop:
            if f < abs(fit):
                return popi
            f -= abs(fit)
        return sorted_fitpop[-1][1]       

        
    def crossover(self, p1, p2):
        """Implements the crossover fuction. 
        
        Parameters
        ----------
        n: int
         The dimensionality of the search space
        p1: list
          genotype 1
        p2: list 
          genotype 2

        Notes
        -----
        *   Takes two parents and combines them by choosing a point 
            on each genotype (bitstring) to split each list intwo two, and joing the first sublist from 
            one genotype with the second sublist of the second genotype.
        """
        split = random.randint(1, len(p1 -1))
        p1a = p1[0:split]
        p2b = p2[split:]
        child = np.append(p1a, p2b)
        return child


    def mutation(self, individu):
        """ Mutates an individu

        Parameters
        ---------- 
        individu: ###
            One bit string

        Notes
        -----
        *   The mutation operator changes an individu using 2 different types of mutation.
            (1) point mutation: by flipping the bit, (2) swapping two bits with eachother.
        """
        mut = random.randint(0, 1)

        if (mut < self.mutation_rate):
            m = random.randint(0, 1)
            n = len(individu)
            if m == 0: #point mutation
                j = random.randint(0, n - 1)
                individu[j] = 1 - individu[j]
            elif m == 1: #swap mutation
                j = random.randint(0, n - 1)
                k = random.randint(0, n - 1)
                tmp = individu[j]
                individu[j] = individu[k]
                individu[k] = tmp
            ##insert mutation (pick two bits at random, move the second to follow the first
            # shifting the rest)
            # elif m == 2: 
            #     j = random.randint(0, n - 2)
            #     k = random.randint(j, n - 1)
            #     individu[k] = individu[j+1]
            #     return

            ## inversion mutation (pick to bits at random and invert substring between them)
            # else:
            #     j = random.randint(0, n - 2)
            #     k = random.randint(j, n - 1)
            #     for i in range (k-j):
            #       individu[]

        return
    
    
def test_algorithm(dimension, instance=9):
    """A function to test if your implementation solves a OneMax problem.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.

    instance: int
        The instance of the problem. Trying different instances of the problem,
        can be interesting if you want to check the robustness, of your GA.
    """

    budget = int(dimension * 5e3)
    problem = ioh.get_problem("OneMax", instance, dimension, "Integer")
    ga = GeneticAlgorithm(budget)
    solution = ga(problem)

    print("GA found solution:\n", solution)

    assert problem.state.optimum_found, "The optimum has not been reached."
    assert problem.state.evaluations <= budget, (
        "The GA has spent more than the allowed number of evaluations to "
        "reach the optimum."
    )

    print(f"OneMax was successfully solved in {dimension}D.\n")


def collect_data(dimension=100, nreps=1):
    """OneMax + LeadingOnes functions 10 instances.

    This function should be used to generate data, for A1.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.

    nreps: int
        The number of repetitions for each problem instance.
    """

    budget = int(dimension * 5e2)
    suite = ioh.suite.PBO([1, 2], list(range(1, 11)), [dimension])
    logger = ioh.logger.Analyzer(algorithm_name="GeneticAlgorithm")
    suite.attach_logger(logger)

    for problem in suite:
        print("Solving: ", problem)

        for _ in range(nreps):
            ga = GeneticAlgorithm(budget)
            ga(problem)
            problem.reset()
    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")


if __name__ == "__main__":
    # Simple test for development purpose
    # test_algorithm(1)

    # Test required for A1, your GA should be able to pass this!
    # test_algorithm(100)

    # If your implementation passes test_algorithm(100)
    collect_data(100)
