"""NACO assignment 22/23.

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

        self.budget = budget

        self.M = 1000 # Constant for now ("all of the parameters you use should be modifyable by the user")

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

        x = self.generate_population(n=problem.meta_data.n_variables)

        for evaluation in range(self.budget):
            # Do crossover

            # Do mutation

            # Calculate fitness

            # Check if termination criteria is satisfied

            # Do selection (-> new population)


        return problem.state.current_best

    def generate_population(self, n):
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

        x = []
        for i in range(self.M):
            x.append(random.choices((0, 1), n))

        return x
        
    def select_individual(self):
        """Implements Roulette Wheel selection of individuals based on their fitness

        Parameters
        ----------
        n: int
         The dimensionality of the search space

        Notes
        -----
        *   
        """
        #sum_fitness = 
        # f = random.randint(0, int(sum_fitness))
        # for individu in self.M:
        #     if f < fitness:
        #         return individu
        # return 
        
    def crossover(self, p1, p2):
        """Implements the crossover fuction. Takes two parents and combines them by choosing a point 
        on each genotype (bitstring) to split each list intwo two, and joing the first sublist from 
        one genotype with the second sublist of the second genotype.
        

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
        *   
        """
        chance = random.randint(0,1)
        parent1 = random.randint(1, len(p1))
        parent2 = random.randint(1, len(p2))
        p1a = p1[0:parent1]
        p2b = p2[parent2:-1]
        child = dict()
        child = p1a + p2b
        return child


    def mutation(self):
        m = random.randint(0,2)
        if m == 0: #punt mutation
          return  
        elif m == 1: #switch two bits
            return
        else: #move one bit to different spot
            return
    return
    
    
def test_algorithm(dimension, instance=1):
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


def collect_data(dimension=100, nreps=5):
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
    test_algorithm(10)

    # Test required for A1, your GA should be able to pass this!
    # test_algorithm(100)

    # If your implementation passes test_algorithm(100)
    collect_data(100)
