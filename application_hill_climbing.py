
from consts import FUNCTION_DE_JONG_2ND, FUNCTION_DE_JONG_4TH, FUNCTION_GRIEWANGKS, FUNCTION_RASTRIGIN
from local_search import LocalSearch
from utils import create_top_text, de_jong_1st_objective_function, de_jong_2nd_rosenbrock_s_saddle, de_jong_4th, griewangks, plot_line, rastrigin

local_search = LocalSearch()

def find_best_with_hill_climbing(problem_dimension = 5, optimize_function_name = "De Jong 1st", optimization_function = de_jong_1st_objective_function):
    epochs = problem_dimension*10000

    epoch_results = local_search.fit(optimization_function, epochs, problem_dimension,algorithm_type="stochastic_hill_climbing")

    text = create_top_text(epochs, 
                           local_search.best_solution_value, 
                           local_search.best_solution,
                           optimize_function_name, 
                           problem_dimension=problem_dimension)

    plot_line(range(0, epochs),epoch_results, text=text)
    
find_best_with_hill_climbing()
find_best_with_hill_climbing(problem_dimension=10)
find_best_with_hill_climbing(problem_dimension=20)

find_best_with_hill_climbing(optimization_function=de_jong_2nd_rosenbrock_s_saddle, optimize_function_name=FUNCTION_DE_JONG_2ND)
find_best_with_hill_climbing(problem_dimension=10, optimization_function=de_jong_2nd_rosenbrock_s_saddle, optimize_function_name=FUNCTION_DE_JONG_2ND)
find_best_with_hill_climbing(problem_dimension=20, optimization_function=de_jong_2nd_rosenbrock_s_saddle, optimize_function_name=FUNCTION_DE_JONG_2ND)

find_best_with_hill_climbing(optimization_function=de_jong_4th, optimize_function_name=FUNCTION_DE_JONG_4TH)
find_best_with_hill_climbing(problem_dimension=10, optimization_function=de_jong_4th, optimize_function_name=FUNCTION_DE_JONG_4TH)
find_best_with_hill_climbing(problem_dimension=20, optimization_function=de_jong_4th, optimize_function_name=FUNCTION_DE_JONG_4TH)

find_best_with_hill_climbing(optimization_function=rastrigin, optimize_function_name=FUNCTION_RASTRIGIN)
find_best_with_hill_climbing(problem_dimension=10, optimization_function=rastrigin, optimize_function_name=FUNCTION_RASTRIGIN)
find_best_with_hill_climbing(problem_dimension=20, optimization_function=rastrigin, optimize_function_name=FUNCTION_RASTRIGIN)

find_best_with_hill_climbing(optimization_function=griewangks, optimize_function_name=FUNCTION_GRIEWANGKS)
find_best_with_hill_climbing(problem_dimension=10, optimization_function=griewangks, optimize_function_name=FUNCTION_GRIEWANGKS)
find_best_with_hill_climbing(problem_dimension=20, optimization_function=griewangks, optimize_function_name=FUNCTION_GRIEWANGKS)