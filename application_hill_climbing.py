
from consts import FUNCTION_DE_JONG_2ND, FUNCTION_DE_JONG_4TH, FUNCTION_GRIEWANGKS, FUNCTION_RASTRIGIN
from hill_climbing import HillClimbing
from utils import create_top_text, de_jong_1st_objective_function, de_jong_2nd_rosenbrock_s_saddle, de_jong_4th, griewangks, plot_line, rastrigin

hill_climbing = HillClimbing()

def find_best_with_hill_climbing(problem_dimension = 5, optimize_function_name = "De Jong 1st", optimization_function = de_jong_1st_objective_function):
    epochs = problem_dimension*1000

    epoch_results = hill_climbing.fit(optimization_function, problem_dimension, epochs)

    text = create_top_text(epochs, 
                           hill_climbing.best_solution_value, 
                           hill_climbing.best_solution,
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