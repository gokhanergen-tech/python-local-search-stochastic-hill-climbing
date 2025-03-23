import numpy as np

class LocalSearch:
    
    best_solution = None
    best_solution_value = None
    
    def fit(self,objective_function, domain, dim, epochs, neighborhood_count, size):
        min_generate_value, max_generate_value = domain
        
        initial_population = np.random.uniform(min_generate_value, max_generate_value, dim)
        
        self.best_solution = initial_population
        self.best_solution_value = objective_function(self.best_solution)
        
        radius = size*(max_generate_value-min_generate_value)
        
        def calculate_points():
            axis_values = []
            
            for i in range(dim):
                
                min_value = self.best_solution[i] - radius
                min_value = min_value < min_generate_value and min_generate_value or min_value
                
                max_value = self.best_solution[i] + radius
                max_value = max_value > max_generate_value and max_generate_value or max_value
                
                mean = (max_value+min_value)/2
                std = abs((max_value-min_value)/2)
                axis_values.append(np.random.normal(mean, std, neighborhood_count))
                
            return np.vstack(axis_values).transpose()
        
        epoch_results = []
        
        for _ in range(epochs):
             neighbour_hoods = calculate_points()
             solutions = np.array([objective_function(np.array(neighborhood)) for neighborhood in neighbour_hoods])
             best_neighbourhood_index = np.argmin(solutions)
             best_neighbourhood_value = solutions[best_neighbourhood_index]

             if best_neighbourhood_value < self.best_solution_value:
                 self.best_solution_value  = best_neighbourhood_value
                 self.best_solution = np.array(neighbour_hoods[best_neighbourhood_index])
             epoch_results.append(self.best_solution_value)
                 
                 
        return epoch_results