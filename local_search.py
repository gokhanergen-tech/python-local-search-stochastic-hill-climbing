import numpy as np

class LocalSearch:
    
    best_solution = None
    best_solution_value = None
    
    def calculate_points(self, neighborhood_count,dim,radius_scale_size, domain):
            min_generate_value, max_generate_value = domain
            radius = radius_scale_size*(max_generate_value-min_generate_value)
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
        
    def calculate_initial_solution(self, domain, dim):
        min_generate_value, max_generate_value = domain
        return np.random.uniform(min_generate_value, max_generate_value, dim)
    
    def fit(self,objective_function, epochs,dim, domain = (-5.12, 5.12), neighborhood_count = 1000, radius_scale_size=0.1, algorithm_type = ""):
        initial_solution = self.calculate_initial_solution(domain, dim)
        self.best_solution = initial_solution
        self.best_solution_value = objective_function(self.best_solution)
        
        epochs_results = []
      
        for _ in range(epochs):
             neighbour_hoods = self.calculate_points(neighborhood_count,dim,radius_scale_size,domain)
             
             
             match algorithm_type:
                 case "stochastic_hill_climbing":
                  selected_neighbourhood_index = np.random.choice(neighborhood_count)
                  selected_neighbourhood_value = objective_function(neighbour_hoods[selected_neighbourhood_index])
                 case _:
                  solutions = np.array([objective_function(np.array(neighborhood)) for neighborhood in neighbour_hoods])
                  selected_neighbourhood_index = np.argmin(solutions)
                  
                  selected_neighbourhood_value = solutions[selected_neighbourhood_index]

             if selected_neighbourhood_value < self.best_solution_value:
                 self.best_solution_value  = selected_neighbourhood_value
                 self.best_solution = neighbour_hoods[selected_neighbourhood_index]
             else:
                 if algorithm_type != "stochastic_hill_climbing":
                     epochs_results.append(self.best_solution_value)
                     break
             epochs_results.append(self.best_solution_value)
                 
               
        return epochs_results