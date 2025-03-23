import numpy as np

class HillClimbing:
    best_solution = None
    best_solution_value = None
    domain = None
    step_size = None
    
    def __init__(self, domain = (-5.12, 5.12), step_size = 0.1):
        self.domain = domain
        self.step_size = step_size
    
    def generate_neighborhood_solution(self, dim):
        steps = np.random.normal(0, self.step_size, dim)
        return np.clip(self.best_solution+steps, [self.domain[0]]*dim, [self.domain[1]]*dim)
    
    def update_best_solution(self, solution, quality_value):
        self.best_solution, self.best_solution_value = solution, quality_value
        
    def generate_initial_solution(self, dim):
        min_generate_value, max_generate_value = self.domain
        
        mean = (min_generate_value+max_generate_value)/2
        std = (abs((min_generate_value-max_generate_value))/2)
        
        return np.random.uniform(mean, std,dim)
        
    
    def fit(self,objective_function, dim, epochs):
        
        initial_solution = self.generate_initial_solution(dim)
       
        self.update_best_solution(initial_solution, objective_function(initial_solution))

        epoch_results = []
        
        for _ in range(epochs):
            new_solution = self.generate_neighborhood_solution(dim)
            new_solution_quality = objective_function(new_solution)
            
            if new_solution_quality < self.best_solution_value:
                self.update_best_solution(new_solution, new_solution_quality)
                
            epoch_results.append(self.best_solution_value)
        
        return epoch_results