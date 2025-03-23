import numpy as np
import matplotlib.pyplot as plt

def de_jong_1st_objective_function(x):
    return np.sum(x**2)

def de_jong_2nd_rosenbrock_s_saddle(x):
    return np.sum(np.array([(100*((x[i+1]**2-x[i])**2)+(1-x[i])**2) for i in range(0,len(x)-1)]))

def de_jong_4th(x):
    return np.sum(np.multiply((x**4),np.arange(x.shape[0])+1))

def rastrigin(x):
    D = x.shape[0]
    return 10*D+np.sum(x**2-10*np.cos(2*np.pi*x))

def create_top_text(maxFES, best_result,best_points, function_name, problem_dimension):
    return f"{function_name} in {problem_dimension}D (maxFES = {maxFES})\nBest solution: {best_result}\n{best_points}"

def griewangks(x):
    return 1+np.sum((x**2)/4000)-np.prod(np.cos(x/np.sqrt(np.arange(x.shape[0])+1)))

def plot_line(x, y, label_x='FES', label_y='f(x)', text=""):
    plt.plot(x, y, label='Solution Quality')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.grid(True)

    plt.annotate(text, xy=(0,0.9), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    
    plt.show()
    
def create_top_text(maxFES, best_result,best_points, function_name, problem_dimension):
    return f"{function_name} in {problem_dimension}D (maxFES = {maxFES})\nBest solution: {best_result}\n{best_points}"
