import random

def sphere_function(x):
    s = 0
    for i in range(len(x)):
        s += x[i] ** 2
    return s

def local_search(fun, domain, dim, maxIt, NP, size):

    x0 = [random.uniform(domain[0], domain[1]) for _ in range(dim)]
    f0 = fun(x0)

    it = 0
    while(it < maxIt):

        rad = size * (domain[1] - domain[0])
        pop = [[random.uniform(x0[i] - rad, x0[i] + rad) for i in range(dim)] for _ in range(NP)]
        #clip the solutions in pop to domain
        fPop = [fun(p) for p in pop]
        #print(fPop)
        # select the best solution from fPop including the original fx0 (depends on algorithm)
        x0 = pop[0]
        fx0 = fPop[0]

        it += 1

    return x0, f0



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    res = local_search(sphere_function, [-10, 10], 2, 10000, 10, 0.1)
    print(res)




