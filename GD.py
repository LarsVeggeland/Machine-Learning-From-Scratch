#---------- Imported libraries ----------

from random import random



#---------- Objective function and partial derivatives ----------

def func(x, y) -> float: return 25 * (y + x**2)**2 + (1 + x)**2


def dfdx(x, y) -> float: return (100*x)*(y+x**2) + 2 * (1 + x)


def dfdy(x, y) -> float: return 50 * (y + x**2)



#---------- Stoppong criteria functions ----------

def  stop_by_low_improvement(x, y, new_x, new_y, limit) -> bool:
    # returns true if the rate of improvment is too small
    return func(x, y) - func(new_x, new_y) <= limit


def stop_by_convergence(x, y, limit):
    # returns true if the slope of both variables are too small
    return abs(dfdx(x, y)) <= limit and abs(dfdy(x, y)) <= limit



#---------- Gradient Descent function ----------

def GD(max_iter : int, bounds : list, alpha : float, stopping_criteria, limit : float) -> list:

    # Generating 100 random (x, y) pairs distributed within the bounds
    # This is implemented in an effort to increase the likelyhood of finding the global optima
    x_vals = [bounds[0] + random()*(bounds[1]-bounds[0]) for i in range(100)]
    y_vals = [bounds[0] + random()*(bounds[1]-bounds[0]) for i in range(100)]
    res = []

    #print("init x,y =", (x, y))

    for i in range(100):

        x = x_vals[i]
        y = y_vals[i]

        for j in range(max_iter):

            new_x = x - alpha * dfdx(x, y)
            new_y = y - alpha * dfdy(x, y)

            # We are not interested in values residing outside the specified bounds
            if new_x < bounds[0] or new_x > bounds[1] or new_y < bounds[0] or new_y > bounds[1]:
                break

            if (stopping_criteria is  stop_by_low_improvement):
                if (stopping_criteria(x, y, new_x, new_y, limit)):
                    break
            
            else:
                if (stopping_criteria(x, y, limit)):
                    break

            x = new_x
            y = new_y
        
        res.append([[x, y], func(x, y)])

    return min(res, key=lambda x : x[1])


def print_result():

    si = "stop_by_improvment"
    sc = "stop_by_convergence"

    for iter in [100, 1000, 100000]:
        with open(f"GD_{iter}_iterations.csv", "a") as csv:
            csv.write("Learning_rate, Stop_criteria, limit, x, y, fitness\n")
            for alpha in [0.000001, 0.001, 1]:
                for stop_criteria in [ stop_by_low_improvement, stop_by_convergence]:
                    for limit in [0.01, 0.1]:
                        res = GD(iter, [-10, 10], alpha, stop_criteria, limit)
                        print(f"iterations: {iter} alpha: {alpha} stop_criteria: {si if stop_criteria is  stop_by_low_improvement else sc} limit: {limit} x: {res[0][0]} y: {res[0][1]} fitness: {res[1]}")
                        csv.write(f"{alpha}, {si if stop_criteria is  stop_by_low_improvement else sc}, {limit}, {res[0][0]}, {res[0][1]}, {res[1]}\n")
        print()

def test_fitness_of_learningrate():
    with open("GD_LR_More.txt", 'a') as file:
        for iterations in range(1000, 100001, 1000):
            for i in [0.000001, 0.001, 1]:
                avg = sum([GD(iterations, [-10, 10], i, stop_by_convergence, 0.01)  [1] for j in range(10)])/10
                file.write(str([avg, iterations]) + "\n")

#print_result()

def test_best_stopping_criteria():
    with open("GD_SC.txt", 'a') as file:
        for i in range(20, 1001, 20):
            avg = sum([GD(i, [-10, 10], 0.001, stop_by_convergence, 0.01)  [1] for j in range(10)])/10
            file.write(str([avg, i]) + "\n")
            avg = sum([GD(i, [-10, 10], 0.001,  stop_by_low_improvement, 0.01)  [1] for j in range(10)])/10
            file.write(str([avg, i]) + "\n")

test_fitness_of_learningrate()
#test_best_stopping_criteria()
