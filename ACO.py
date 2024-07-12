from numpy import zeros, sqrt, sum, where
import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
import time
import sys

# This program contains a more optimized version for the ACO algorithm
# The syntax of this program is: ACO.py <number of ants> <number of iterations> <number of processes>
# The program writes in two files:
# - The best path found is added to the ACO_path.csv file
# - The curve of the highest accuracy achieved through the iterations is added to the ACO_accuracy_curve.csv file

# ======================== hyperparameters ========================

ALPHA = 0.95 # parameters used in the cost function
BETA = 1 - ALPHA 
BOTTOM_FEATURES = 50 # bottom number of features to search with
TOP_FEATURES = 120 # top numbers of features to search with
MIN_PHER = 0.01 # minimum quantity of pheromone
Q_ACO = 0.2
ALPHA_ACO = 1
BETA_ACO = 1
EVAPORATION_RATE = 0.5
NEIGH = 30

#credit:
#https://towardsdatascience.com/create-your-own-k-nearest-neighbors-algorithm-in-python-eb7093fc6339

def most_common(lst):
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return sqrt(sum((point - data)**2, axis=1))

class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train    
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])        
        return list(map(most_common, neighbors))    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


def count_features(agent):
    """
    Returns the number of 1s of a given agent
    
    Parameters:
    agent -- The agent for which the number of features is
    counted

    Return:
    the number of features of the agent
    """

    ind = np.where(agent == 1)[0]

    return len(ind)


def cost_func(agent, alpha, beta, num_samples_train, num_samples_test, train_x, train_y, test_x, test_y, num_char, max_features, k):     
    """
    Cost functions of the algorithms.

    The cost is calculated as the error of the agent
    divided by the average error for a agent with
    the same number of features

    Parameters:
    agent -- The agent evaluated

    Returns:
    The cost of the agent
    """ 

    inputY = train_y
    testY = test_y
    
    ind = where(agent == 1)[0]
    if (len(ind) == 0):
        return [1,[0.0]]
    inputX = zeros((num_samples_train,len(ind)), dtype=float)
    testX = zeros((num_samples_test,len(ind)), dtype=float)
    aux = [i for i in range(len(ind))]
    inputX[:,aux] = train_x[:,ind]
    testX[:,aux] = test_x[:,ind]

    neigh = KNeighborsClassifier(k)
    neigh.fit(inputX,inputY)

    num_success = 0
    prediction = neigh.predict(testX)

    for i in range(num_samples_test):

        if (prediction[i] == testY[i]):
            num_success += 1

    acc = num_success/num_samples_test
    # an adjustment is performed to the accuracy value
    # to take into account the samples guessed right 'by luck'
    value = acc - 1/(num_char -1)*(1 - acc)
    error = 1 - value

    return alpha * error + beta * len(ind)/max_features


def ant_path_maker(num_samples_train, 
                   num_samples_test, 
                   train_x, 
                   train_y, 
                   test_x, 
                   test_y, 
                   num_char, 
                   max_features, 
                   pheromone, 
                   lut, 
                   bottom_features, 
                   top_features, 
                   alpha, 
                   beta, 
                   rng,
                   k):
    
    """
    Function run in parallel. It contains the part of the ACO
    algorithm where the path is created.

    It evaluates the path at each new addition of a feature. It is
    done only in between desired_n_features/2 and desired_n_features
    number of features the path in construction contains.

    Parameters:

    desired_n_features -- Number that represents the maximum number of
    features that the constructed path must have.

    Returns:

    best_path_ant -- Best path found
    best_path_score_ant -- Score of the best path found

    """

    visited = np.array([False]*max_features)  # list that indicates which features are selected already               
    unvisited = np.where(np.logical_not(visited))[0]    # list of the positions of features not selected
    path = []
    best_path_ant = []     # best path of the current ant
    best_path_score_ant = 2             # score of the best path of the current ant

    selected_features = 0
    run = True
    while selected_features <= top_features and run:    # in each iteration, a new feature is added to the path
        unvisited = np.where(np.logical_not(visited))[0]
        probabilities = np.zeros(len(unvisited))

        for i, unvisited_feature in enumerate(unvisited):
            probabilities[i] = pheromone[unvisited_feature]**alpha * (heuristic(unvisited_feature, lut))**beta
        sum = np.sum(probabilities)

        if(sum > 0):
            probabilities /= np.sum(probabilities)

            next_feature = rng.choice(unvisited, p=probabilities)

            path.append(next_feature)
            selected_features += 1
            visited[next_feature] = True

            #if(selected_features > desired_n_features//4):
            if (selected_features >= bottom_features):
                path_score = cost_func(visited,  ALPHA, BETA, num_samples_train, num_samples_test, train_x, train_y, test_x, test_y, num_char, max_features,k)

                if(path_score < best_path_score_ant):
                    best_path_score_ant = path_score
                    best_path_ant = path.copy()
        else:

            run = False
    return best_path_ant, best_path_score_ant

def heuristic(feature, lut_):
    """Heuristic of the algorithm"""
    return lut_[feature]

if __name__ == "__main__":

    #here are imported all the libraries necessary for the main process
    import random as rand           
    import numpy as np
    import csv
    from matplotlib import pyplot as plt
    import time
    from multiprocessing.pool import Pool
    from multiprocessing import Process, Array
    from sklearn.preprocessing import StandardScaler

    def print_syntax():
        """Shows the correct syntax to execute the programm"""

        print("Correct syntax: ACO.py <number"
            " of agents> <number of iterations> <number of processes>")

    def manage_error(msg):
        """
        Function that is executed in the case of an error
        
        Parameters:
        msg -- message containing information of the error
        """
        print(msg)
        print_syntax()
        sys.exit(2)

    try:
        num_proc = int(sys.argv[3]) # number of processes the algorithm will execute
    except ValueError:
        manage_error("Error. Number of processes must be an integer")

    except IndexError:
        manage_error("Error. Number of processes not provided")            

    try:
        num_it = int(sys.argv[2]) # number of iterations the algorithm will perform
    except ValueError:
        manage_error("Error. Number of iterations must be an integer")

    except IndexError:
        manage_error("Error. Number of iterations not provided")

    try:
        num_ag = int(sys.argv[1]) # number of agents the algorithm will work with
    except ValueError:
        manage_error("Error. Number of agents must be an integer")

    except IndexError:
        manage_error("Error. Number of agents not provided")

    if (num_it <= 0 or num_ag <= 0):
        manage_error("Error. All integers parameters must be greater than 0")

    # learning data
    NUM_CHAR = 3 # number of different classes
    MAX_FEATURES = 3600 # total number of features

    NUM_SAMPLES_TRAIN = 178 # number of samples used for training
    NUM_SAMPLES_TEST = 178 # number of samples used for testing

    TRAIN_X = np.empty((NUM_SAMPLES_TRAIN,MAX_FEATURES))
    TRAIN_Y = np.empty(NUM_SAMPLES_TRAIN)
    TEST_X = np.empty((NUM_SAMPLES_TEST,MAX_FEATURES))
    TEST_Y = np.empty(NUM_SAMPLES_TEST)


    try:
        with open('Essex/104_training_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)

            i = 0
            for row in reader:
                for j in range(MAX_FEATURES):
                    TRAIN_X[i,j] = row[j]
                i+=1
    except ValueError:
        manage_error("Error reading training data")

    try:
        with open('Essex/104_training_class.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)

            i = 0
            for row in reader:
                TRAIN_Y[i] = row[0]
                i+=1
    except ValueError:
        manage_error("Error reading training classes")

    try:
        with open('Essex/104_testing_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)

            i = 0
            for row in reader:
                for j in range(MAX_FEATURES):
                    TEST_X[i,j] = row[j]
                i+=1
    except ValueError:
        manage_error("Error reading testing data")

    try:
        with open('Essex/104_testing_class.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)

            i = 0
            for row in reader:
                TEST_Y[i] = row[0]
                i+=1
    except ValueError:
        manage_error("Error reading testing classes")

    scaler = StandardScaler()
    TRAIN_X = scaler.fit_transform(TRAIN_X)
    TEST_X = scaler.fit_transform(TEST_X)

    print("Number of agents: ", num_ag)
    print("Number of iterations: ", num_it)
    print("Number of processes: ", num_proc)
    print("")



    def cost_to_acc(cost, agent):
        """
        Calculates the accuracy of a agent given its cost

        Patameters:
        cost -- The cost of the agent. Calculated previously
        by cost_func

        agent -- The agent

        Returns:
        The accuracy of the agent
        """
        selected_features = count_features(agent)

        # the inverse functions used to calculate the cost are executed
        # in order to obtain the accuracy of the agent
        error = (cost - BETA*selected_features/MAX_FEATURES)/ALPHA
        value = 1 - error
        return ((NUM_CHAR -1)*value +1)/NUM_CHAR 

    def write_time(time, filename):
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([time])

    def write_accuracy(accuracy, filename):
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([accuracy])

    def write_path(path, filename):
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(path)

    def write_accuracy_curve(curve, filename):
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(curve)


    def path_to_solution(path):
        """
        Function that converts a path in an agent

        Paths are list that contains the positions of the selected features,
        whereas solutions are lists of 1s and 0s, that indicate which
        features are selected
        """
        solution = np.zeros(MAX_FEATURES,int)
        for p in path:
            solution[p] = 1
        return solution


    # algorithm based on https://induraj2020.medium.com/implementation-of-ant-colony-optimization-using-python-solve-traveling-salesman-problem-9c14d3114475

    def ant_colony_optimization(bottom_features, top_features, alpha, beta, Q, evaporation_rate, pheromone_min, k):
        pheromone = np.ones(MAX_FEATURES)
        best_path = None        # global best path found
        best_path_score = 2     # score of the global best path
        pheromone_max = Q/(1-evaporation_rate)    # maximum quantity of pheromone that a feature can have
        best_score_so_far = np.zeros(num_it)  # list that contains the best score found with each iteration
        
        for iteration in range(num_it):
            best_path_iteration = None      # best path at the current iteration
            best_path_score_iteration = 2   # score of the best path at the current iteration

            args = []

            for j in range(num_ag):
                args.append([NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, 
                            TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, NUM_CHAR, MAX_FEATURES, pheromone, LUT, bottom_features, top_features, alpha, beta, np.random.default_rng(), k])    

            with Pool(processes=num_proc) as pool:
                ant_path = pool.starmap(ant_path_maker,args,chunksize=num_ag//num_proc + int(num_ag%num_proc != 0))

            print("Iteration ", iteration)
            for i in range(num_ag):
                
                print("Ant ", i, "| Best ant score ", 
                    ant_path[i][1], "| Features ", len(ant_path[i][0]))


                if(ant_path[i][1] < best_path_score_iteration):     # if the constructed path is better than any of the paths
                    best_path_score_iteration = ant_path[i][1]      # of the current iteration, a new best path of the iteration
                    best_path_iteration = ant_path[i][0].copy()           # is selected

            print("Best fitness of all ants:",best_path_score_iteration)

            if (best_path_score_iteration < best_path_score):          # if the constructed path is better than the best path
                best_path = best_path_iteration.copy()                 # found until now, a new global best path is selected
                best_path_score = best_path_score_iteration

            print("Best accuracy so far:", cost_to_acc(best_path_score,path_to_solution(best_path)))

            best_score_so_far[iteration] = best_path_score
            #print("ACO Best score of the iteration: ", best_path_score_iteration, "Best score so far: ", best_path_score) 
            time_point = time.time()
            print("Elapsed time:", "{:.2f}".format(time_point-t1))
            print("")

            pheromone *= evaporation_rate

            # A MAX-MIN system with a hybrid approach using both the
            # best global score and the best score of the iteration
            # is implemented here 

            for i in range(len(best_path)):
                pheromone[best_path[i]] += Q/(best_path_score)/2
                    
            for i in range(len(best_path_iteration)):
                pheromone[best_path_iteration[i]] += Q/(best_path_score_iteration)/2

            for i in range(len(pheromone)):
                if (pheromone[i] < pheromone_min):
                    pheromone[i] = pheromone_min
                # elif (pheromone[i] > pheromone_max):
                #     pheromone[i] = pheromone_max

        return best_path, best_score_so_far
    

    # Read of a look up table that will serve as
    # the heuristic of the algorithm

    LUT = np.zeros(MAX_FEATURES)
    with open('LUT.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            for n in row:
                LUT[i] = n
                i+=1

    t1 = time.time()
    try:
        path, best_solutions_fitness = ant_colony_optimization(bottom_features=BOTTOM_FEATURES,
                                                               top_features=TOP_FEATURES,
                                                               alpha=ALPHA_ACO, 
                                                               beta=BETA_ACO, 
                                                               Q=Q_ACO,
                                                               evaporation_rate=EVAPORATION_RATE, 
                                                               pheromone_min=MIN_PHER,
                                                               k=NEIGH)
    finally:
        t2 = time.time()
        time_spent = "{:.2f}".format(t2-t1)
        print("Total elapsed time: ", time_spent)
        print("")

        # =========================== example ===========================

        # A path consists of a set of indexes of features
        example_path = [2378, 3250, 3209, 916, 387, 2358, 2364, 1120, 2034, 2350, 1638, 3448, 3478, 926, 2008, 580, 1424, 1098, 388, 2356, 2530, 3084, 2182, 3280, 1848, 3438, 1824, 1090, 1448, 2928, 910, 2391, 746, 3430, 2904, 1826, 1399, 1108, 2908, 3460, 2980, 3070, 940, 2738, 1270, 924, 3569, 1646, 2009, 2004, 1458, 2890, 1278, 919, 1308, 1251, 3551, 744, 2710, 730, 2724, 417, 380, 1284, 1444, 1025, 3101, 1644, 2917, 1110, 3066, 174, 1818, 922, 1668, 742, 3078, 3003, 1096, 1290, 2180, 1088, 3100, 3231, 1236, 550, 3092, 3276, 3086, 2718, 1810, 1428, 1450, 2170, 2560, 3516, 2016, 3258, 2607, 1318, 1158]
        
        # A solution is an array of 0s and 1s which indicates which features are selected
        example_solution = path_to_solution(example_path)

        # The fitness of a solution is the value returned by cost_func
        example_solution_fitness = cost_func(example_solution, ALPHA, BETA, 178, 178, TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, 3, 3600, NEIGH) 
        
        # The accuracy of the solution is the rate of classification success
        example_solution_accuracy = cost_to_acc(example_solution_fitness, example_solution)

        print(f"Number of features of example solution : {count_features(example_solution)}")
        print(f"Fitness value of example solution = {example_solution_fitness}")
        print(f"Accuracy of example solution = {example_solution_accuracy}")
        print("")
        
        # ======================== this execution ======================

        #best path found
        print(f"Parameters of the best solution : {path}")
        print("")

        #solution found
        solution = path_to_solution(path)

        #fitness of the solution
        solution_fitness = cost_func(solution, ALPHA, BETA, 178, 178, TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, 3, 3600, NEIGH) 
        
        #accuracy of the solution
        accuracy = cost_to_acc(solution_fitness, solution)

        #accuracy curve with iterations
        accuracy_array = np.zeros(len(best_solutions_fitness))
        for i in range(len(accuracy_array)):
            accuracy_array[i] = cost_to_acc(best_solutions_fitness[i],2)

        print(f"Number of features of the best solution : {count_features(solution)}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Accuracy of the best solution = {accuracy}")
        print("")
        write_path(path, "ACO_path.csv")
        write_accuracy_curve(accuracy_array, "ACO_accuracy_curve.csv")

        # ================ plot of the accuracy curve of this execution  ================

        iterations_x = np.arange(1, num_it+1)
        plt.plot(iterations_x, accuracy_array)
        plt.title("ACO" + " - Iteration vs. Accuracy " + time_spent + " s")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")

        plt.grid(axis='both', color='0.80')
        plt.ylim(0.5,1)
        plt.show()
