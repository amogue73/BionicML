import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from matplotlib import pyplot as plt
import csv
import math
import pygad
from scipy.special import erfinv
import pandas as pd
import dataframe_image as dfi

TOTAL_SAMPLES = 2500
SAMPLES_AVERAGES = 20


def div_down(a,b):
    div = a//b
    return div

def div_up(a,b):
    div = a//b
    if(a%b != 0):
        div+=1
    return div


def regressors_maker_energy(n_agents,n_iterations, n_proc, 
                            tm, 
                            tmn, 
                            to, 
                            to2,
                            tp, 
                            ti
                            ):
    R = div_up(n_agents,n_proc)

    T = tmn*n_iterations*n_agents + ti + tm*n_iterations+ to*n_iterations*div_up(n_agents,R) + to2*n_iterations*int(n_proc>1)*div_down(n_agents,R) + tp*n_iterations*R
    C4 = tp*n_iterations*n_agents
    C8 = to*n_iterations*div_up(n_agents,R)*(div_up(n_agents,R)) 
    adjust = 1/3600000

    return (T*adjust, C4*adjust, C8*adjust)


def regressors_maker_time(n_agents,n_generations, n_proc):

    R = div_up(n_agents,n_proc)

    return [n_generations,
            n_generations*n_agents, 
            n_generations*div_up(n_agents,R),
            n_generations*int(n_proc>1)*(div_down(n_agents,R)),
            n_generations*R, 
            1
            ]


def delete_values(averages, y_raw_values, dist_values_skipped, values_skipped):
    for chunk in range(TOTAL_SAMPLES//SAMPLES_AVERAGES):
        bottom = chunk*SAMPLES_AVERAGES
        top = bottom+SAMPLES_AVERAGES

        average = 0
        n_skip = 0
        for i in range(bottom,top):
            if i in values_skipped:
                n_skip += 1
            else:
                average += y_raw_values[i]
        
        average /= (SAMPLES_AVERAGES - n_skip)
        averages[chunk] = average
        dist_values_skipped[n_skip]+=1



def make_y_values(thres, y_raw_values_unord):
    y_deviation = np.zeros(TOTAL_SAMPLES)
    averages = np.zeros(TOTAL_SAMPLES//SAMPLES_AVERAGES)
    y_raw_values = np.zeros(TOTAL_SAMPLES)

    average = 0
    count = 0
    jump = TOTAL_SAMPLES//SAMPLES_AVERAGES

    for i in range(TOTAL_SAMPLES):
        y_raw_values[i] = y_raw_values_unord[i//SAMPLES_AVERAGES + jump*i%TOTAL_SAMPLES]
    
    for i in range(len(y_raw_values)):
        average += y_raw_values[i]
        count +=1
        if count == SAMPLES_AVERAGES:
            average /= SAMPLES_AVERAGES
            averages[i//SAMPLES_AVERAGES] = average
            count = average = 0

    for i in range(len(y_raw_values)):
        y_deviation[i] = y_raw_values[i] - averages[i//SAMPLES_AVERAGES]

    values_skipped = np.where(y_deviation > thres)[0]
    dist_values_skipped = np.zeros(SAMPLES_AVERAGES+1)
    delete_values(averages, y_raw_values, dist_values_skipped, values_skipped)

    for i in range(len(y_raw_values)):
        y_deviation[i] = y_raw_values[i] - averages[i//SAMPLES_AVERAGES]

    values_skipped = np.where(y_deviation <-thres)[0]
    delete_values(averages, y_raw_values, dist_values_skipped, values_skipped)

    return averages, dist_values_skipped



def make_regression_time(averages):
    regressors_list = []
    
    it = 0
    for i in range(10,31,5):
        for j in range(10,31,5):
            for k in 1, 2 ,4, 6, 8:
                regressors_list.append(regressors_maker_time(i,j,k))
                it += 1

    print("")
    regressors = np.array(regressors_list)

    reg = LinearRegression(fit_intercept=False,positive=True).fit(regressors, averages)

    pred_values = np.zeros(TOTAL_SAMPLES//SAMPLES_AVERAGES)
    for i in range(len(pred_values)):
        pred_values[i] = np.dot(reg.coef_,regressors[i])

    return reg, pred_values, regressors

def make_regression_energy(averages, times):
    regressors_list = []

    it = 0
    for i in range(10,31,5):
        for j in range(10,31,5):
            for k in 1, 2 ,4, 6, 8:
                regressors_list.append(regressors_maker_energy(i,j,k,*times))
                it += 1
    regressors = np.array(regressors_list)

    reg = LinearRegression(fit_intercept=False,positive=True).fit(regressors, averages)

    pred_values = np.zeros(TOTAL_SAMPLES//SAMPLES_AVERAGES)
    for i in range(len(pred_values)):
        pred_values[i] = np.dot(reg.coef_,regressors[i])

    return reg, pred_values, regressors


def print_results(dist_values_skipped, averages, pred_values, regressors, reg, mode):
    print(dist_values_skipped)
    rmse = np.sqrt(mse(averages, pred_values))

    avg_averages = 0
    for average in averages:
        avg_averages += average
    avg_averages/=len(averages)


    print(alg + " " + mode)
    print("RMSE:", rmse)
    print("RRMSE:", rmse/avg_averages)
    print("average " + mode + ":", avg_averages)
    print("R2:", reg.score(regressors, averages))
    print("Coeficients:", reg.coef_)
    print("----------------------------------------")

def print_deviation(pred_values, averages):
    pred_deviation = []
    for i in range(len(pred_values)):
        pred_deviation.append(pred_values[i] - averages[i])

algorithms = ["GA", "PSO", "ACO", "WOA", "GWO"]

for alg in algorithms:

    if (alg == "ACO"):
        TOTAL_SAMPLES = 2500
        SAMPLES_AVERAGES = 20
        THRESHOLD_TIMES = 40
        THRESHOLD_ENERGY = 0.001
    else:
        TOTAL_SAMPLES = 2500
        SAMPLES_AVERAGES = 20 
        THRESHOLD_TIMES = 1
        THRESHOLD_ENERGY = 0.00002


    y_raw_values_times_unord = np.zeros(TOTAL_SAMPLES)
    y_raw_values_energy_unord = np.zeros(TOTAL_SAMPLES)
    y_raw_values_accuracy_unord = np.zeros(TOTAL_SAMPLES)

    raw_values_power_unord = np.zeros(TOTAL_SAMPLES)
    raw_values_power = np.zeros(TOTAL_SAMPLES)

    with open("Experimental_measurements/" + alg + "/times.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            y_raw_values_times_unord[i] = float(row[0])
            i+=1

    with open("Experimental_measurements/" + alg + "/emissions.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = -1
        index_cpu_energy = 0
        index_cpu_power = 0
        for row in reader:
            if (i==-1):
                index_cpu_energy = row.index('cpu_energy')
                index_cpu_power = row.index('cpu_power')
            else:
                y_raw_values_energy_unord[i] = float(row[index_cpu_energy])
                raw_values_power_unord[i] = float(row[index_cpu_power])
            i+=1
            


    averages_times, dist_values_skipped_times = make_y_values(THRESHOLD_TIMES,y_raw_values_times_unord)
    averages_energy, dist_values_skipped_energy = make_y_values(THRESHOLD_ENERGY,y_raw_values_energy_unord)


    reg_time, pred_values_time, regressors_time = make_regression_time(averages_times)

    pred_values_time = np.zeros(TOTAL_SAMPLES//SAMPLES_AVERAGES)
    for i in range(len(pred_values_time)):
        pred_values_time[i] = np.dot(reg_time.coef_,regressors_time[i])

    reg_energy, pred_values_energy, regressors_energy  = make_regression_energy(averages_energy, reg_time.coef_)

    print_results(dist_values_skipped_times, averages_times, pred_values_time, regressors_time, reg_time, 'time')
    print_results(dist_values_skipped_energy, averages_energy, pred_values_energy, regressors_energy, reg_energy, 'energy')

    

    