'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cloudpickle as pickle
from struct import unpack
import brian2 as b
from brian2 import *

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     
def get_labeled_data(picklename, bTrain = True):
    """ Read input-vector (image) and target class (label, 0-9) 
    and return as list of tuples
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # open gzip images in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images-idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 'train-labels-idx1-ubyte', 'rb')
        else:
            images = open(MNIST_data_path + 't10k-images-idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte', 'rb')
        # get metadata for images
        images.read(4) #skip magic number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # get metadata for labels
        labels.read(4) #skip magic number
        N = unpack('>I', labels.read(4))[0]
        
        if number_of_images != N:
            raise Exception('Number of labels did not match number of images!')
        # get data
        x = np.zeros((N, rows, cols), dtype=np.uint8) # initialize numpy array
        y = np.zeros((N,1), dtype=np.uint8) # initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)] for unused_row in range(rows)]
            y[i] = unpack('>B', labels.read(1))[0]
        
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    print(result_monitor.shape)
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in range(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
    return assignments

MNIST_data_path = '/Users/TyTheWiseGuy/brian2/MNIST/'
data_path = '/Users/TyTheWiseGuy/brian2/Brian2STDPMNIST/activity/'
training_ending = '10000'
testing_ending = '10000'
start_time_training = 0
end_time_training = int(training_ending)
start_time_testing = 0
end_time_testing = int(testing_ending)

n_e = 400 # Number of excitatory neurons
n_input = 784 # Number of input neurons
ending = ''

print('Load MNIST\n')
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)

print('Load results')
training_result_monitor = np.load(data_path + 'resultPopVecs' + training_ending + ending + '.npy')
training_input_numbers = np.load(data_path + 'inputNumbers' + training_ending + '.npy')
testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + '.npy')
testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + '.npy')
print(training_result_monitor.shape)
print('')

print('Get assignments\n')
test_results = np.zeros((10, end_time_testing-start_time_testing))
test_results_max = np.zeros((10, end_time_testing-start_time_testing))
test_results_top = np.zeros((10, end_time_testing-start_time_testing))
test_results_fixed = np.zeros((10, end_time_testing-start_time_testing))
assignments = get_new_assignments(training_result_monitor[start_time_training:end_time_training], training_input_numbers[start_time_training:end_time_training])
print(assignments)

counter = 0 
num_tests = end_time_testing / 10000
my_list = [0]
my_array = np.array(my_list)
sum_accuracy = my_array * num_tests
while (counter < num_tests):
    end_time = min(end_time_testing, 10000*(counter+1))
    start_time = 10000*counter
    test_results = np.zeros((10, end_time-start_time))
    print('\nCalculate accuracy of performance:')
    for i in range(end_time - start_time):
        test_results[:,i] = get_recognized_number_ranking(assignments, testing_result_monitor[i+start_time,:])
    
    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accuracy[counter] = correct/float(end_time-start_time) * 100
    print('The accuracy is ', sum_accuracy[counter], '%. The number of incorrect readings is ', len(incorrect), ' out of 10000.')
    counter += 1
#print('Sum response - accuracy --> mean: ', np.mean(sum_accuracy),  '--> standard deviation: ', np.std(sum_accuracy))

b.show()