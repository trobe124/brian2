#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 20:06:02 2023

@author: TyTheWiseGuy
"""
import numpy as np
import pandas as pd
import sys
import matplotlib
import time
import os.path
import scipy
import cloudpickle as pickle
import brian2 as b
import brian2tools
from struct import unpack
from brian2 import *
from brian2tools import *
from brian2.units.allunits import * 

ms = second/1000
b.defaultclock.dt = 0.5*b.ms
b.codegen.target = 'cython'
b.prefs.codegen.loop_invariant_optimisations = True
b.prefs.codegen.cpp.extra_compile_args_gcc = ['-ffast-math -march=native']

# Location of MNIST data
MNIST_data_path = '/Users/TyTheWiseGuy/brian2/MNIST/'

#-----------
#-functions-
#-----------

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

def get_matrix_from_file(fileName, n_src, n_tgt):
    readout = np.load(fileName, allow_pickle=True)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr
            
            #####
def save_connections():
    print('save connections')
    conn = connections['XeAe']
    connListSparse = zip(conn.i, conn.j, conn.w)
    np.save(data_path + 'weights/XeAe', connListSparse)

def save_theta():
    print('save theta')
    np.save(data_path + 'weights/theta_A', neuron_groups['Ae'].theta)
        
def normalize_weights():

    len_source = len(connections['XeAe'].source)
    len_target = len(connections['XeAe'].target)
    connection = np.zeros((len_source, len_target))
    connection[connections['XeAe'].i, connections['XeAe'].j] = connections['XeAe'].w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = 78./colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    connections['XeAe'].w = temp_conn[connections['XeAe'].i, connections['XeAe'].j]

def get_2d_input_weights():
    name = 'XeAe'            
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt * n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = connections[name][:]
    weight_matrix = np.copy(connMatrix)
    #
    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
            rearranged_weights[i*n_in_sqrt:(i+1)*n_in_sqrt, j*n_in_sqrt:(j+1)*n_in_sqrt] = \
                weight_matrix[:, i+j*n_e_sqrt]. reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights
#
#
def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig
    
def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num):
    num_evaluations = int(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b.ylim(ymax = 100)
    b.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance
    
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments
    
    
#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print('time needed to load training set:', end - start)
 
start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print('time needed to load test set:', end - start)


#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = True

np.random.seed(0)
data_path = '/Users/TyTheWiseGuy/brian2/Brian2STDPMNIST/'
if test_mode:
    num_examples = 10000
else:
    num_examples = 60000 * 3

n_input = 784
n_e = 400
n_i = n_e 
single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:    
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:    
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

v_rest_e = -65. * b.mV 
v_rest_i = -60. * b.mV 
v_reset_e = -65. * b.mV
v_reset_i = 'v=-45.*mV'
v_thresh_e = '-52.*mV'
v_thresh_i = 'v>-40.*mV'
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b.mV
v_thresh_e = '(v>(theta - offset + -52.*mV)) and (timer>refrac_e)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms): volt (unless refractory)
        I_synE = ge * nS * -v: amp
        I_synI = gi * nS * (-100.*mV-v): amp
        dge/dt = -ge/(1.0*ms): 1
        dgi/dt = -gi/(2.0*ms): 1
        '''
if test_mode:
    neuron_eqs_e += '\n theta: volt'
else:
    neuron_eqs_e += '\n dtheta/dt = -theta / (tc_theta): volt'
neuron_eqs_e += '\n dtimer/dt = 0.1: second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms): volt (unless refractory)
        I_synE = ge * nS * -v: amp
        I_synI = gi * nS * (-85.*mV-v): amp
        dge/dt = -ge/(1.0*ms): 1
        dgi/dt = -gi/(2.0*ms): 1
        '''
eqs_stdp_ee = '''
                post2before: 1.0
                dpre/dt = -pre/(tc_pre_ee): 1.0 (event-driven)
                dpost1/dt = -post1/(tc_post_1_ee): 1.0 (event-driven)
                dpost2/dt = -post2/(tc_post_2_ee): 1.0 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w-nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'
    
b.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
record_spikes = {}

result_monitor = np.zeros((update_interval,n_e))
if test_mode:
    result_monitor = np.zeros((num_examples, n_e))

neuron_groups['Ae'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups['Ai'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')


#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 
print('create neuron group A')
    
neuron_groups['Ae'].v = v_rest_e - 40.*b.mV
neuron_groups['Ai'].v = v_rest_i - 40.*b.mV
if test_mode:
    neuron_groups['Ae'].theta = np.load(data_path + 'weights/theta_A.npy') * b.volt
else:
    neuron_groups['Ae'].theta = np.ones((n_e)) * 20.0*b.mV
    
print('create recurrent connections')
weightMatrix = get_matrix_from_file(data_path + 'random/AeAi.npy', n_e, n_i)
connections['AeAi'] = b.Synapses(neuron_groups['Ae'], neuron_groups['Ai'], model='w:1', on_pre = 'ge_post+=w')
connections['AeAi'].connect(True)
connections['AeAi'].w = weightMatrix[connections['AeAi'].i, connections['AeAi'].j]

weightMatrix = get_matrix_from_file(data_path + 'random/AiAe.npy', n_i, n_e)
connections['AiAe'] = b.Synapses(neuron_groups['Ai'], neuron_groups['Ae'], model='w:1', on_pre = 'gi_post+=w')
connections['AiAe'].connect(True)
connections['AiAe'].w = weightMatrix[connections['AiAe'].i, connections['AiAe'].j]

print('create monitors for Ae')
rate_monitors['Ae'] = b.PopulationRateMonitor(neuron_groups['Ae'])
rate_monitors['Ai'] = b.PopulationRateMonitor(neuron_groups['Ai'])
spike_counters['Ae'] = b.SpikeMonitor(neuron_groups['Ae'])
    

if record_spikes:
    spike_monitors['Ae'] = b.SpikeMonitor(neuron_groups['Ae'])
    spike_monitors['Ai'] = b.SpikeMonitor(neuron_groups['Ai'])

#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 
input_groups['Xe'] = b.PoissonGroup(n_input, 0*Hz)

print('Create connections between X and A')
if test_mode:
    weightMatrix = get_matrix_from_file(data_path + 'weights/XeAe.npy', n_input, n_e)
else:
    weightMatrix = get_matrix_from_file(data_path + 'random/XeAe.npy', n_input, n_e)
model = 'w:1'
pre = 'ge_post += w'
post = ''

if not test_mode:
    print('Create STDP for connection XeAe')
    model += eqs_stdp_ee
    pre += '; ' + eqs_stdp_pre_ee
    post = eqs_stdp_post_ee
    
connections['XeAe'] = b.Synapses(input_groups['Xe'], neuron_groups['Ae'], model=model, on_pre=pre, on_post=post)
    
minDelay = 0*b.ms
maxDelay = 10*b.ms
deltaDelay = maxDelay - minDelay

connections['XeAe'].connect(True)
connections['XeAe'].delay = "minDelay + rand() * deltaDelay"
connections['XeAe'].w = weightMatrix[connections['XeAe'].i, connections['XeAe'].j]

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 

net = Network()
for obj_list in [neuron_groups, input_groups, connections, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(n_e)
input_numbers = [0] * num_examples
input_groups['Xe'].rates = 0*Hz
net.run(0*second)
j = 0
while j < (int(num_examples)):
    if test_mode:
        rate = testing['x'][j%10000,:,:].reshape((n_input)) / 8.*input_intensity
    else:
        normalize_weights()
        rate = training['x'][j%60000,:,:].reshape((n_input)) / 8.*input_intensity
    input_groups['Xe'].rates = rate*Hz
    net.run(single_example_time, report='text')
    
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
        input_groups['Xe'].rates = 0 * Hz
        net.run(resting_time)
    else:
        if test_mode:
            result_monitor[j,:] = current_spike_count
            input_numbers[j] = testing['y'][j%10000][0]
        if j % 100 == 0 and j > 0:
            print('runs done:', j, 'of', int(num_examples))
        input_groups['Xe'].rates = 0 * Hz
        net.run(resting_time)
        input_intensity = start_input_intensity
        j += 1
        
#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print('save results')
if not test_mode:
    save_theta()
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
    

#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
if rate_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        b.subplot(len(rate_monitors), 1, i)
        b.plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
        b.title('Rates of population ' + name)
    
if spike_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        b.subplot(len(spike_monitors), 1, i)
        b.raster_plot(spike_monitors[name])
        b.title('Spikes of population ' + name)
        
if spike_counters:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_counters):
        b.subplot(len(spike_counters), 1, i)
        b.plot(spike_counters['Ae'].count[:])
        b.title('Spike count of population ' + name)
        
plot_2d_input_weights()

plot.figure(5)

subplot(3,1,1)

brian_plot(connections['XeAe'].w)
subplot(3,1,2)

brian_plot(connections['AeAi'].w)

subplot(3,1,3)

brian_plot(connections['AiAe'].w)


plot.figure(6)

subplot(3,1,1)

brian_plot(connections['XeAe'].delay)
subplot(3,1,2)

brian_plot(connections['AeAi'].delay)

subplot(3,1,3)

brian_plot(connections['AiAe'].delay)


b.ioff()
b.show()