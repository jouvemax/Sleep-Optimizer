#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np;
import matplotlib.pyplot as plt
from pomegranate import *
import itertools
import collections
from scipy.stats import poisson
from scipy.stats import norm
import math


# Reading the csv file
test = np.genfromtxt('sleepdata.csv', delimiter=';', dtype=None, names=('start', 'end', 'quality', 'timeInBed', 'wake up', 'note', 'heart rate', 'activity'))

#Removing the title and taking only the quality of sleep and the time in bed
test = test[1:len(test)]
quality = test['quality']
time = test['timeInBed']


# method for converting the quality (string) into int
def f1 (string):
    return int(string[0:len(string)-1])

# method for converting the sleeping time in hours (string) into minutes (int)
def f2 (string):
    string  = string.decode('utf8')
    if string[0:2].find(':') != -1:
        hours = int(string[0])
        minutes = int (string[2:4])
    else:
        hours = int(string[0:2])
        minutes = int (string[3:5])
    return hours * 60 + minutes
    

qualityFormatted = np.array([f1(xi) for xi in quality])
timeFormatted = np.array([f2(xi) for xi in time])

dataSet = np.zeros((887, 2))
dataSet[:, 0] = qualityFormatted
dataSet[:, 1] = timeFormatted



# filtering noise ("nap"). We only keep sleeping duration > 240 minutes
# Our data set is now usable
dataSet = dataSet[dataSet[:,1] > 240,:]



# All the possible sleeping duration intervals (in minutes)
# lower one is excluded and upper is included
#0-360       
#6 - 6:30   360-390   
#6:30 - 7   390-420 
#7 - 7:30   420-450
#7:30 - 8   450-480
#8 - 8:30   480-510
#8:30 - 9   510-540
#9 - 9:30   540-570
#9:30 - 10  570-600
#>10   600-1000

# ALl the possible sleeping quality intervals (in %)
# lower one is excluded and upper is included
# 0-10     DELETED BECAUSE THE USER WAS NEVER IN THIS STATE
# 10-20    DELETED BECAUSE THE USER WAS NEVER IN THIS STATE
# 20-30
# 30-40
# 40-50
# 50-60
# 60-70
# 70-80
# 80-90
# 90-100

# This method builds our markov chain for the hidden states
# markovChain[i][j] corresponds to the probability of going from states i to j
def buildingMarkovChainQualityOfSleep():
    markovChain = np.zeros((10, 10))
    for k in range(10):
        counter = 0
        lower = 10 * k
        upper = 10 * (k+1)
        for i in range(len(dataSet[:, 0])-1):
            quality = dataSet[:, 0][i]
            if quality <= upper and quality > lower:
                counter +=1
                nextQuality = dataSet[:, 0][i+1]
                index = (nextQuality-1)/10
                markovChain[k][int(index)] +=1
        if np.sum(markovChain[k]) != 0:
            markovChain[k] /= counter
    return markovChain

# This method builds the matrix corresponding to emission probabilities
# matrix[i][j] is the probability of being in hidden state i and seeing observation j
def buildingEmissionProbabilities():
    matrix = np.zeros((10, 10))
    for k in range(10):
        counter = 0
        lower = 10 * k
        upper = 10 * (k+1)
        for i in range(len(dataSet[:, 0])-1):
            quality = dataSet[:, 0][i]
            if quality <= upper and quality > lower:
                counter +=1
                timeInBed = dataSet[:, 1][i]
                index = int((timeInBed-1)/30) - 11
                if index < 0:
                    index = 0
                if index >= 10:
                    index = 9
                matrix[k][index] += 1
                
        if np.sum(matrix) != 0:
            matrix[k] /= counter
                    
                
    return matrix

# Method computing a vector that corresponds to the starting state probabilities
# The probability of starting in a state is just the probability of seeing this starting state in the whole dataset
def computingStartingState():
    startingDistribution = np.zeros(10)
    for quality in dataSet[:,0]:
        index = quality/10 -1
        if index == -1:
            index = 0
        startingDistribution[int(index)] += 1
        
    return startingDistribution/len(dataSet[:,0])

# Matrix that corresponds to the hidden states transition matrix
matrixHidden = buildingMarkovChainQualityOfSleep()
# Matrix that corresponds to the emission probability matrix
matrixEmission = buildingEmissionProbabilities()


# IMPORTANT NOTE
# States corresponding to 0-10% and 10-20% were deleted because the user was never in thoses states
# WE THUS HAVE 8 HIDDEN STATES IN TOTAL
# When we create the model we thus never use matrixEmission[0], matrixEmission[1], matrixHidden[0] and matrixHidden[1]
# because it corresponds to state that are never reached.

#Define all the hidden states and their emission probability to each observation
d1 = DiscreteDistribution({'0-360' : matrixEmission[2][0], '360-390' : matrixEmission[2][1], '390-420' : matrixEmission[2][2], '420-450' : matrixEmission[2][3], '450-480' : matrixEmission[2][4], '480-510' : matrixEmission[2][5],'510-540' : matrixEmission[2][6],'540-570' : matrixEmission[2][7], '570-600' : matrixEmission[2][8], '600-1000' : matrixEmission[2][9]})
d2 = DiscreteDistribution({'0-360' : matrixEmission[3][0], '360-390' : matrixEmission[3][1], '390-420' : matrixEmission[3][2], '420-450' : matrixEmission[3][3], '450-480' : matrixEmission[3][4], '480-510' : matrixEmission[3][5],'510-540' : matrixEmission[3][6],'540-570' : matrixEmission[3][7], '570-600' : matrixEmission[3][8], '600-1000' : matrixEmission[3][9]})
d3 = DiscreteDistribution({'0-360' : matrixEmission[4][0], '360-390' : matrixEmission[4][1], '390-420' : matrixEmission[4][2], '420-450' : matrixEmission[4][3], '450-480' : matrixEmission[4][4], '480-510' : matrixEmission[4][5],'510-540' : matrixEmission[4][6],'540-570' : matrixEmission[4][7], '570-600' : matrixEmission[4][8], '600-1000' : matrixEmission[4][9]})
d4 = DiscreteDistribution({'0-360' : matrixEmission[5][0], '360-390' : matrixEmission[5][1], '390-420' : matrixEmission[5][2], '420-450' : matrixEmission[5][3],'450-480' : matrixEmission[5][4], '480-510' : matrixEmission[5][5],'510-540' : matrixEmission[5][6],'540-570' : matrixEmission[5][7], '570-600' : matrixEmission[5][8], '600-1000' : matrixEmission[5][9]})
d5 = DiscreteDistribution({'0-360' : matrixEmission[6][0], '360-390' : matrixEmission[6][1], '390-420' : matrixEmission[6][2], '420-450' : matrixEmission[6][3], '450-480' : matrixEmission[6][4], '480-510' : matrixEmission[6][5],'510-540' : matrixEmission[6][6],'540-570' : matrixEmission[6][7], '570-600' : matrixEmission[6][8], '600-1000' : matrixEmission[6][9]})
d6 = DiscreteDistribution({'0-360' : matrixEmission[7][0], '360-390' : matrixEmission[7][1], '390-420' : matrixEmission[7][2], '420-450' : matrixEmission[7][3], '450-480' : matrixEmission[7][4], '480-510' : matrixEmission[7][5],'510-540' : matrixEmission[7][6],'540-570' : matrixEmission[7][7], '570-600' : matrixEmission[7][8], '600-1000' : matrixEmission[7][9]})
d7 = DiscreteDistribution({'0-360' : matrixEmission[8][0], '360-390' : matrixEmission[8][1], '390-420' : matrixEmission[8][2], '420-450' : matrixEmission[8][3], '450-480' : matrixEmission[8][4], '480-510' : matrixEmission[8][5],'510-540' : matrixEmission[8][6],'540-570' : matrixEmission[8][7], '570-600' : matrixEmission[8][8], '600-1000' : matrixEmission[8][9]})
d8 = DiscreteDistribution({'0-360' : matrixEmission[9][0], '360-390' : matrixEmission[9][1], '390-420' : matrixEmission[9][2], '420-450' : matrixEmission[9][3], '450-480' : matrixEmission[9][4], '480-510' : matrixEmission[9][5],'510-540' : matrixEmission[9][6],'540-570' : matrixEmission[9][7], '570-600' : matrixEmission[9][8], '600-1000' : matrixEmission[9][9]})

#Defining each hidden state
s1 = State(d1, name="s1")
s2 = State(d2, name="s2")
s3 = State(d3, name="s3")
s4 = State(d4, name="s4")
s5 = State(d5, name="s5")
s6 = State(d6, name="s6")
s7 = State(d7, name="s7")
s8 = State(d8, name="s8")

startingState = computingStartingState()

# Creating the Hidden Markov Model
model = HiddenMarkovModel('Sleeping Hidden Markov Chain')

# Adding states to the model
model.add_states([s1, s2, s3, s4, s5, s6, s7, s8])

# Adding the starting state probability
model.add_transition(model.start, s1, startingState[2])
model.add_transition(model.start, s2, startingState[3])
model.add_transition(model.start, s3, startingState[4])
model.add_transition(model.start, s4, startingState[5])
model.add_transition(model.start, s5, startingState[6])
model.add_transition(model.start, s6, startingState[7])
model.add_transition(model.start, s7, startingState[8])
model.add_transition(model.start, s8, startingState[9])

# Adding transitions from state 1 to all others states
model.add_transition(s1, s1, matrixHidden[2][2])
model.add_transition(s1, s2, matrixHidden[2][3])
model.add_transition(s1, s3, matrixHidden[2][4])
model.add_transition(s1, s4, matrixHidden[2][5])
model.add_transition(s1, s5, matrixHidden[2][6])
model.add_transition(s1, s6, matrixHidden[2][7])
model.add_transition(s1, s7, matrixHidden[2][8])
model.add_transition(s1, s8, matrixHidden[2][9])

# Adding transitions from state 2 to all others states
model.add_transition(s2, s1, matrixHidden[3][2])
model.add_transition(s2, s2, matrixHidden[3][3])
model.add_transition(s2, s3, matrixHidden[3][4])
model.add_transition(s2, s4, matrixHidden[3][5])
model.add_transition(s2, s5, matrixHidden[3][6])
model.add_transition(s2, s6, matrixHidden[3][7])
model.add_transition(s2, s7, matrixHidden[3][8])
model.add_transition(s2, s8, matrixHidden[3][9])

# Adding transitions from state 3 to all others states
model.add_transition(s3, s1, matrixHidden[4][2])
model.add_transition(s3, s2, matrixHidden[4][3])
model.add_transition(s3, s3, matrixHidden[4][4])
model.add_transition(s3, s4, matrixHidden[4][5])
model.add_transition(s3, s5, matrixHidden[4][6])
model.add_transition(s3, s6, matrixHidden[4][7])
model.add_transition(s3, s7, matrixHidden[4][8])
model.add_transition(s3, s8, matrixHidden[4][9])

# Adding transitions from state 4 to all others states
model.add_transition(s4, s1, matrixHidden[5][2])
model.add_transition(s4, s2, matrixHidden[5][3])
model.add_transition(s4, s3, matrixHidden[5][4])
model.add_transition(s4, s4, matrixHidden[5][5])
model.add_transition(s4, s5, matrixHidden[5][6])
model.add_transition(s4, s6, matrixHidden[5][7])
model.add_transition(s4, s7, matrixHidden[5][8])
model.add_transition(s4, s8, matrixHidden[5][9])

# Adding transitions from state 5 to all others states
model.add_transition(s5, s1, matrixHidden[6][2])
model.add_transition(s5, s2, matrixHidden[6][3])
model.add_transition(s5, s3, matrixHidden[6][4])
model.add_transition(s5, s4, matrixHidden[6][5])
model.add_transition(s5, s5, matrixHidden[6][6])
model.add_transition(s5, s6, matrixHidden[6][7])
model.add_transition(s5, s7, matrixHidden[6][8])
model.add_transition(s5, s8, matrixHidden[6][9])

# Adding transitions from state 6 to all others states
model.add_transition(s6, s1, matrixHidden[7][2])
model.add_transition(s6, s2, matrixHidden[7][3])
model.add_transition(s6, s3, matrixHidden[7][4])
model.add_transition(s6, s4, matrixHidden[7][5])
model.add_transition(s6, s5, matrixHidden[7][6])
model.add_transition(s6, s6, matrixHidden[7][7])
model.add_transition(s6, s7, matrixHidden[7][8])
model.add_transition(s6, s8, matrixHidden[7][9])

# Adding transitions from state 7 to all others states
model.add_transition(s7, s1, matrixHidden[8][2])
model.add_transition(s7, s2, matrixHidden[8][3])
model.add_transition(s7, s3, matrixHidden[8][4])
model.add_transition(s7, s4, matrixHidden[8][5])
model.add_transition(s7, s5, matrixHidden[8][6])
model.add_transition(s7, s6, matrixHidden[8][7])
model.add_transition(s7, s7, matrixHidden[8][8])
model.add_transition(s7, s8, matrixHidden[8][9])

# Adding transitions from state 8 to all others states
model.add_transition(s8, s1, matrixHidden[9][2])
model.add_transition(s8, s2, matrixHidden[9][3])
model.add_transition(s8, s3, matrixHidden[9][4])
model.add_transition(s8, s4, matrixHidden[9][5])
model.add_transition(s8, s5, matrixHidden[9][6])
model.add_transition(s8, s6, matrixHidden[9][7])
model.add_transition(s8, s7, matrixHidden[9][8])
model.add_transition(s8, s8, matrixHidden[9][9])

# The model is finally created
model.bake()



# This method take an array that corresponds to the distribution of bein in all hidden states
# It then plots the distribution of being in each hidden states
# It also tell you the probability of having a sleep quality between 80% and 100%
# Finally it tells you what is your average quality of sleep for this given distribution
def plotDistribution(distribution):
    height = distribution[0:len(distribution)-2]
    keys = np.arange(8)
    y_pos = np.arange(len(keys))
    
    plt.bar(y_pos, height)
    ticks = ['20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

    #Custom ticks
    plt.xticks(y_pos, ticks, color='red', fontweight='bold', rotation = 45, fontsize='10', horizontalalignment='right')
   
    # Custom Axis title
    plt.xlabel('Sleep quality Distribution', fontweight='bold', color = 'black', fontsize='10', horizontalalignment='center')

    print("The probability of having a quality of sleep between 80% and 100% is:")
    print("%d %%" %int((height[6] + height[7]) * 100))
    meanQuality = computeAverageQuality(distribution)
    print("You can expect to have a sleeping quality of %d %%" %meanQuality)
    plt.show()

# This method takes an array corresponding to the mean of each distribution. 
# means[0] corresponds to the mean quality by having a '0-360' min of sleep during your next night
# ...
# means[9] corresponds to the mean quality by having a '600-1000' min of sleep during your next night
def plotOptimalSleepDuration(means):
    timeInterval = ['0-360', '360-390', '390-420', '420-450', '450-480', '480-510','510-540', '540-570', '570-600', '600-1000']

    y_pos = np.arange(len(means))
    
    plt.plot(y_pos, means, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12)


    #Custom ticks
    plt.xticks(y_pos, timeInterval, color='red', fontweight='bold', rotation = 45, fontsize='10', horizontalalignment='right')
   
    # Custom Axis title
    plt.xlabel('Sleep quality (%) as a function of the duration of your next night (min)', fontweight='bold', color = 'black', fontsize='10', horizontalalignment='center')
    plt.show()
# This method computes the average quality of sleep given a the distribution of being in each hidden state
def computeAverageQuality(distribution):
    quality = [25, 35, 45, 55, 65, 75, 85, 95]
    distribution = distribution[0:len(distribution) - 2]
    mean = 0
    return sum(quality * distribution)


sequence = ['390-420', '450-480', '390-420', '480-510']

# This method asks the user to enter the sequence of sleep duration during the last few days
# sequence is an array containing the sleep duration of the user for the last days.
# sequence[0] is the duration of  the first night of the sequence and sequence[len(sequence) - 1] is the duration
# of the last night of the user
# 
# The program then tells the user:
# -what are the statistics for the last night he had (distribution of quality of sleep, 
# average quality of sleep and probability of having a quality of sleep between 80% and 100%)
#
# -what are the predicted statistics for his/her next night(same statistics as for last night)
#
# -We then generate the distribution for the next day for all possible sleep duration intervals.
# We can then show the user how he can maximize his sleep quality for the next night
# We show him/her all the sleeping duration interval and their corresponding expected sleeping quality so he can choose
# an interval that is convenient for him/her
def analyzingSleepingData(sequence):
    forwardArray = np.e ** np.array(model.forward(sequence))
    lastDay = forwardArray[len(forwardArray) - 1]
    normalizedDistribution = lastDay/sum(lastDay)
    print("Last night")
    plotDistribution(normalizedDistribution)
    
    nextDayDistribution = normalizedDistribution @ model.dense_transition_matrix()
    print("For this night")
    plotDistribution(nextDayDistribution)
    
    timeInterval = ['0-360', '360-390', '390-420', '420-450', '450-480', '480-510','510-540', '540-570', '570-600', '600-1000']
    
    means = []
    for k in timeInterval:
        newSequence = sequence.copy()
        newSequence.append(k)
        forwardArray2 = np.e ** np.array(model.forward(newSequence))
        nextDay = forwardArray2[len(forwardArray2) - 1]
        normalizedDistribution2 = nextDay/sum(nextDay)
    
        
        meanQuality = computeAverageQuality(normalizedDistribution2)
        means.append(meanQuality)
            
    print("We have computed all possibles sleeping duration for the next night and found the ones that will increase your sleeping quality the most")
    plotOptimalSleepDuration(means)
    
analyzingSleepingData(sequence)











#The following code was not presented during the presentation
#We generated all possible sequences of sleep for n days (user can choose n but computation becomes too long for n = 6)
#We then computed the probability distribution of the total amount of sleep he would get during n days
#We then compared this distribution to the Poisson distribution with estimated parameter (maximum likelihood estimator)
#and the normal Distribution with estimated distribution
def plottingSleepingDistribution(n):
    #the different sleeping time interval
    timeInterval = ['0-360', '360-390', '390-420', '420-450', '450-480', '480-510','510-540', '540-570', '570-600', '600-1000']
    timeAverage = [345, 375, 405, 435, 465, 495, 525, 555, 585, 615]

    #generating all possible sequence of sleeping time
    #then computing the corresponding total time of sleep in those 5 days
    #and assigning the corresponding probability using the model.probability(sequence) method
    testArray = ["".join(seq) for seq in itertools.product("0123456789", repeat=n)]
    distribution = {}
    for k in range(len(testArray)):
        s = testArray[k]
        s = list(s)
        Seq = []
        Time = []
        for i in s:
            Seq.append(timeInterval[int(i)])
            Time.append(timeAverage[int(i)])
        totalTime = sum(Time)
        probability = model.probability(Seq)
        previous = distribution.get(totalTime)
        if previous == None:
            distribution.update({totalTime : probability})
        else:
            distribution.update({totalTime:probability + previous})

    #sorting the distribution in increasing order
    distribution = collections.OrderedDict(sorted(distribution.items()))

    # Basic plot
    height = distribution.values()
    keys = list(distribution.keys())
    bars = []
    for i in range(0, len(keys), 5):
        bars.append(keys[i])
    
    y_pos = np.arange(len(keys))
    plt.bar(y_pos, height)

    #Custom ticks
    plt.xticks(np.arange(0, len(keys), 5), bars, color='red', fontweight='bold', fontsize='10', horizontalalignment='right')

    
    # Custom Axis title
    plt.xlabel('Distribution of the time of sleep in n days', fontweight='bold', color = 'black', fontsize='10', horizontalalignment='center')
    print(" n = %d" %n)
    values = list(height)
    plt.show()
    
          #estimating mean
    mean = 0
    for k in range(len(values)):
        mean += values[k] * keys[k]
    #estimating variance
    variance = 0
    for k in range(len(values)):
        variance += values[k] * (keys[k] - mean)**2
   
    def poissonPmf(x):
        return poisson.pmf(x,mean) * 30

    def normPdf(x):
        return norm.pdf(x, mean, math.sqrt(variance)) * 30
    
    # Basic plot
    height = list(map(poissonPmf, distribution.keys())) 
    keys = list(distribution.keys())
    bars = []
    for i in range(0, len(keys), 5):
        bars.append(keys[i])
    
    y_pos = np.arange(len(keys))
    plt.bar(y_pos, height)

    # Custom ticks
    plt.xticks(np.arange(0, len(keys), 5), bars, color='red', fontweight='bold', fontsize='10', horizontalalignment='right')
    # Custom Axis title
    plt.xlabel('Poisson distribution with estimated parameters', fontweight='bold', color = 'black', fontsize='10', horizontalalignment='center')
    plt.show()
    
    # Basic plot
    height = list(map(normPdf, distribution.keys())) 
    keys = list(distribution.keys())
    bars = []
    for i in range(0, len(keys), 5):
        bars.append(keys[i])
    
    y_pos = np.arange(len(keys))
    plt.bar(y_pos, height)

    # Custom ticks
    plt.xticks(np.arange(0, len(keys), 5), bars, color='red', fontweight='bold', fontsize='10', horizontalalignment='right')
    # Custom Axis title
    plt.xlabel('Normal distribution with estimated parameters', fontweight='bold', color = 'black', fontsize='10', horizontalalignment='center')
    plt.show()
    print("mean = %d" %mean)
    print("variance = %d" %variance)
    
plottingSleepingDistribution(4)


# In[ ]:





# In[ ]:





# In[ ]:




