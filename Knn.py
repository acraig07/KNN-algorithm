# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:28:00 2021

@author: Andrew
"""
import random
import math
import statistics
from statistics import StatisticsError

# Loads the datafile
def load_data(filename):
    dataset = []
    with open(filename, 'r') as file:
        dataset = file.readlines()
    
    return dataset

# Converts the string list into a float list
def convert_str_to_float(dataset):
    count = 0
    newDataset = []
    while count < len(dataset):
        temp = [float(i) for i in dataset[count].split(',')]  
        newDataset.append(temp)
        count+=1
    
    return newDataset

# Randomize the data since the data is sorted by outcome
def randomize_data(dataset):
    random.seed()
    random.shuffle(dataset)
    return dataset

# Cleans the data by removing the first column from the list
# Which was just an index for an data entry    
def clean_data(dataset):
    i = 0
    while i < len(dataset):
       dataset[i].pop(0)
       i+=1
  
    randomize_data(dataset)
    return dataset

# First finds the min and max values fore each data attribute
# Then performs min-max normalization and returns the 
# Min_max normalized dataset
def min_max_normalization(dataset):
    minVal = []
    maxVal = []
    temp = []
    normalizedDataset = dataset.copy()
    for i in range(len(dataset[0])):
        for j in range(len(dataset)):
            temp.append(dataset[j][i])
        valMin = min(temp)
        valMax = max(temp)
        minVal.append(valMin)
        maxVal.append(valMax)
        temp.clear()
    for i in range(len(dataset[0])):
        for j in range(len(dataset)):
            temp1 = (dataset[j][i] - minVal[i]) / (maxVal[i] - minVal[i]) 
            normalizedDataset[j][i] = temp1
 
    return normalizedDataset   

# Splits the dataset into 80% for training and 20% for testing
def split_dataset_train_test(dataset):
    train = len(dataset) * 0.8
    train = int(train)
    trainSet = []
    testSet = []
    
    trainSet = dataset[:train][:]
    testSet = dataset[train:][:]
   
    
    return trainSet, testSet

def euclidean_distance(trainset, testset):
    euclidean = []
    
    #calculates the difference for euclidean distance
    for r in testset:
        for j in range(len(trainset)):
            for i, c in zip(range(len(trainset[j])), r):
                euclidean.append((trainset[j][i] - c) * (trainset[j][i] - c)) 
    N = len(testset[0])
    euclidean2 = [euclidean[n:n+N] for n in range(0, len(euclidean), N)]

    temp = list(map(sum, euclidean2))
    temp1 = []
    
    #completes the eudlidean distance by square rooting the values
    for i in range(len(temp)):
        temp1.append(math.sqrt(temp[i]))
  
    N1 = len(trainset)
    temp1 = [temp1[n:n+N1] for n in range(0, len(temp1), N1)]

    return temp1

#Finds the nearesrt Nieghbor and compares the predicted result to the acutal 
#for all test instances and returns the percentage of how accurate KNN was
def knn(trainset, testset, distance, k):
    
    distanceCopy = []
    distanceCopy = distance.copy()
    minIndexList =[]
    h = k
    
    #creates a list that store the indexes of the min values to be used later on in the knn
    for i in range(len(distanceCopy)):
        for j in range(len(distanceCopy[i])):
            while h > 0:
                result = min(distanceCopy[i])
                indexMin = distanceCopy[i].index(result)
                minIndexList.append(indexMin)
                distanceCopy[i].remove(result)
                h -=1
        h = k
    #formating of list     
    minIndexList = [minIndexList[n:n+k] for n in range(0, len(minIndexList), k)]
    count = 0
    count1 = 0
    temp = []
    
    #increases the value of the index in the list 
    #due to how the values were origionally storred
    #only increase the index if needed 
    while count1 < len(minIndexList):
        check = minIndexList[count1][0]
        while count < k:
            x = minIndexList[count1].pop(0)
            if (x < check):
                temp.append(x)
            else:
                temp.append(x + count)
            count += 1
        count = 0
        count1 += 1
        
    temp = [temp[n:n+k] for n in range(0, len(temp), k)]
    classIndex = len(trainset[0]) - 1
    nearestNieghbor = []
    
    #find the neatest nieghbor and stores its classification 
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            nearestNieghbor.append(trainset[temp[i][j]][classIndex])
            
    #formating of nearest nieghbor list
    nearestNieghbor = [nearestNieghbor[n:n+k] for n in range(0, len(nearestNieghbor), k)]
    results = []
    
    #list used for tie breaker if the modes of the knn are similar
    joinedList = trainset + testset

    tie = []
    for i in range(len(trainset)):
        tie.append(joinedList[i][classIndex])
    
    tie2 = statistics.mode(tie) #the tie breaker is the most common classifier in the data set
    tie3 = []
    #finds the predicted classifcation 
    for i in range(len(nearestNieghbor)):
        tie3.append(statistics.multimode(nearestNieghbor[i])) #sets the tie breaker to be the first value of the multimode
        try:
            results.append(statistics.mode(nearestNieghbor[i])) #appends the predicted result for classification
        except StatisticsError: #if the mode is not unique goes to tie breakers 
            results.append(tie3[i][0]) 
            #results.append(tie2)
    
    #checks if the predicted result of knn is correct
    correct = 0
    total = len(results)
    for i in range(len(testset)):
        are_close = math.isclose(testset[i][classIndex], results[i])
        if (are_close == True):
            correct += 1
    
    percentage = correct/total * 100 #calculates the precentage of the accuracy of the knn
  
    return percentage
    
def execute_knn(filename, k, amount):
    lists = []
    for i in range(0,amount):
       f, g = split_dataset_train_test(min_max_normalization(clean_data(convert_str_to_float(load_data(filename)))))
       result = knn(f, g, euclidean_distance(f, g), k)
       lists.append(result)
    print ("Executed KNN %d times with k = %d" % (amount, k))
    print ("KNN had an average accuracy of: ")
    print(sum(lists)/len(lists))
   
if __name__ == '__main__':
    filename = 'glass.data'
    k = 39
    amount = 1000
    execute_knn(filename, k, amount)
   
    
   