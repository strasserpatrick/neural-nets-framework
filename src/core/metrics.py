import numpy as np

def recall(confusion_matrix, label):
    return confusion_matrix[label][label] / np.sum(confusion_matrix[label])
    
def precision(confusion_matrix, label):
    return confusion_matrix[label][label] / np.sum(confusion_matrix.T[label])

def f1_score(confussion_matrix, label):
    return 2*((precision(confussion_matrix, label) * recall(confussion_matrix, label))/
              (precision(confussion_matrix, label) + recall(confussion_matrix, label)))

def accuracy(confussion_matrix):
    return np.trace(confussion_matrix) / np.sum(confussion_matrix)

def getMeanConfussionMatrix(list_of_cms):
    summed_cm = np.zeros(list_of_cms[0].shape)
    for cm in list_of_cms:
        summed_cm += cm
    return summed_cm/len(list_of_cms)