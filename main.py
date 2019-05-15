import sys
import numpy as np
from enum import Enum
import random as rand
from scipy import stats

# Define Parameters for matrix
rows = 3
columns = 8
wPerceptron = [[0.0] * columns] * rows
wSVM = [[0.0] * columns] * rows
wPA = [[0.0] * columns] * rows
# Const Perceptron
CP = 1
# Consts for SVM
etaSVM = 1.5
lambdaSVM = 0.5
# Consts for PA
taoPA = 0


def CalculateErrorRate(DataSet_x, DataSet_y, weights):
    """
    This function calculate the error rate
    :param DataSet_x: a given data set x
    :param DataSet_y: the corresponded data y
    :param weights: the weight function
    :return: print the error rate
    """
    counter1 = 0
    for c, line in enumerate(DataSet_x):
        my_y_hat = y_HatCalculation(weights, line)
        if my_y_hat != (DataSet_y[c]):
            counter1 += 1
    print("The Error Rate  is " + str(counter1 / len(DataSet_y)))


def stripAndSplit(line):
    """
    This Function parsed the data set, line by line
    :param line: a given line to parsed
    :return: parsed line - as float list
    """
    line = line.strip()
    line = line.split(',')

    if line[0] == 'M':
        line[0] = Gender.Male.value
    elif line[0] == 'F':
        line[0] = Gender.Female.value
    else:
        line[0] = Gender.Infant.value
    temp_float = [float(i) for i in line]
    # In case we want to normalized the line with "Z-SCORE" function
    # temp_float = stats.mstats.zscore(temp_float, ddof=1)
    return temp_float


def y_HatCalculation(w, line):
    """
    :param w: the weight function
    :param line: test line with features
    :return: the argument that got the maximum between the matrix and the given line
    """
    return np.argmax(np.dot(w, line))


def splitterForLines(line_y):
    """

    :param line_y: a given line y
    :return: the parsed y to float
    """
    line_y = float(line_y.strip())
    return line_y


def calcPerceptron(line_per, original_y, y_hat):
    """

    :param line_per: the given test
    :param original_y: the original Y
    :param y_hat: the predicted y_hat
    :return: updated weight function of Perceptron
    """
    line_per = np.multiply(line_per, CP)
    wPerceptron[int(original_y)] = [x1 + y1 for x1, y1 in zip(wPerceptron[int(original_y)], line_per)]
    wPerceptron[y_hat] = [x2 - y2 for x2, y2 in zip(wPerceptron[int(y_hat)], line_per)]


def calcSVM(line_SVM, original_y, y_hat):
    """
    Function to calculate the SVM Weight function
    :param line_SVM: a given test for SVM predicction
    :param original_y: the original Y
    :param y_hat: the predicted y_hat
    :return: updated weight function of SVM
    """
    line_SVM = np.multiply(line_SVM, etaSVM)
    oneMinus = 1 - etaSVM * lambdaSVM

    # Calc for y
    wSVM[int(original_y)] = np.multiply(oneMinus, wSVM[int(original_y)])
    wSVM[int(original_y)] = np.add(wSVM[int(original_y)], line_SVM)

    # Calc for y_hat
    wSVM[y_hat] = np.multiply(oneMinus, wSVM[y_hat])
    wSVM[y_hat] = np.subtract(wSVM[y_hat], line_SVM)

    # Calc for not y and not y hat (all the others, we have 1 only..)
    for i in range(rows):
        if i != y_hat and i != y:
            wSVM[i] = np.multiply(oneMinus, wSVM[i])


def calcSVMForAllVectors():
    """
    :return: the weight function for SVM algorithm in case the y equals y hat
    """
    oneMinus = 1 - etaSVM * lambdaSVM
    for i in range(rows):
        wSVM[i] = np.multiply(oneMinus, wSVM[i])


def calcPA(linePA, y, y_hat_pa):
    """
    This Function calc the PA weight matrix and update it each iteration
    :param linePA:  the given test
    :param y: the original Y
    :param y_hat_pa: the predicted y
    :return: updated weight function for PA
    """
    linePA = np.multiply(linePA, taoPA)
    wPA[y] = np.add(wPA[y], linePA)
    wPA[y_hat_pa] = np.subtract(wPA[y_hat_pa], linePA)


def printAlgorithms(test_x):
    """
    This Function classify the test with each algorithm
    :param test_x: a given test to print its class
    :return: print the class of the test with each algorithm prediction int the requested format
    """
    Perceptron_y_hat = y_HatCalculation(wPerceptron, test_x)
    SVM_y_hat = y_HatCalculation(wSVM, test_x)
    PA = y_HatCalculation(wPA, test_x)
    print("perceptron: " + str(Perceptron_y_hat) + ", svm: " + str(SVM_y_hat) + ", pa: " + str(PA))


def minMaxAlgorithm(dataSet):
    """
    :param dataSet: our data set.
    :return: normalized data set with min & max algorithm to normalized data.
    """
    normalized = np.array([np.array(xi) for xi in dataSet])
    normalized = normalized.transpose()
    for i, n in enumerate(normalized):
        min_num = min(n)
        max_num = max(n)
        if max_num - min_num != 0:
            normalized[i] = np.divide(np.subtract(n, min_num), (max_num - min_num))

    backToOrigin = normalized.transpose()
    return backToOrigin


def testingSet(test_set):
    """
    This function normalized the data set and print the predction for each line
    :param test_set: the test set
    :return: print prediction
    """
    # normalize the test data set
    normal_test = minMaxAlgorithm(test_set)
    for myline in normal_test:
        printAlgorithms(myline)


class Gender(Enum):
    """
    The Enum Class for first feature
    """
    Male = 0.2
    Female = 0.4
    Infant = 0.6


"""
MAIN function
"""
# Open files
arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
file_x = open(arg1)
file_y = open(arg2)
file_test = open(arg3)
# Read content from file
DataSet_X = file_x.read()
Labels_Y = file_y.read()
testingSet_x = file_test.read()
# Close files
file_x.close()
file_y.close()
file_test.close()

# Parse dataSet X without lines, commas and after normalized each line.
DataSet_X = DataSet_X.splitlines(True)
DataSet_X = [stripAndSplit(i) for i in DataSet_X]
DataSet_X = minMaxAlgorithm(DataSet_X)

# Parse dataSet Y without lines, and commas.
Labels_Y = Labels_Y.splitlines(True)
Labels_Y = [splitterForLines(i) for i in Labels_Y]

# Parse the test set, the normalized will be in the function
testingSet_x = testingSet_x.splitlines(True)
testingSet_x = [stripAndSplit(i) for i in testingSet_x]

indexForConstansts = 0
for i in range(20):
    c = list(zip(DataSet_X, Labels_Y))
    rand.shuffle(c)
    shuffled_x, shuffled_y = zip(*c)
    for line, y in zip(shuffled_x, shuffled_y):
        y_hat_Perceptron = y_HatCalculation(wPerceptron, line)
        y_hat_SVM = y_HatCalculation(wSVM, line)
        y_hat_PA = y_HatCalculation(wPA, line)
        # Calc the const and improve our error rate for all constants and algorithms
        indexForConstansts += 1
        if indexForConstansts % 1000 == 0:
            CP = CP * 0.5
            etaSVM = etaSVM * 0.5
        if y != y_hat_Perceptron:
            # Perceptron Calc
            calcPerceptron(line, int(y), y_hat_Perceptron)
        if y != y_hat_SVM:
            # SVM Calc
            calcSVM(line, int(y), y_hat_SVM)
        else:
            # send to another calc svm for case where y=y_hat - update all
            calcSVMForAllVectors()
        # Calc norm
        norm = 2 * np.power(np.linalg.norm(line, ord=2), 2)
        if norm == 0:
            norm = 1
        taoPA = max(0, 1 - np.dot(wPA[int(y)], line) + np.dot(wPA[y_hat_PA], line)) / norm
        if y != y_hat_PA:
            calcPA(line, int(y), y_hat_PA)

testingSet(testingSet_x)


"""
 This is for self purpose and testing
"""
# magicnumber = 2000
# check_x = DataSet_X[:magicnumber]
# check_y = Labels_Y[:magicnumber]
# print("Perceptron:")
# CalculateErrorRate(check_x, check_y, wPerceptron)
# print("SVM")
# CalculateErrorRate(check_x, check_y, wSVM)
# print("PA")
# CalculateErrorRate(check_x, check_y, wPA)
