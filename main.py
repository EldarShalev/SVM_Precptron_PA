import sys
import numpy as np
from enum import Enum
import random as rand
from scipy import stats

# TODO parse the 3'rd argument and send to print function
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
    counter1 = 0
    for c, line in enumerate(DataSet_x):
        my_y_hat = y_HatCalculation(weights, line)
        if my_y_hat != (DataSet_y[c]):
            counter1 += 1
    print("The Error Rate  is " + str(counter1 / len(DataSet_y)))


def stripAndSplit(line):
    line = line.strip()
    line = line.split(',')

    if line[0] == 'M':
        line[0] = Gender.Male.value
    elif line[0] == 'F':
        line[0] = Gender.Female.value
    else:
        line[0] = Gender.Infant.value
    temp_float = [float(i) for i in line]
    #temp_float = stats.mstats.zscore(temp_float, ddof=1)
    return temp_float


def y_HatCalculation(w, line):
    return np.argmax(np.dot(w, line))


def splitterForLines(line_y):
    line_y = float(line_y.strip())
    return line_y


def calcPerceptron(line_per, original_y, y_hat):
    line_per = np.multiply(line_per, CP)
    wPerceptron[int(original_y)] = [x1 + y1 for x1, y1 in zip(wPerceptron[int(original_y)], line_per)]
    wPerceptron[y_hat] = [x2 - y2 for x2, y2 in zip(wPerceptron[int(y_hat)], line_per)]


def calcSVM(line_SVM, original_y, y_hat):
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
    oneMinus = 1 - etaSVM * lambdaSVM
    for i in range(rows):
        wSVM[i] = np.multiply(oneMinus, wSVM[i])


def calcPA(linePA, y, y_hat_pa):
    linePA = np.multiply(linePA, taoPA)
    wPA[y] = np.add(wPA[y], linePA)
    wPA[y_hat_pa] = np.subtract(wPA[y_hat_pa], linePA)


# TODO make sure we get normalized data
def printAlgorithms(test_x):
    Perceptron_y_hat = y_HatCalculation(wPerceptron, test_x)
    SVM_y_hat = y_HatCalculation(wSVM, test_x)
    PA = y_HatCalculation(wPA, test_x)
    print("perceptron: " + str(Perceptron_y_hat) + ", svm: " + str(SVM_y_hat) + ", pa: " + str(PA))


def minMaxAlgorithm(dataSet):
    normalized = np.array([np.array(xi) for xi in dataSet])
    normalized = normalized.transpose()
    for i, n in enumerate(normalized):
        min_num = min(n)
        max_num = max(n)
        if max_num - min_num != 0:
            normalized[i] = np.divide(np.subtract(n, min_num), (max_num - min_num))

    backToOrigin = normalized.transpose()
    return backToOrigin


class Gender(Enum):
    Male = 0.2
    Female = 0.4
    Infant = 0.6


# Open files
arg1 = sys.argv[1]
arg2 = sys.argv[2]
file_x = open(arg1)
file_y = open(arg2)
# Read content from file
DataSet_X = file_x.read()
Labels_Y = file_y.read()
# Close files
file_x.close()
file_y.close()

# arg3=sys.argv[3]

# Get the dataSet X without lines, commas and after normalized each line.
DataSet_X = DataSet_X.splitlines(True)
DataSet_X = [stripAndSplit(i) for i in DataSet_X]
DataSet_X = minMaxAlgorithm(DataSet_X)

# Test on 5/6 from the data set
magicNumber = 2700
train_x = DataSet_X[:magicNumber]

# Get the dataSet Y without lines, and commas.
Labels_Y = Labels_Y.splitlines(True)
Labels_Y = [splitterForLines(i) for i in Labels_Y]
train_y = Labels_Y[:magicNumber]

# Configure the Weight matrix

counter123 = 0
for i in range(20):
    c = list(zip(train_x,train_y))
    rand.shuffle(c)
    shuffled_x, shuffled_y = zip(*c)
    for line, y in zip(shuffled_x, shuffled_y):
        y_hat_Perceptron = y_HatCalculation(wPerceptron, line)
        y_hat_SVM = y_HatCalculation(wSVM, line)
        y_hat_PA = y_HatCalculation(wPA, line)
        # Calc the const and improve our error rate for all constants and algorithms
        counter123+=1
        if counter123%1000 == 0:
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

# magicNumber = 2700
testing_x = DataSet_X[magicNumber:]
testing_y = Labels_Y[magicNumber:]
print("Perceptron:")
CalculateErrorRate(testing_x, testing_y, wPerceptron)
print("SVM")
CalculateErrorRate(testing_x, testing_y, wSVM)
print("PA")
CalculateErrorRate(testing_x, testing_y, wPA)

# for myline in testing_x:
#     printAlgorithms(myline)
