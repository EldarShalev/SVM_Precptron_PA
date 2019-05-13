import sys
import numpy as np
from enum import Enum
import random as rand
from scipy import stats



def CalculateErrorRate(DataSet_x, DataSet_y, weights):
    counter1 = 0
    for c, line in enumerate(DataSet_x):
        my_y_hat = y_HatCalculation(weights, line)
        if my_y_hat != (DataSet_y[c]):
            counter1 += 1
    print("The Error Rate for Perceptone is " + str(counter1 / len(DataSet_y)))


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
    temp_float=stats.mstats.zscore(temp_float,ddof=1)
    return temp_float


def y_HatCalculation(w, line):
    return np.argmax(np.dot(w, line))


def splitterForLines(line_y):
    line_y = float(line_y.strip())
    return line_y


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

# Get the dataSet Y without lines, and commas.
Labels_Y = Labels_Y.splitlines(True)
Labels_Y = [splitterForLines(i) for i in Labels_Y]
# Configure the Weight matrix
rows = 3
columns = 8
w = [[0.0] * columns] * rows

for i in range(50):
    counter_right = 0
    counter_wrong = 0
    c = list(zip(DataSet_X, Labels_Y))
    rand.shuffle(c)
    shuffled_x,shuffled_y=zip(*c)
    for line, y in zip(shuffled_x, shuffled_y):
        y_hat = y_HatCalculation(w, line)
        if y != y_hat:
            #print("Y HAR IS :  " + str(y_hat) + "   Y IS " + str(y))
            counter_wrong+=1
            w[int(y)] = [x1 + y1 for x1, y1 in zip(w[int(y)], line)]
            w[y_hat] = [x2 - y2 for x2, y2 in zip(w[int(y_hat)], line)]
        else:
            counter_right+=1
            #print("Y HAR IS :  " + str(y_hat) + "   Y IS " + str(y))

CalculateErrorRate(DataSet_X, Labels_Y, w)


print("RIGHT !!!!!!!!!!!!!!!!!" + str(counter_right))
print("Wrong !!!!!!!!!!!!!!!!!" + str(counter_wrong))


print(w[0])
print(w[1])
print(w[2])
