import sys
import numpy as np
from enum import Enum


class Gender(Enum):
    Male = 1.0
    Female = 2.0
    Infant = 3.0


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

# print(DataSet_X)
# print(Labels_Y)
# arg3=sys.argv[3]

# Split the data set
DataSet_X = DataSet_X.splitlines(True)
Labels_Y=Labels_Y.splitlines(True)

# Configure the Weight matrix
rows = 3
columns = 8
w = [[0.0] * columns] * rows
line = []


for i in range(10):
    counter = 0
    for x, y in zip(DataSet_X, Labels_Y):
        # Parse each line without commas and new lines
        temp = (DataSet_X[counter])
        temp = temp.strip()
        line = (temp.split(','))

        if (line[0] == 'M'):
            line[0] = (Gender.Male.value)
        elif (line[0] == 'F'):
            line[0] = (Gender.Female.value)
        else:
            line[0] = (Gender.Infant.value)

        line=[float(i) for i in line]
        counter += 1
        y_hat = np.argmax(np.dot(w, line))
        parsedY=float(y.strip())
        if parsedY != y_hat:
            w[int(parsedY)]=[x1 + y1 for x1, y1 in zip(w[int(parsedY)], line)]
            w[y_hat]=[x2-y2 for x2,y2 in zip(w[int(y_hat)], line)]
            #print(w[int(parsedY)])
            # w[int(y)] = w[int(y)] + line

print(w[0])
print(w[1])
print(w[2])
