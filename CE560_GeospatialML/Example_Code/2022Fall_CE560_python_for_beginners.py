# -*- coding: utf-8 -*-
"""
Python tutorial for beginners 

@author: Jaehoon Jung, OSU
"""

# %% Spyder interface (# %% - section divider)
"""
Editor: edit your code
Console: run code by line, show print results, and interact with data or system 
Variable Explorer: show variables
Debugger: trace each step of your code
Tools -> Preferences -> Python Interpreter: change your project   
"""

# %% Basic console commands
"""
clear: clear the console 
quit: restart kernel
pip freeze: list the installs 
F9: executes selected text in the code
"""  

# %% Variable Types
#-- variables are containers for storing data values  
#-- Python variables do not need declaration. Python automatically recognizes the type of variable.  
a = 3.14
b = -34
c = True
d = 'python'

#-- prints the specified message or output to the console 
print(type(a)); print(type(b)); print(type(c)); print(type(d));
a, b = b, a
print(a, b)
 
# %% Calculation 
 
a = 6 + 12 - 3;     print('a=' + str(a))
b = 2.2 * 3.0;      print('b=' + str(b)) 
c = - - 4;          print('c=' + str(c))
d = 10/3;           print('d=' + str(d))
e = 10.0/3.0;       print('e=' + str(e))
f = (2 + 3) * 4;    print('f=' + str(f))
g = 2 + 3 * 4;      print('g=' + str(g))
h = 2 ** 3 * 4;     print('h=' + str(h))

#-- use the type() function to check variable type 
print(type(a)); print(type(b)); print(type(c)); print(type(d)); print(type(e)); 
print(type(f)); print(type(g)); print(type(h))

# %% Logical Operators 
#-- When you mix the logical operators in an expression, Python will evaluate them in the order which is called the operator precedence.
#-- not: high / and: medium / or: low 
 
a = 3 > 4;                  
b = 4 > 4;                  
c = 4 <= 4;                 
d = 2 + 2 == 4
e = 3.0 - 1.0 != 5.0 - 3.0 
f = True or False
g = False or False
h = not False 

print('a=' + str(a)); print('b=' + str(b)); print('c=' + str(c)); print('d=' + str(d)) 
print('e=' + str(e)); print('f=' + str(f)); print('g=' + str(g)); print('h=' + str(h))

# %% String
#-- text surrounded by either single or double quotation marks 
a = 'python'
print(a); 

#-- Strings are array 
#-- Square brackets can be used to access elements of the string
print(a[1])

#-- to print numbers, simply pass the value in the print()
b = 1
print(b) 

#-- to print string and numbers in the same line, use placeholder 
#-- string formatters %f allows you to specify the number of decimal places 
b = "Data ID: %d, Accuracy: %.2f %%" % (1.0, 98.777)
print(b)

i = 1
acc = 98.77
c = f"Data ID: {i}," + f" Accuracy: {acc} %" 
print(c)


# %% IF statement 
#-- logical conditions

a = 'I like Python' #-- this can be useful to parse text data 
if 'Python' in a:
    print("me too!")

#-- set expiration date of your software that does not load after the date you specified 
from datetime import datetime, date
ExpirationDate = '2022-9-30' # FIXME: check before shipping

if datetime.strptime(ExpirationDate,"%Y-%m-%d").date() < date.today(): 
    print("Your license has expired. Please renew it")
else:    
    print("Running")



# %% FOR statement 
#-- used for iterating a function 

for i in range(10): # start with 0, doesn't include 10
    print(i)
for i in range(1,10):
    print(i)
for i in range(1,10,3):
    print(i)

#-- to get a counter and the value from the list
a = [1,2,3,4,5,6,7,8,9,10] 
for idx, val in enumerate(a): 
    print(f"Data ID: {idx}, Accuracy: {val} %") 


# %% WHILE statement 
#-- run as long as a condition is true 
a = 0
while a < 10:
    print(a)
    a += 1

b = ['data_1','data_2','data_3','data_4']    
c = []
while True:
    print("b=" + str(b))
    if len(b) < 1:
        break 
    c.append(b[-1]) # add the last value
    b.pop()    # remove the last value
print("c=" + str(c))

# %% Function
# function is a block of code which only runs when it is called, can be reused
import numpy as np

def scaleData(data): # transform data to fit within a scale between 0 and 1
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalizeData(data,minV,maxV): # transform data to fit within a specific scale you specified
    return (data - minV) / (maxV - minV)

a = [1,2,3,4,5,6,7,8,9,10] 
a = np.array(a)
b = scaleData(a) # many ML and DL takes scaled values (0 - 1)
print(b)
c = normalizeData(a,0,100) # useful for comparison with other values that have different ranges
print(c)

# %% Global vs Local variables 
#-- local variables are created and accessible only inside the function in which it was created 
#-- global variables are created outside the function and accessible throughout the program and inside every function

import numpy as np

#-- global variables can be accessed by any function
def viewData():
    print(data) 

#-- but global variables can only be modified if you explicitly declare it with the 'global' keyword inside the function.
def scaleData1():
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

def scaleData2():
    global data
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def scaleData3(data): 
    return (data - np.min(data)) / (np.max(data) - np.min(data))  

data = [1,2,3,4,5,6,7,8,9,10]
data = np.array(data)
viewData()
scaleData1()
scaleData2()
data = scaleData3(data)
print(data)

# %% list 
#-- List is a data structure used to store multiple items in a single variable. 
#-- List items can be of any data type

a = []
for i in range(10):
    a.append(i)
print(a)
a.append('b'); print(a) # adds the elements to the end of a list
a.append([10,11]); print(a)
a.pop(); print(a)
a.pop(1); print(a)
a.extend([15,14,13,12,11]); print(a)  # concatenates the first list with another list
a.pop(9)
a.sort(); print(a) 
a.insert(0,0); print(a) 
print(a.count(0))
a = a[0:5]; print(a) # take the first 5 elements
a[-1] = 100; print(a) # modify the last element

#-- generate a list of consecutive numbers 
b = [1,2,3,4,5,6,7,8,9,10]; print(b)
b = [x for x in range(1,11)]; print(b)
b = [x**2 for x in range(1,11)]; print(b)
b = [x for x in range(1,11) if x % 2 == 0]; print(b) # only return the values when the division remainder is 0

# %% Tuple
#-- the values and size of a tuple cannot be chanced once they have been assigned. 
#-- makes your code safer if your data is constant

a = [1,2,3]
a[0] = 4
print(a)

b = (1,2,3)
b[0] = 4 
print(b)

def tuple_func(a,b):
    return a, b, a+b

val = tuple_func(1,2); print(val) # return a tuple containing multiple values 
a,b,c = tuple_func(1,2); print(a,b,c)
__,__,c = tuple_func(1,2); print(c) # Some Python functions return multiple values. _ you are ignoring some values you don't need.  
c = tuple_func(1,2)[2]; print(c)


# %% Numpy: vectorization
#-- used for working with arrays (vector and matrix)
import numpy as np

a = np.arange(25).reshape(5,5); print(a)
#-- vectorized operation is faster than loop operation
a[:,0] = 0; print(a)
a[1,:] = 100; print(a)
a[a>10] = -10; print(a)

a = np.arange(25).reshape(5,5); print(a)
a[[0, 1],:] = a[[1, 0],:]; print('a=' + str(a)) # switch rows
a[:,[0, 1]] = a[:,[1, 0]]; print('a=' + str(a)) # switch columns 

# %% Numpy: matrix calculation 

# A = np.random.rand(3,3); print(A)
# y = np.random.rand(3,1); print(y)
A = np.random.randint(10, size=(3,3)); print(A)
y = np.random.randint(10, size=(3,1)); print(y)

#-- least squares formular using array
AT = np.transpose(A); print(AT)
ATA = np.matmul(AT,A); print(ATA)
invATA = np.linalg.inv(ATA); print(invATA)
ATy = np.matmul(AT,y); print(ATy)
invATA_ATy = np.matmul(invATA,ATy); print(invATA_ATy)

#-- least squares formular using matrix 
A = np.asmatrix(A); type(A)
y = np.asmatrix(y); type(y)
invATA_ATy = np.linalg.inv(A.T*A)*A.T*y; print(invATA_ATy)

# %% Shallow Copy / Deep Copy in the numpy array

import numpy as np 

a = np.arange(25).reshape(5,5); print(a)
b = a # shallow copy: copy only the reference without allocating new memory 
b[0,:] = 100; print(b); print(a)
b = a.copy() # deep copy: allow new memory. completely independent of the original variable
b[0,:] = -10; print(b); print(a)


# %% Shallow Copy / Deep Copy / Deeper Copy in the list 
import copy

a = [1,2,3]
b = a
a[0] = 10; print(a); print(b)

a = [1,2,3]
b = a.copy()
a[0] = 10; print(a); print(b)

a = [1,2,3,[4,5]] # appended list 
b = a
a[3][0] = 10; print(a); print(b)

a = [1,2,3,[4,5]]
b = a.copy()
a[0] = 10; print(a); print(b)
a[3][0] = -7; print(a); print(b)

a = [1,2,3,[4,5]]
b = copy.deepcopy(a)
a[3][0] = -7; print(a); print(b)

# %% Visualization

#-- matplotlib is a comprehensive library for visualizing data in Python.
import matplotlib.pyplot as plt
import numpy as np
  
x = ['a','b','c','d','e']
y1 = [v for v in range(1,6)]; 
y2 = [v**2 for v in range(1,6)]; 
y3 = [v/2 for v in range(1,6)]; 
y4 = [v+2 for v in range(1,6)];

#-- single plot
plt.plot(x, y1, label='recall', c='b', ls='-.', marker='+', ms=10)
plt.plot(x, y2, label='precision', c='r', ls='-', marker='o', ms=5)
plt.ylim(0,30)
plt.xlabel('data')
plt.ylabel('accuracy')
plt.title('accuracy assessment')
plt.legend(loc='upper left')
plt.show()

#-- multiple subplots # adds subplot to a current figure at the specified grid position 
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.set_title("recall")
ax2.set_title("precision")
ax3.set_title("F1-score")
ax4.set_title("IoU")
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)
plt.show()

#-- plot image
img = np.random.randint(100, size=(100,100))
plt.figure()
plt.imshow(img, cmap='jet')
plt.colorbar()
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.grid(False)
plt.show()

#-- save image
img = np.random.randint(100, size=(100,100))
fig = plt.figure()
plt.imshow(img, cmap='jet')
plt.colorbar()
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.grid(False)
plt.savefig('test.png', dpi=300)
plt.close(fig)


































