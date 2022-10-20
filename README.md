# Experiment-3-Implementation-of-MLP-for-non-linear-separable-problem
## <img width="104" alt="NN2" src="https://user-images.githubusercontent.com/115135775/196995240-49585f61-cc9a-4938-9959-49b53a36cbaf.png">
AIM:

To implement a perceptron for classification using Python

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows
XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

## Algorithm :

Step 1 : Initialize the input patterns for XOR Gate

Step 2: Initialize the desired output of the XOR Gate

Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron and 1 output neuron

Step 4: Repeat the  iteration  until the losses become constant and minimum

             ``` (i) Compute the output using forward pass output
	      
             (ii) Compute the error  
	      
	     (iii) Compute the change in weight ‘dw’ by using backward propagation algorithm.
	      
             (iv) Modify the weight as per delta rule.
	     
             (v)   Append the losses in a list```
	     
Step 5: Test for the XOR patterns.

## PROGRAM :

Developed by: PAVITHRA P

Reg.No: 212221220037


```
Program to implement a perceptron for Implementation of MLP for non linearly separable problem using Python programming.

import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

x = np.array([[0,0,1,1],[0,1,0,1]])
y = np.array([[0,1,1,0]])

n_x = 2
n_y = 1
n_h = 2

m =x.shape[1]

lr = 0.1

np.random.seed(2)

w1 = np.random.rand(n_h,n_x) # weight matrix for hidden layer
w2 = np.random.rand(n_y,n_h) # weight matrix for output layer

losses = []

def sigmoid(z):
    z = 1/(1+np.exp(-z))
    return z

def forward_prop(w1,w2,x):
    z1 = np.dot(w1,x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2,a1)
    a2 = sigmoid(z2)
    return z1,a1,z2,a2

def back_prop(m,w1,w2,z1,a1,z2,a2,y):
    dz2 = a2-y
    dw2 = np.dot(dz2,a1.T)/m
    dz1 = np.dot(w2.T,dz2) * a1*(1-a1)
    dw1 = np.dot(dz1,x.T)/m
    dw1 = np.reshape(dw1,w1.shape)
    dw2 = np.reshape(dw2,w2.shape)
    return dz2,dw2,dz1,dw1

iterations = 10000
for i in range(iterations):
    z1,a1,z2,a2 = forward_prop(w1,w2,x)
    loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    losses.append(loss)
    da2,dw2,dz1,dw1 = back_prop(m,w1,w2,z1,a1,z2,a2,y)
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1

# we plot losses to see how our network is doing...
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("LOSS VALUE")

def predict(w1,w2,input):
    z1,a1,z2,a2 = forward_prop(w1,w2,test)
    a2 = np.squeeze(a2)
    if (a2>=0.5):
        print( [i[0] for i in input], 1)
    else:
        print( [i[0] for i in input], 0)

print('Input',' Output')
test=np.array([[0],[0]])
predict(w1,w2,test)
test=np.array([[0],[1]])
predict(w1,w2,test)
test=np.array([[1],[0]])
predict(w1,w2,test)
test=np.array([[1],[1]])
predict(w1,w2,test)
```

 ## OUTPUT :
 
<img width="293" alt="NN" src="https://user-images.githubusercontent.com/115135775/196995437-0585fe22-f026-462d-8574-4d8ffeef48bf.png">

<img width="104" alt="NN2" src="https://user-images.githubusercontent.com/115135775/196995485-71880d68-4d19-4f70-8bf6-0c9e0c76df47.png">

   
## RESULT :

Thus, a program involving python to implement a perceptron for Implementation of MLP for non linearly separable problem is developed and executted successfully.

