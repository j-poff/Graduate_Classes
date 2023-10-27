# -*- coding: utf-8 -*-
"""
Forward and backward propagation in Neural Network 

@author: Jaehoon Jung, PhD, OSU

last updated: Oct 10, 2022
"""

from scipy.special import expit

def gradientDescent(x,dx,n):
    return x - n*dx

X1 = 2  # input data1
X2 = 1  # input data2 
T = 1   # observation 
n = 0.5 # learning rate

w1 = 0.1
w2 = 0.2
w3 = 0.3
w4 = 0.4
w5 = 0.5
w6 = 0.6

for i in range(100):
    # %% feedforward
    H1 = X1*w1 + X2*w3

    sH1 = expit(H1) # sigmoid function
    
    H2 = X1*w2 + X2*w4
    
    sH2 = expit(H2)
    
    Y = sH1*w5 + sH2*w6
    sY = expit(Y)
    
    R = 0.5*(T-sY)**2
    
    print('Iteration: %d / Squared Residual: %.4f' % (i, R))
    
    # %% backpropagation
    dR_dsY = -(T-sY)
    dsY_dY = sY*(1-sY)
    
    dY_dw5 = sH1
    dR_dw5 = dR_dsY*dsY_dY*dY_dw5
    w5 = gradientDescent(w5,dR_dw5,n)

    dY_dw6 = sH2
    dR_dw6 = dR_dsY*dsY_dY*dY_dw6
    w6 = gradientDescent(w6,dR_dw6,n)
    
    dR_dw1 = (dR_dsY*dsY_dY*w5)*(sH1*(1-sH1))*X1
    w1 = gradientDescent(w1,dR_dw1,n)
    
    dR_dw2 = (dR_dsY*dsY_dY*w6)*(sH2*(1-sH2))*X1
    w2 = gradientDescent(w2,dR_dw2,n)
    
    dR_dw3 = (dR_dsY*dsY_dY*w5)*(sH1*(1-sH1))*X2
    w3 = gradientDescent(w3,dR_dw3,n)
    
    dR_dw4 = (dR_dsY*dsY_dY*w6)*(sH2*(1-sH2))*X2
    w4 = gradientDescent(w4,dR_dw4,n)
    
    
    
    
    
    
    
    
