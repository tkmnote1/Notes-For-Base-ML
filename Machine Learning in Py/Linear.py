import numpy as np
from numpy import random
# DATA
Train= np.array([
    [0,1],
    [1,3],
    [2,5],
    [3,7],
    [4,9],
    [5,11]
],dtype=float)

# predict that y=wx+b
w=random.rand() 
b=random.rand()
learning_rate=0.01
epochs = 100
eps=0.000001
for epoch in range(epochs):
    total_lost=0.0
    for i in range(len(Train)):
        x=Train[i,0]
        y=Train[i,1]
        ##
        y_predict = w*x+b
        error = y_predict-y
        loss = error*error
        total_lost+=loss
        ## loss = (wx+b-y)^2, grad_w = dL/dw, ((w+eps)x+b-y)^2-((w+eps)x+b-y)^2/eps
        grad_w= (((w+eps)*x+b-y)**2 - loss)/eps
        grad_b = ((w*x+(b+eps)-y)**2 -loss)/eps
        ## update
        w -= learning_rate*grad_w
        b -= learning_rate*grad_b
    print(f"Epoch {epoch+1:5d} | Mất mát (Loss): {total_lost/len(Train):.6f} | w: {w:.4f}, b: {b:.4f}")
print(f"Dự đoán: y = {w:.4f} * x + {b:.4f}")
