# GİZEM BOĞAN
#280201071


import numpy as np
import random
from matplotlib import pyplot as plt

# Experiment 1

ar_A = []
ar_B = []
ar_C = []
ar_X = []

av_A = []
av_B = []
av_C = []
av_X = []
vr_X = []

# Populate the given arrays.   
iteration_num = 30000

for i in range(iteration_num):

    A = int((random.random() * 6) + 1) # Random variable A represents the outcome of a fair 6-face dice
    ar_A.append(A) 

    B = int((random.random() * 4) + 1) # Random variable B represents the outcome of a fair 4-face dice
    ar_B.append(B)  

    C_temp = random.random()    # Random variable C represents the outcome of a fair coin (-1 => tails, +1=> heads)
    if C_temp < 0.5:
        C = -1
    else:
        C = 1
    ar_C.append(C)  

    X = A + (B * C)  #Random variable X is a calculated variable from A, B and C random variables
    ar_X.append(X)  
    
    e_value_a = 0
    e_value_b = 0
    e_value_c = 0
    e_value_x = 0
    e_value_x2 = 0

    for val in range(len(ar_A)):
        e_value_a += (ar_A[val] / len(ar_A))  #computing the expected values
        e_value_b += (ar_B[val] / len(ar_B))
        e_value_c += (ar_C[val] / len(ar_C))

        e_value_x += (ar_X[val] / len(ar_X))
        e_value_x2 += ((ar_X[val])**2 / len(ar_X))  #it is calculated for computing the variance
    
    av_A.append(e_value_a)
    av_B.append(e_value_b)
    av_C.append(e_value_c)

    av_X.append(e_value_x)
    vr_X.append(e_value_x2 - (e_value_x**2))



# Inspect the following plots.
plt.figure() 
plt.hist(ar_A,6,range=(1,7),align='left',density=True, rwidth=0.8)
plt.figure()
plt.hist(ar_B,4,range=(1,5),align='left',density=True, rwidth=0.8)
plt.figure()
plt.hist(ar_C,3,range=(-1,2),align='left',density=True, rwidth=0.8)
plt.figure()
plt.hist(ar_X,14,range=(-3,11),align='left',density=True, rwidth=0.8)


# Plot the average and variance values.
plt.figure()    #Average values of A
plt.hist(av_A, 6,range=(1,7), align='left',density=True, rwidth=0.8)

plt.figure()    #Average values of B
plt.hist(av_B, 4,range=(1,5), align='left',density=True, rwidth=0.8)

plt.figure()    #Average values of C
plt.hist(av_C, 3,range=(-1,2), align='left',density=True, rwidth=0.8)

plt.figure()    #Average values of X
plt.hist(av_X, 14,range=(-3,11), align='left',density=True, rwidth=0.8)

plt.figure()    #Variance of X
plt.hist(vr_X, 14,range=(0,14), align='left',density=True, rwidth=0.8)



# Experiment 2

# Part a (Inverse Transform Method)
U = []
Xa = []
av_Xa = []
vr_Xa = []

# Populate the given arrays.
# F(x) = x**2

def inverse_F(u):
    return u**(0.5)


for i in range (iteration_num):
    u = random.random()
    U.append(u)

    x = inverse_F(u) #inverse transformation method
    Xa.append(x)

    e_val_xa = 0
    e_val_xa2 = 0
    
    for val in range(len(Xa)):
        e_val_xa += (Xa[val] / len(Xa))
        e_val_xa2 += ((Xa[val])**2 / len(Xa)) #calculated for computing the variance
        
    av_Xa.append(e_val_xa)
    vr_Xa.append(e_val_xa2 - (e_val_xa)**2)


# Inspect the following plots.
plt.figure()
for i in range(len(Xa)):
    plt.plot([Xa[i],U[i]],[1,1.2])
plt.figure()
hU = plt.hist(U,100,alpha=0.5,density=True)
hXa = plt.hist(Xa,100,alpha=0.5,density=True)
plt.figure()
plt.plot(np.cumsum(hU[0]))
plt.plot(np.cumsum(hXa[0]))

# Plot the average and variance values.

plt.figure()    #Average values of Xa
plt.hist(av_Xa, 10,range=(0,1), align='left',density=True, rwidth=0.8)

plt.figure()    #Variance of Xa
plt.hist(vr_Xa, 10,range=(0,0.2), align='left',density=True, rwidth=0.8)


# Part b (Rejection Method)

Xb = []
av_Xb = []
vr_Xb = []

# Populate the given arrays.

# f(x) = 2*x -> This is the pdf of given cdf F(x) = x^2


for i in range (iteration_num):
    x = random.random()
    y = random.random()


    ex_val_b = 0
    ex_val_2_b =0
    
    if y > 2*x:  
        pass        #reject the point
    else:
        Xb.append(x)
        
        for j in range (len(Xb)):
            ex_val_b += (x / len(Xb))
            ex_val_2_b += (x**2 / len(Xb))
            

        av_Xb.append(ex_val_b)
        vr_Xb.append(ex_val_2_b - (ex_val_b)**2)


# Inspect the following plots.
plt.figure()
hXb = plt.hist(Xb,100,density=True)
plt.figure()
plt.plot(np.cumsum(hXb[0]))

# Plot the average and variance values.

plt.figure()    #Average values of Xb
plt.hist(av_Xb, 10,range=(0,1), align='left',density=True, rwidth=0.8)

plt.figure()    #Variance of Xb
plt.hist(vr_Xb, 8,range=(0,0.5), align='left',density=True, rwidth=0.8)

plt.show()
