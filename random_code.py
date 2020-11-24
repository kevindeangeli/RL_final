'''
Created by: Kevin De Angeli
Email: Kevindeangeli@utk.edu
Date: 11/24/20
'''
import numpy as np
import matplotlib.pyplot as plt
select_prob = [.1,.1,.1,.7]
A= [0,1,2,3]

a= np.random.choice(A, 1, p=select_prob)[0]
print(a)

class pursuit_Params():
    def __init__(self):
        self.B=.1
        self.SelectionProb = np.ones([2, 2, 4]) * .25 #This is a variable used during UCB

caca = pursuit_Params()

caca.B = 2
print(caca.B)
caca = pursuit_Params()
print(caca.B)