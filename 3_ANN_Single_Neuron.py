
"""-----------------------------------------------------------------------------------------------------
                         Single Neuron Calculations
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement: ReLu Activation Function 
--------------------------------------------------------------------------------------------------------"""
import random
import numpy as np
#####################################################################################################
#   Function Name   :   relu
#   Input Params    :   x
#   Output Params   :   max(0,x)
#   Description     :   ReLu Activation function
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def relu(x):
    return max(0,x)
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def main():
    inputs=np.array([random.randint(10,20),
                     random.randint(20,30),
                     random.randint(1,10)])
    weights = np.array([random.uniform(0,1),
                        random.uniform(0,1),
                        random.uniform(0,1)])
    
    bias=1.0
    
    weightedSum=np.dot(inputs,weights)+bias
    
    finalOutput=relu(weightedSum)
    
    print(f"Input   :   {inputs}")
    print(f"Weights :   {weights}")
    print(f"bias    :   {bias}")
    print(f"Weighted Sum    :   {weightedSum}")
    print(f"Final Output    :{finalOutput}")
    
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

