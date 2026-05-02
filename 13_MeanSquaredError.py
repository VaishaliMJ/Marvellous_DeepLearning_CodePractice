
"""-----------------------------------------------------------------------------------------------------
                    Mean Squared Error Calculations
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement: Mean Squared Error Calculations
--------------------------------------------------------------------------------------------------------"""
import random
import numpy as np
import matplotlib.pyplot as plt
BORDER="-"*65
#####################################################################################################
#   Function Name   :   Calculate_MSE
#   Input Params    :   yTrue,yPred
#   Output Params   :   MSE
#   Description     :   MSE calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def Calculate_MSE(yTrue,yPred):
    total_loss=0
    outputLen=len(yTrue)
    for i in range(outputLen):
        error=yTrue[i]-yPred[i]
        total_loss+=error**2
    mse = total_loss / outputLen   
    return mse
#####################################################################################################
#   Function Name   :   generategenerateData
#   Input Params    :   None
#   Output Params   :   inputs,weights,bias
#   Description     :   Genrates inputs,wts and bias randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateData():
    actualOutput=np.array([random.randint(10,20),
                     random.randint(20,30),
                     random.randint(1,10)])
    
    predictedOutput=np.array([random.randint(10,20),
                     random.randint(20,30),
                     random.randint(1,10)])
    return actualOutput,predictedOutput
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def main():
    actualOutput,predictedOutput=generateData()
    print(BORDER)
    print(f"y and y^ values")
    print(BORDER)
    print(f"y   :   {actualOutput}")
    print(f"y^  :   {predictedOutput}")
    print(BORDER)
    mse=Calculate_MSE(actualOutput,predictedOutput)
    print(f"MSE :   {mse}")
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

