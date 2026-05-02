
"""-----------------------------------------------------------------------------------------------------
                    Mean Absolute Error Calculations
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement: Mean Absolute Error Calculations
--------------------------------------------------------------------------------------------------------"""
import random
import numpy as np
import matplotlib.pyplot as plt
BORDER="-"*65
#####################################################################################################
#   Function Name   :   Calculate_MAE
#   Input Params    :   yTrue,yPred
#   Output Params   :   MAE
#   Description     :   Mean Absolute Error Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def Calculate_MAE(yTrue,yPred):
    total_loss=0
    outputLen=len(yTrue)
    for i in range(outputLen):
        error=abs(yTrue[i]-yPred[i])
        total_loss+=error
    mae = total_loss / outputLen   
    return mae
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
    mae=Calculate_MAE(actualOutput,predictedOutput)
    print(f"MAE :   {mae}")
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

