
"""-----------------------------------------------------------------------------------------------------
                        Binary Cross Entropy
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Binary Cross Entropy
--------------------------------------------------------------------------------------------------------"""
import random,math
import numpy as np
import matplotlib.pyplot as plt
BORDER="-"*65
#####################################################################################################
#   Function Name   :   Calculate_BinaryCrossEntropy
#   Input Params    :   yTrue,yPred
#   Output Params   :   Binary Cross Entropy
#   Description     :   Binary Cross Entropy calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def Calculate_BinaryCrossEntropy(yTrue,yPred):
    outputLen=len(yTrue)
    total_loss =0
    for i in range(outputLen):
        y = yTrue[i]
        p = yPred[i] 
        #Avoid 0
        p = max(min(p, 0.999), 0.001)
        loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        total_loss += loss 
    bceLoss= total_loss/outputLen     
    return round(bceLoss,2)
#####################################################################################################
#   Function Name   :   generategenerateData
#   Input Params    :   None
#   Output Params   :   inputs,weights,bias
#   Description     :   Genrates inputs,wts and bias randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateData():
    actualOutput=np.array([random.randint(0,1),
                     random.randint(0,1),
                     random.randint(0,1)])
    
    
    predictedOutput=np.array([round(random.uniform(0,1),2),
                     round(random.randint(0,1),2),
                     round(random.randint(0,1),2)])
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
    loss=Calculate_BinaryCrossEntropy(actualOutput,predictedOutput)
    print(f"Loss Binary Cross Entropy :   {loss}")
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

