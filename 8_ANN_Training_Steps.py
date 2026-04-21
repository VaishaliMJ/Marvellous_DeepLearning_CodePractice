"""-----------------------------------------------------------------------------------------------------
                         ANN Training Steps
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement: ANN Training Steps
--------------------------------------------------------------------------------------------------------"""
import random,math
import numpy as np
import matplotlib.pyplot as plt

BORDER="-"*65
Y=10
LR=0.1
#####################################################################################################
#   Function Name   :   generateWeights
#   Input Params    :   None
#   Output Params   :   weights
#   Description     :   Genrates weights randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateWeights():
    w=round(random.uniform(-1,1),2)
    #w12=round(random.uniform(-1,1),2)
    return w
#####################################################################################################
#   Function Name   :   trainingPhase
#   Input Params    :   x,w
#   Output Params   :   None
#   Description     :   Training of network
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def trainingPhase(x,w):
    for epoch in range(1,11):
        print(f"\n------------ EPOCH {epoch} ------------")
        
        predicted_output = x * w
        print("\nFORWARD PASS")
        print(f"Predicted Output = {x} * {w} = {predicted_output}")
        
        error = Y - predicted_output
        print("\nERROR CALCULATION")
        print(f"Error = {Y} - {predicted_output} = {error}")
        
        loss = error ** 2
        print("\nLOSS CALCULATION")
        print(f"Loss = Error^2 = {loss}")
        
        print("\nWEIGHT UPDATE")

        print(f"Old Weight = {w}")

        # Simple update rule
        w = w + (LR * error * x)

        print(f"New Weight = Old Weight + (lr * error * input)")
        print(f"New Weight = {w}")
    return w    
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def main():
    print(BORDER)
    x=generateInput()
    w=generateWeights()
    print(BORDER)
    print(f"Input Value:{x}")
    print(f"Expected Output:{Y}")
    print(f"Weights:{w} Learning Rate:{LR}")
    
    w=trainingPhase(x,w)
    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print("\n===================================================")
    print("FINAL RESULT AFTER TRAINING")
    print("===================================================\n")

    final_output = x * w

    print("Final Weight     :", w)
    print("Final Prediction :", final_output)
    print("Expected Output  :", Y)
    
#####################################################################################################
#   Function Name   :   generateInput
#   Input Params    :   None
#   Output Params   :   inputs,weights,bias
#   Description     :   Genrates inputs,wts and bias randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################

def generateInput():
    x1=random.randint(1,5)
    return x1

####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    