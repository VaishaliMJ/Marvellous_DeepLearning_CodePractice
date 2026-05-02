"""-----------------------------------------------------------------------------------------------------
                         Layered ANN Simplified
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement: Layered ANN Simplified
--------------------------------------------------------------------------------------------------------"""
import random,math
import numpy as np
import matplotlib.pyplot as plt

BORDER="-"*65

#####################################################################################################
#   Function Name   :   generateInput
#   Input Params    :   None
#   Output Params   :   inputs,weights,bias
#   Description     :   Genrates inputs,wts and bias randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################

def generateInput():
    x1=random.randint(1,5)
    x2=random.randint(1,5)
                     
    return x1,x2
#####################################################################################################
#   Function Name   :   generateWeights
#   Input Params    :   input_size, neuron_count
#   Output Params   :   weights
#   Description     :   Genrates weights randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateWeights():
    w11=round(random.uniform(-1,1),2)
    w12=round(random.uniform(-1,1),2)
    return w11,w12
#####################################################################################################
#   Function Name   :   generateBias
#   Input Params    :   None
#   Output Params   :   weights
#   Description     :   Genrates bias randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateBias():
    bias=round(random.uniform(-1, 1),2)
    return bias
####################################################################################################
#   Function Name   :   calculateWeightedSum
#   Input Params    :   x1,x2,w1,w2,b
#   Output Params   :   Weighted Sum
#   Description     :   Hidden Layers Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################    
def calculateWeightedSum(x1,x2,w1,w2,b):
    z=x1*w1 + x2*w2 + b
    return round(z,3)
#####################################################################################################
#   Function Name   :   relu
#   Input Params    :   z
#   Output Params   :   max(0,z)
#   Description     :   ReLu Activation function
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def relu(z):
    return max(0,z)
#####################################################################################################
#   Function Name   :   sigmoid
#   Input Params    :   z
#   Output Params   :   max(0,z)
#   Description     :   sigmoid Activation function
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def sigmoid(z):
    return 1 / (1 + math.exp(-z))
####################################################################################################
#   Function Name   :   displayFinalNNSummary
#   Input Params    :   hidden_outputs,final_output
#   Output Params   :   None
#   Description     :   Hidden Layers Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def displayFinalNNSummary(hidden_output1,hidden_output2,final_output):
    print(BORDER)
    print("\n================ FINAL SUMMARY ================\n")
    print(f"Hidden Layer Output h1 : {hidden_output1}")
    print(f"Hidden Layer Output h2 : {hidden_output2}")

    print(f"Final Network Output : {final_output:.3f}")
    print(f"Confidence Percentage: {final_output * 100:.2f}%")
    print(BORDER)

    if final_output >= 0.5:
        print("Prediction           : Positive Class")
    else:
        print("Prediction           : Negative Class")
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def main():
    print(BORDER)
    print("STEP 1 : INPUT LAYER")
    print(BORDER)
    x1,x2=generateInput()
    print("Input Features (x):")
    print(f"  x1 = {x1}")
    print(f"  x2 = {x2}")   
    print(BORDER)
    
    print("STEP 2 : HIDDEN LAYER (2 NEURONS)")

    print("HIDDEN NEURON 1")
    print("----------------")   
    print("Hidden layer Neurons")
    print(BORDER)
    
    w11,w12=generateWeights()
    print("Weights:")
    print(f"  w11 = {w11}, w12 = {w12}")
    b1 =generateBias()
    print("\n\nBias:")
    print(f"  b1 = {b1}")
    print(BORDER)

    z1=calculateWeightedSum(x1,x2,w11,w12,b1)
    print(f"Weighted Sum Calculations:\n(x1*w11 + x2*w12 + b1) : {x1}*{w11} + {x2}*{w12} + {b1}: {z1}")
    
    h1_output=relu(z1)
    print("\nApplied ReLU Activation function")
    print(f"  ReLU(z1) = max(0, {z1}) = {h1_output}")
    print("\nOutput of Hidden Neuron 1 (h1):", h1_output)
    
    #   Hidden Layer 2

    print(BORDER)
    print("HIDDEN NEURON 2")
    print(BORDER)
    w21,w22=generateWeights()
    print("Weights:")
    print(f"  w21 = {w21}, w22 = {w22}")
    b2 =generateBias()
    print("\n\nBias:")
    print(f"  b2 = {b2}")
    print(BORDER)
    
    z2=calculateWeightedSum(x1,x2,w21,w22,b2)
    print(f"Weighted Sum Calculations:\n(x1*w21 + x2*w22 + b2) : {x1}*{w21} + {x2}*{w22} + {b2}: {z2}")
    h2_output=relu(z2)
    print("\nApplied ReLU Activation function")
    print(f"  ReLU(z2) = max(0, {z2}) = {h2_output}")
    print("\nOutput of Hidden Neuron 2 (h2):", h2_output)
    
    # ---------------------------------------------------------
    # STEP 3 : OUTPUT LAYER
    # ---------------------------------------------------------

    print(BORDER)
    print("STEP 3 : OUTPUT LAYER (FINAL PREDICTION)")
    print(BORDER)
    w_out1,w_out2=generateWeights()
    b_out=generateBias()
    print("Weights:")
    print(f"  w_out1 = {w_out1}, w_out2 = {w_out2}")
    print("Bias:")
    print(f"  b_out = {b_out}")

    print("\nFormula:")
    print("  z_out = (h1_outputo*w_out1 + h2_output*w_out2) + b_out")
    z_out=calculateWeightedSum(h1_output,h2_output,w_out1,w_out2,b_out)
    
    final_output=sigmoid(z_out)
    print(f"  Sigmoid(z_out) = 1 / (1 + e^(-{z_out})) = {final_output}")
    displayFinalNNSummary(h1_output,h2_output,final_output)
    
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

