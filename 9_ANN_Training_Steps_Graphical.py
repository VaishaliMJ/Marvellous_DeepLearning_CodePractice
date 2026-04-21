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
    steps = []
    loss_list = []
    weight_list = []
    prediction_list = []
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

        steps.append(epoch)
        loss_list.append(loss)
        weight_list.append(w)
        prediction_list.append(predicted_output)
        
        # Simple update rule
        w = w + (LR * error * x)

        print(f"New Weight = Old Weight + (lr * error * input)")
        print(f"New Weight = {w}")
        
    return steps,loss_list,weight_list,prediction_list,w   
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
    
    steps,loss,weight_list,prediction_list,w=trainingPhase(x,w)
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
    
    plot_graphs(steps,loss,weight_list,prediction_list,w)
#####################################################################################################
#   Function Name   :   plot_graphs
#   Input Params    :   steps,loss,weight_list,prediction_list,w
#   Output Params   :   None
#   Description     :   Plots different graphs
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def plot_graphs(steps,loss,weight_list,prediction_list,w):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,8)) 
    axes[0][0].plot(steps, loss, marker='o')
    axes[0][0].set_title("Loss Decreasing During Training")
    axes[0][0].set_xlabel("Steps")
    axes[0][0].set_ylabel("Loss")
    axes[0][0].grid()
    
    actual_line = [Y] * len(steps)

    axes[0][1].plot(steps, prediction_list, marker='o', label="Predicted Output")
    axes[0][1].plot(steps, actual_line, linestyle='--', label="Actual Output")
    axes[0][1].set_title("Prediction Approaching Actual Output")
    axes[0][1].set_xlabel("Steps")
    axes[0][1].set_ylabel("Value")
    axes[0][1].legend()
    axes[0][1].grid()
    
    
    axes[1][0].plot(steps, weight_list, marker='o')
    axes[1][0].set_title("Weight Adjustment During Learning")
    axes[1][0].set_xlabel("Steps")
    axes[1][0].set_ylabel("Weight Value")
    axes[1][0].grid()
    
    
    
    plt.show()
#####################################################################################################
#   Function Name   :   generateInput
#   Input Params    :   None
#   Output Params   :   inputs,weights,bias
#   Description     :   Genrates inputs
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################

def generateInput():
    x1=random.randint(1,5)
    return x1

####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    