"""-----------------------------------------------------------------------------------------------------
                         ANN Multi Neuron Training Steps
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:ANN Multi Neuron Training Steps
--------------------------------------------------------------------------------------------------------"""
import random,math
import numpy as np
import matplotlib.pyplot as plt

BORDER="-"*65
ACTUAL_OUTPUT=10
LR=0.01
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
#   Function Name   :   generateWeights
#   Input Params    :   None
#   Output Params   :   weights
#   Description     :   Genrates weights randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateWeights():
    w1=round(random.uniform(-1,1),2)
    w2=round(random.uniform(-1,1),2)
    return w1,w2
#####################################################################################################
#   Function Name   :   generateBias
#   Input Params    :   None
#   Output Params   :   bias
#   Description     :   Genrates bias randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateBias():
    bias=round(random.uniform(-1, 1),2)
    return bias
#####################################################################################################
#   Function Name   :   trainingPhase
#   Input Params    :   inputsWtsBias
#   Output Params   :   None
#   Description     :   Training of network
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def trainingPhase(inputsWtsBias):
    #[x1,x2,w11,w12,b1,w21,w22,b2,w_out1,w_out2,b_out]
    steps = []
    loss_list = []
    prediction_list = []
    actual_list = []

    h1_list = []
    h2_list = []

    w11_list = []
    w12_list = []
    w21_list = []
    w22_list = []
    wout1_list = []
    wout2_list = [] 
    x1=inputsWtsBias[0] 
    x2=inputsWtsBias[1]
    w11=inputsWtsBias[2] 
    w12=inputsWtsBias[3]
    b1=inputsWtsBias[4]
    w21=inputsWtsBias[5]
    w22=inputsWtsBias[6]
    b2=inputsWtsBias[7]
    w_out1=inputsWtsBias[8]
    w_out2=inputsWtsBias[9]
    b_out=inputsWtsBias[10]
    for epoch in range(1,21):
        print(f"\n------------ EPOCH {epoch} ------------")
        z1=calculateWeightedSum(x1,x2,w11,w12,b1)
        h1=relu(z1)
        
        print("\nHidden Neuron 1")
        print(f"z1 = ({x1} * {w11}) + ({x2} * {w12}) + {b1} = {z1}")
        print(f"h1 = ReLU(z1) = {h1}")
        
        z2=calculateWeightedSum(x1,x2,w21,w22,b2)
        h2=relu(z2)
        
        print("\nHidden Neuron 2")
        print(f"z2 = ({x1} * {w21}) + ({x2} * {w22}) + {b2} = {z2}")
        print(f"h2 = ReLU(z2) = {h2}")
        
        predicted_output = calculateWeightedSum(h1,h2,w_out1,w_out2,b_out)

        print("\nOutput Neuron")
        print(f"Predicted Output = ({h1} * {w_out1}) + ({h2} * {w_out2}) + {b_out}")
        print(f"Predicted Output = {predicted_output}")
        
        error = ACTUAL_OUTPUT - predicted_output
        loss = error ** 2
    
        print("\nError and Loss")
        print(f"Error = {ACTUAL_OUTPUT} - {predicted_output} = {error:.4f}")
        print(f"Loss  = Error^2 = {loss:.4f}")
        
        steps.append(epoch)
        loss_list.append(loss)
        prediction_list.append(predicted_output)
        actual_list.append(ACTUAL_OUTPUT)

        h1_list.append(h1)
        h2_list.append(h2)

        w11_list.append(w11)
        w12_list.append(w12)
        w21_list.append(w21)
        w22_list.append(w22)
        wout1_list.append(w_out1)
        wout2_list.append(w_out2)
        
        
        # Output layer weights update
        w_out1 = w_out1 + (LR * error * h1)
        w_out2 = w_out2 + (LR * error * h2)
        b_out = b_out + (LR * error)

        # Hidden layer rough update
        if z1 > 0:
             w11 = w11 + (LR * error * x1 * w_out1 * 0.1)
             w12 = w12 + (LR * error * x2 * w_out1 * 0.1)
             b1 = b1 + (LR * error * w_out1 * 0.1)

        if z2 > 0:
             w21 = w21 + (LR * error * x1 * w_out2 * 0.1)
             w22 = w22 + (LR * error * x2 * w_out2 * 0.1)
             b2 = b2 + (LR * error * w_out2 * 0.1)
        print("\nUpdated Parameters")
        print(f"w11 = {w11:.4f}, w12 = {w12:.4f}, b1 = {b1:.4f}")
        print(f"w21 = {w21:.4f}, w22 = {w22:.4f}, b2 = {b2:.4f}")
        print(f"w_out1 = {w_out1:.4f}, w_out2 = {w_out2:.4f}, b_out = {b_out:.4f}")

    
    final_z1 = calculateWeightedSum(x1,x2, w11, w12, b1)
    final_h1 = relu(final_z1)

    final_z2 = calculateWeightedSum(x1,x2 ,w21, w22, b2)
    final_h2 = relu(final_z2)

    final_output = calculateWeightedSum(final_h1, final_h2,w_out1, w_out2, b_out)

    print(BORDER)
    print("FINAL RESULT AFTER TRAINING")
    print(BORDER)
    print(f"Final Hidden Output h1 = {final_h1}")
    print(f"Final Hidden Output h2 = {final_h2}")
    print(f"Final Prediction       = {final_output}")
    print(f"Actual Output          = {ACTUAL_OUTPUT}")
    print(f"Final Error            = {ACTUAL_OUTPUT - final_output}")

    plot_graphs([steps,loss_list, 
                 prediction_list,actual_list, 
                h1_list,h2_list, 
                w11_list,w12_list, 
                w21_list,w22_list,
                wout1_list,wout2_list])
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def main():
    print(BORDER)
    x1,x2=generateInput()
    
    print(f"Input Value:{x1}    {x2}")
    print(f"Actual Output:{ACTUAL_OUTPUT}")
    print(f"Learning Rate:{LR}")
    
    process_Neurons(x1,x2)
    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    
#####################################################################################################
#   Function Name   :   process_Neurons
#   Input Params    :   Process Multi Neurons
#   Output Params   :   None
#   Description     :   Process Neurons Step by Step
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def process_Neurons(x1,x2):
    #   Generate weights  and Bias for Hidden Neuron 1
    w11,w12=generateWeights()
    b1=generateBias()
    
    w21,w22=generateWeights()
    b2=generateBias()
    
    w_out1,w_out2=generateWeights()
    b_out=generateBias()
    
    print(f"Inital Neural Network Parameters:")
    print(BORDER)    
    print(f"w11:{w11}   w12:{w12}   b1:{b1}")
    print(f"w21:{w21}   w22:{w22}   b2:{b2}")
    print(f"w_out1:{w_out1}   w_out2:{w_out2}   b_out:{b_out}")
    print(f"Learning Rate:{LR}")
    
    
    trainingPhase([x1,x2,w11,w12,b1,w21,w22,b2,w_out1,w_out2,b_out])
#####################################################################################################
#   Function Name   :   plot_graphs
#   Input Params    :   steps,loss,weight_list,prediction_list,w
#   Output Params   :   None
#   Description     :   Plots different graphs
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def plot_graphs(parmsList):
    steps=parmsList[0]
    loss_list=parmsList[1] 
    prediction_list=parmsList[2]
    actual_list=parmsList[3]
    h1_list=parmsList[4]
    h2_list=parmsList[5]
    w11_list=parmsList[6]
    w12_list=parmsList[7] 
    w21_list=parmsList[8]
    w22_list=parmsList[9]
    wout1_list=parmsList[10]
    wout2_list=parmsList[11]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,8)) 
    axes[0][0].plot(steps, loss_list, marker='o')
    axes[0][0].set_title("Loss Decreasing During Training")
    axes[0][0].set_xlabel("Steps")
    axes[0][0].set_ylabel("Loss")
    axes[0][0].grid()
    
    #actual_line = [ACTUAL_OUTPUT] * len(steps)

    axes[0][1].plot(steps, prediction_list, marker='o', label="Predicted Output")
    axes[0][1].plot(steps, actual_list, linestyle='--', label="Actual Output")
    axes[0][1].set_title("Prediction Approaching Actual Output")
    axes[0][1].set_xlabel("Steps")
    axes[0][1].set_ylabel("Value")
    axes[0][1].legend()
    axes[0][1].grid()
    
    
    axes[1][0].plot(steps, h1_list, marker='o',label="Hidden Neuron h1")
    axes[1][0].plot(steps, h2_list, marker='s',label="Hidden Neuron h1")
    axes[1][0].set_title("Hidden Neuron Activations During Training")
    axes[1][0].set_xlabel("Training Step")
    axes[1][0].set_ylabel("Activation Value")
    axes[1][0].grid()
    
    
    axes[1][1].plot(steps, wout1_list, marker='o', label="w_out1")
    axes[1][1].plot(steps, wout2_list, marker='s', label="w_out2")
    axes[1][1].set_title("Output Layer Weights During Training")
    axes[1][1].set_xlabel("Training Step")
    axes[1][1].set_ylabel("Weight Value")
    axes[1][1].legend()
    axes[1][1].grid(True)
    
    
    
    
    """axes[2][0].plot(steps, w11_list, label="w11")
    axes[2][0].plot(steps, w12_list, label="w12")
    axes[2][0].plot(steps, w21_list, label="w21")
    axes[2][0].plot(steps, w22_list, label="w22")
    axes[2][0].set_title("Hidden Layer Weights During Training")
    axes[2][0].set_xlabel("Training Step")
    axes[2][0].set_ylabel("Weight Value")
    axes[2][0].legend()
    axes[2][0].grid(True)"""
 
    
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, w11_list, label="w11")
    plt.plot(steps, w12_list, label="w12")
    plt.plot(steps, w21_list, label="w21")
    plt.plot(steps, w22_list, label="w22")
    plt.title("Hidden Layer Weights During Training")
    plt.xlabel("Training Step")
    plt.ylabel("Weight Value")
    plt.legend()
    plt.grid(True)
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
    x2=random.randint(1,5)
    return x1,x2

####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    