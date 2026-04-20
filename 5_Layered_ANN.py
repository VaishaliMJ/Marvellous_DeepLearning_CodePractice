
"""-----------------------------------------------------------------------------------------------------
                         Layered ANN Calculations
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement: Layered ANN Calculations
--------------------------------------------------------------------------------------------------------"""
import random,math
import numpy as np
import matplotlib.pyplot as plt

BORDER="-"*65
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
#   Function Name   :   plot_relu
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   ReLu Activation function plot
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def plot_relu():
    z_values=np.linspace(-10,10,200)
    reluVal=np.maximum(0,z_values)
    
    plt.figure(figsize=(8,5))
    plt.plot(z_values,reluVal,label="ReLu Activation Function",linewidth=2,color="blue")
    
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.axvline(x=0, color="gray", linestyle="--")
    
    # Labels and title
    plt.title("ReLU Activation Function", fontsize=16)
    plt.xlabel("Input (z)", fontsize=14)
    plt.ylabel("Output", fontsize=14)

    # Grid and legend
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Show graph
    plt.show()

                
    
#####################################################################################################
#   Function Name   :   generateInput
#   Input Params    :   None
#   Output Params   :   inputs,weights,bias
#   Description     :   Genrates inputs,wts and bias randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################

def generateInput():
    inputs=np.array([random.randint(10,20),
                     random.randint(20,30)])
                     #random.randint(1,10)])
    return inputs
#####################################################################################################
#   Function Name   :   generateWeights
#   Input Params    :   input_size, neuron_count
#   Output Params   :   weights
#   Description     :   Genrates weights randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################

def generateWeights(rows,cols):
    weights = np.random.uniform(-1,1,size=(rows,cols))
    return weights
#####################################################################################################
#   Function Name   :   generateBias
#   Input Params    :   None
#   Output Params   :   weights
#   Description     :   Genrates bias randomly
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateBias(col):
    bias=np.random.uniform(-1, 1, size=(col,))
    return bias
#####################################################################################################
#   Function Name   :   hidden_Layers_Calculations
#   Input Params    :   inputs
#   Output Params   :   None
#   Description     :   Hidden Layers Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def hidden_Layers_Calculations(inputs):
    hidden_weights =generateWeights(2,2)
    print("Hidden LAYER \n")
    print(BORDER)

    hidden_biases = generateBias(col=2)
    print(f"Hidden Layer 1 weights  = {hidden_weights}")
    print(f"Hidden layer 1 biases  = {hidden_biases}")

    #Same Function used
    output_weights = generateWeights(1,2)
    output_bias = generateBias(col=1)
    print(f"Output Layer  weights  = {output_weights}")
    print(f"Output layer  biases  = {output_bias}")
    print(BORDER)

    # Layer 1 Calculations
    hidden_outputs=processHiddenLayer(inputs,hidden_weights,hidden_biases)
    # Process output layer
    z_output, final_output = processOutputLayer(
        hidden_outputs,
        output_weights[0],
        output_bias
    )
    displayFinalNNSummary(hidden_outputs, final_output)
    return hidden_outputs
####################################################################################################
#   Function Name   :   displayFinalNNSummary
#   Input Params    :   hidden_outputs,final_output
#   Output Params   :   None
#   Description     :   Hidden Layers Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def displayFinalNNSummary(hidden_outputs, final_output):
    print(BORDER)
    print("\nFINAL SUMMARY\n")
    print(f"Hidden Layer Outputs : {hidden_outputs}")
    print(f"Final Network Output : {final_output:.3f}")
    print(f"Confidence Percentage: {final_output * 100:.2f}%")
    print(BORDER)

    if final_output >= 0.5:
        print("Prediction           : Positive Class")
    else:
        print("Prediction           : Negative Class")

####################################################################################################
#   Function Name   :   processOutputLayer
#   Input Params    :   hidden_outputs, output_weights, output_bias
#   Output Params   :   None
#   Description     :   Hidden Layers Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def processOutputLayer(hidden_outputs, output_weights, output_bias): 
    print("\n================ OUTPUT LAYER ================\n")
    print(BORDER)

    print("Output Neuron:")
    print("  Step 1: Multiply hidden layer outputs by output weights")
    for index in range(len(hidden_outputs)):
        print(
            f"    ({output_weights[index]} * {hidden_outputs[index]:.3f}) = "
            f"{output_weights[index] * hidden_outputs[index]:.3f}"
        )
    # Calculate weighted sum for output layer
    z_output = calculate_Weighted_Sum(hidden_outputs, output_weights, output_bias)
    print(f"  Step 2: Add all multiplication results and bias {output_bias}")
    print(f"    z = {z_output[0]}")
     # Apply Sigmoid activation
    final_output = sigmoid(z_output[0])
    print("  Step 3: Apply Sigmoid activation")
    print(f"    Sigmoid({z_output[0]:.3f}) = {final_output:.3f}")

    return z_output[0], final_output
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
#   Function Name   :   processHiddenLayer
#   Input Params    :   inputs,hidden_weights,hidden_biases
#   Output Params   :   None
#   Description     :   Hidden Layers Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def processHiddenLayer(inputs,hidden_weights,hidden_biases): 
    hidden_outputs = []
    print(BORDER)
    print("\n================ HIDDEN LAYER ================\n")
    print(BORDER)
    for neuronIndex in range(len(hidden_weights)):
        print(f"Hidden Neuron : {neuronIndex+1}")
        print(BORDER)
        current_weights = hidden_weights[neuronIndex]
        current_bias = hidden_biases[neuronIndex]
        displayMultiplcationDetails(inputs,current_weights)
        z_value = calculate_Weighted_Sum(inputs, current_weights, current_bias)
        print(f"  Step 2: Add all multiplication results and bias {current_bias}")
        print(BORDER)
        print(f"    z = {z_value:.3f}")
        print(BORDER)
        
        activated_output = relu(z_value)
        print(f"  Step 3: Apply ReLU activation")
        print(f"    ReLU({z_value:.3f}) = {activated_output:.3f}\n")

        hidden_outputs.append(activated_output)
    return hidden_outputs
#####################################################################################################
#   Function Name   :   relu
#   Input Params    :   x
#   Output Params   :   max(0,x)
#   Description     :   ReLu Activation function
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def relu(x):
    return max(0,x)


####################################################################################################
#   Function Name   :   displayMultiplcationDetails
#   Input Params    :   inputs,hidden_weights
#   Output Params   :   None
#   Description     :   Hidden Layers Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def displayMultiplcationDetails(inputs,weights):
    print(BORDER)
    print("  Step 1: Multiply inputs by corresponding weights")
    print(BORDER)

    for index in range(len(inputs)):
        print(
            f"    ({weights[index]} * {inputs[index]}) = {weights[index] * inputs[index]:.3f}"
        )
    print(BORDER)
    
####################################################################################################
#   Function Name   :   calculate_Weighted_Sum
#   Input Params    :   inputs,weights,biases
#   Output Params   :   None
#   Description     :   Hidden Layers Calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################    
def calculate_Weighted_Sum(inputs, weights, bias):
    #weighted_sum = sum(weight * input_value for weight, input_value in zip(weights, inputs)) + bias
    
    weighted_sum = np.dot(inputs, weights) + bias
    return weighted_sum
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def main():
    inputs=generateInput()
    print(BORDER)
    print("INPUT LAYER \n")
    print(BORDER)
    print(f"Input x1 = {inputs[0]}")
    print(f"Input x2 = {inputs[1]}")
    print(BORDER)
    hidden_Layers_Calculations(inputs)
    
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

