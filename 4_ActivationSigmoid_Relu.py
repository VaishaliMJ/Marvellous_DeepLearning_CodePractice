
"""-----------------------------------------------------------------------------------------------------
                         Activation Sigmoid Vs Relu
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement: Sigmoid Vs ReLu Activation Function 
--------------------------------------------------------------------------------------------------------"""
import random,math
import numpy as np
import matplotlib.pyplot as plt
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
#   Function Name   :   sigmoid
#   Input Params    :   z
#   Output Params   :   max(0,z)
#   Description     :   sigmoid Activation function
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def sigmoid(z):
    return 1 / (1 + math.exp(-z))
#####################################################################################################
#   Function Name   :   plot_sigmoid_relu
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   ReLu Activation function plot
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def plot_sigmoid_relu():
    
    z_values=np.linspace(-10,10,200)
    sigmoid_values = 1 / (1 + np.exp(-z_values))
    relu_values = np.maximum(0, z_values)

    plt.figure(figsize=(8, 5))

    # Plot both functions
   
    
    plt.figure(figsize=(8,5))
    plt.plot(z_values, sigmoid_values, label="Sigmoid", linewidth=2)
    plt.plot(z_values, relu_values, label="ReLU", linewidth=2)
    
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.axhline(y=1, color="black", linewidth=0.5)
    plt.axvline(x=0, color="gray", linestyle="--")
    
    # Labels and title
    plt.title("Sigmoid Vs ReLu Activation Function", fontsize=16)
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
                     random.randint(20,30),
                     random.randint(1,10)])
    weights = np.array([random.uniform(0,1),
                        random.uniform(0,1),
                        random.uniform(0,1)])
    bias=random.uniform(0.1,1)
    return inputs,weights,bias
#####################################################################################################
#   Function Name   :   activation_function
#   Input Params    :   inputs,weights,bias,activation_fun_name
#   Output Params   :   None
#   Description     :   Activation function calculations
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def activation_function(inputs,weights,bias,activation_fun_name):
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    y_hat = activation_fun_name(z)
    
    print(f"Input   :   {inputs}")
    print(f"Weights :   {weights}")
    print(f"bias    :   {bias}")
    print(f"Weighted Sum (z=sum(w.x+b))   :   {z}")
    print(f"Final Output    :{y_hat}")
    
    print(f" Output of Activation Function :{activation_fun_name.__name__} is : {y_hat}")
    return z,y_hat
#####################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def main():
    inputs,weights,bias=generateInput()

    
    
    z,y_hat=activation_function(inputs,weights,bias,sigmoid)
    z,y_hat=activation_function(inputs,weights,bias,relu)

    
    
    plot_sigmoid_relu()
    
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

