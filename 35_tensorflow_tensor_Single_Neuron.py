"""-----------------------------------------------------------------------------------------------------
                        Tensor Single Neuron
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Tensor Single Neuron
--------------------------------------------------------------------------------------------------------"""
import tensorflow as tf
BORDER="-"*65

#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    print(BORDER)
    # Input values
    inputs = tf.constant([1.0, 2.0, 3.0])
    #   Weight values
    weights=tf.constant([2.0,0.5,0.4])
    
    bias=tf.constant(0.1)
    
    weightedSum=tf.reduce_sum(inputs*weights)+bias
    
    output=tf.sigmoid(weightedSum)
    
    print(f"Inputs:{inputs.numpy()}")
    print(f"Weights:{weights.numpy()}")
    print(f"Bias:{bias.numpy()}")
    print(f"Weighted Sum:{weightedSum.numpy()}")
    print(f"Neuron Output after Sigmoid: {output.numpy()}")
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
