"""-----------------------------------------------------------------------------------------------------
                        Tensor Reshape
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Tensor Reshape
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
    # Create 1D tensor with 12 values
    originalTensor = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    print(BORDER)
    print(f"Original Tensor : {originalTensor}")
    
    print(BORDER)
    reshapeTensor=tf.reshape(originalTensor,(3,4))
    print(f"Reshaped Tensor :\n {reshapeTensor}")
    print(BORDER)
    
    reshaped3D=tf.reshape(originalTensor,(3,2,2))
    print(f"Reshaped 3D :\n {reshaped3D}")
    print(BORDER)
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
