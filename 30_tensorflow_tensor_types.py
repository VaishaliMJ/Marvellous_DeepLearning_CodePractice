"""-----------------------------------------------------------------------------------------------------
                        Tensor Types
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Tensor Types
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
    # Scalar tensor (single value)
    print(BORDER)
    scalar_tensor = tf.constant(10)
    print(f"Scalar Tensor:{scalar_tensor}")
    print(scalar_tensor)
    print(BORDER)
    #   1-D Tensor
    vector_tensor=tf.constant([10,20,30,40])
    print(f"1-D Tensor/Vector:{vector_tensor}")
    print(vector_tensor)
    #   Matrix/2-D tensor
    TwoD_Tensor=tf.constant([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(BORDER)
    print("2D Tensor / Matrix:")
    print(TwoD_Tensor)
    print(BORDER)
    
    #   3-D tensor
    threeD_Tensor=tf.constant([
        [[1,2],[3,4]],
        [[5,6],[7,8]]
    ])
    print("3D Tensor:")
    print(threeD_Tensor)
    print(BORDER)
    
    # Show shape of tensors
    print("Shape of scalar tensor:", scalar_tensor.shape)
    print("Shape of vector tensor:", vector_tensor.shape)
    print("Shape of matrix tensor:", TwoD_Tensor.shape)
    print("Shape of 3D tensor:", threeD_Tensor.shape)
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
