"""-----------------------------------------------------------------------------------------------------
                        Tensor matrix operations
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Tensor matrix operations
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
    matrix1=tf.constant([
        [1,2],
        [3,4]
    ],dtype=tf.float32)
    
    matrix2=tf.constant([
        [5,6],
        [7,8]
    ],dtype=tf.float32)
    print(BORDER)
    
    print(f"Matrix 1 :\n{matrix1}")
    
    print(BORDER)
    print(f"Matrix 2:\n{matrix2}")
    
    print(BORDER)
    result=tf.matmul(matrix1,matrix2)
    print(f"\nMatrix Multiplication:\n{result}")
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
