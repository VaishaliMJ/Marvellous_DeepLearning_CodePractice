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
    tensor1 = tf.constant([10, 20, 30])
    tensor2 = tf.constant([40,50,60])
    
    print(BORDER)
    print("Tensor Flow Operations")
    
    resultAdd=tf.add(tensor1,tensor2)
    print(f"Addition of {tensor1} and {tensor2} is : {resultAdd}")
    
    print(BORDER)
    
    resultSub=tf.subtract(tensor1,tensor2)
    print(f"Subtraction of {tensor1} and {tensor2} is : {resultSub}")
    
    print(BORDER)
    resultMult=tf.multiply(tensor1,tensor2)
    print(f"Multiplication of {tensor1} and {tensor2} is : {resultMult}")
    
    print(BORDER)
    resultDiv=tf.divide(tensor1,tensor2)
    print(f"Division of {tensor1} and {tensor2} is : {resultDiv}")
    
    print(BORDER)
    resultSquare=tf.square(tensor1)
    print(f"Sqaure of {tensor1} : {resultSquare}")
    
    print(BORDER)
    resultReduceSum=tf.reduce_sum(tensor1)
    print(f"Reduced Sum {tensor1} :{resultReduceSum}")
    
    print(BORDER)
    resultReduceMean=tf.reduce_mean(tensor1)
    print(f"Reduced Sum {tensor1} :{resultReduceMean}")


#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
