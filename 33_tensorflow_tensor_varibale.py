"""-----------------------------------------------------------------------------------------------------
                        Tensor Variable
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Tensor Variable
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
    weight=tf.Variable(5.5)
    print("Initial Weight Value:")
    print(weight)
    print(BORDER)
    print(weight.numpy())
    print(BORDER)
    
    weight.assign(10.2)
    print("Updated Weight Value:")
    print(weight.numpy())
    print(BORDER)
    
    weight.assign_add(3.2)
    print("After add 3.2 Updated Weight Value:")
    print(weight.numpy())
    print(BORDER)
    
    weight.assign_sub(3.2)
    print("After Subtraction 3.2  Updated Weight Value:")
    print(weight.numpy())
    print(BORDER)
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
