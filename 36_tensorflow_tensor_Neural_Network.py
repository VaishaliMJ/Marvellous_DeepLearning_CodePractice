"""-----------------------------------------------------------------------------------------------------
                        Tensor Neural Network
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Tensor Neural Network
--------------------------------------------------------------------------------------------------------"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
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
    X=np.array([[1],[2],[3],[4],[5]],dtype=float)
    Y=np.array([[2],[4],[6],[8],[10]],dtype=float)
    
    model=Sequential()
    model.add(Dense(4,activation='relu'))
    # Add output layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    # Train model
    model.fit(X, Y, epochs=200, verbose=0)

    print("Training completed")
    # Test prediction
    test_value = np.array([[6]], dtype=float)
    prediction = model.predict(test_value, verbose=2)

    print(f"Prediction for input 6: {prediction[0][0]:.4f}")
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
