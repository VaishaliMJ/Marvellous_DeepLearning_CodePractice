"""-----------------------------------------------------------------------------------------------------
                        Salary Prediction 
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Salary Prediction 
--------------------------------------------------------------------------------------------------------"""
import random,math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
BORDER="-"*65
TEST_SIZE=0.25
RANDOM_STATE=42

#####################################################################################################
#   Function Name   :   generateData
#   Input Params    :   None
#   Output Params   :   Features,Labels
#   Description     :   Genrates Features and Labels
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def generateData():
    # ----------------------------------------------------
    # Dataset
    # [Experience, Education Score, Skill Rating, Certificates]
    # ----------------------------------------------------
    x = [
        [1,5,4,0],
        [2,6,5,1],
        [3,6,6,1],
        [4,7,7,2],
        [5,7,8,2],
        [6,8,8,3],
        [7,8,9,3],
        [8,9,9,4],
        [10,9,10,5],
        [9,9,10,4]
    ]

    # Salary Output
    y = [22000,26000,32000,40000,47000,
        54000,62000,70000,85000,78000]

    # Convert y into 2D form for scaling
    y = np.array(y).reshape(-1, 1)

        
    return x,y
#####################################################################################################
#   Function Name   :   SplitDataset
#   Input Params    :   x,y
#   Output Params   :   inputs,weights,bias
#   Description     :   Split data set
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def SplitDataset(x,y):
    xTrain,xTest,yTrain,yTest=train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return xTrain,xTest,yTrain,yTest
#####################################################################################################
#   Function Name   :   SplitDataset
#   Input Params    :   xTrain,xTest,yTrain,yTest
#   Output Params   :   Scaled Parameters
#   Description     :   Scale Data Set
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def scaleDataSet(xTrain,xTest,yTrain,yTest):

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(xTrain)
    X_test_scaled  = scaler_X.transform(xTest)

    y_train_scaled = scaler_y.fit_transform(yTrain).ravel()
    y_test_scaled  = scaler_y.transform(yTest).ravel()
    
    return X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y,scaler_X

#####################################################################################################    
#   Function Name   :   createFNNMOdel
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Create RNN model
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def createFNNMOdel():
    model = MLPRegressor(
    hidden_layer_sizes=(6,),
    activation='relu',
    solver='lbfgs',        # better for very small dataset
    max_iter=5000,
    random_state=RANDOM_STATE
)
    return model
#####################################################################################################    
#   Function Name   :   BuildModel
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Build model
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def BuildModel(X_train_scaled,X_test_scaled,y_train_scaled,model):

    # Train Model
    model.fit(X_train_scaled, y_train_scaled)

    # Predict on test data
    pred_scaled = model.predict(X_test_scaled)
    return pred_scaled
#####################################################################################################    
#   Function Name   :   testNewData
#   Input Params    :   scaler_X,scaler_y,model
#   Output Params   :   None
#   Description     :   Test New emp record
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################   
def testNewData(scaler_X,scaler_y,model):
    # ----------------------------------------------------
    # New Employee Prediction
    # Experience = 5 years
    # Education = 8
    # Skill = 9
    # Certifications = 3
    # ----------------------------------------------------
    new_emp = [[5,8,9,3]]
    new_emp_scaled = scaler_X.transform(new_emp)

    salary_scaled = model.predict(new_emp_scaled)
    salary = scaler_y.inverse_transform(salary_scaled.reshape(-1, 1))

    print("\nPredicted Salary for New Employee:", int(salary[0][0]))
#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    x,y=generateData()
    print(BORDER)
    print(f"\t\tInput data:")
    print(BORDER)
    print("Experience Education Score Skill Rating Certificates Salary")
    for i in range(len(y)):
        print(f"{x[i]}    {y[i]}")
    xTrain,xTest,yTrain,yTest=SplitDataset(x,y)
    X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y,scaler_X=scaleDataSet(xTrain,xTest,yTrain,yTest)
    model=createFNNMOdel()
    pred_scaled=BuildModel(X_train_scaled,X_test_scaled,y_train_scaled,model)
    predictions=scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
    
    print("Actual Salaries   :", yTest.ravel())
    print("Predicted Salary  :", predictions.astype(int))

    # Error
    error = mean_absolute_error(yTest.ravel(), predictions)
    print("\nAverage Error:", error)
    
    testNewData(scaler_X,scaler_y,model)
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

