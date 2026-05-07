"""-----------------------------------------------------------------------------------------------------
                        Salary Prediction 
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Student Result prediction using Meural Network
--------------------------------------------------------------------------------------------------------"""
import random,math
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error,classification_report
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
    # ------------------------------------------------------------
    # Step 1: Prepare the dataset
    # Each row contains:
    # x: [Study Hours, Attendance, Assignment Score]
    # y: Output:0 = Fail,1 = Pass
    # ------------------------------------------------------------

    x = [
            [1, 40, 30],
            [2, 50, 35],
            [3, 60, 40],
            [4, 65, 50],
            [5, 70, 55],
            [6, 75, 65],
            [7, 80, 70],
            [2, 45, 25],
            [8, 90, 85],
            [1, 35, 20],
            [3, 55, 45],
            [4, 68, 52],
            [5, 72, 58],
            [6, 78, 62],
            [7, 85, 75]
        ]

    y = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
    
    print(BORDER)
    print(f"\t\t Student's data:")
    print(BORDER)
    print(f"'Study Hours''Attendance''Assignment Score' 'Result'") 
    for i in range(len(y)):
        print(f"{x[i]}    {y[i]}")   
    return x,y
#####################################################################################################    
#   Function Name   :   createRNNMOdel
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Create RNN model
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def createRNNMOdel():
    model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=RANDOM_STATE
    )

    return model
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
#   Function Name   :   BuildModel
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Create RNN model
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def BuildModel(xTrain,xTest,yTrain,yTest,model):

    # Train Model
    model.fit(xTrain, yTrain)

    # Predict on test data
    predResult = model.predict(xTest)
    
    print(BORDER)
    print("Actual Result   :", yTest)
    print("Predicted Result  :", predResult)
    print(BORDER)
    return predResult
#####################################################################################################    
#   Function Name   :   calculateLoss
#   Input Params    :   yTest,predResult
#   Output Params   :   None
#   Description     :   Calculate Loss
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def calculateLoss(yTest,predResult):
    # Error
    error = mean_absolute_error(yTest, predResult)
    print("\nAverage Error:", error)
    
    # Detailed report
    print("\nClassification Report:")
    clsReport=classification_report(yTest, predResult)
    print(clsReport)
    print(BORDER)
#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    x,y=generateData()
    xTrain,xTest,yTrain,yTest=SplitDataset(x,y)
    model=createRNNMOdel()
    predResult=BuildModel(xTrain,xTest,yTrain,yTest,model)
    calculateLoss(yTest,predResult)
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

