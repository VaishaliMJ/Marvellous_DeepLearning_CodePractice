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
from sklearn.preprocessing import StandardScaler
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
    xTrain,xTest,yTrain,yTest=train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,stratify=y)
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
    clsReport=classification_report(yTest, predResult,zero_division=0)
    print(clsReport)
    print(BORDER)
#####################################################################################################    
#   Function Name   :   testNewData
#   Input Params    :   model
#   Output Params   :   None
#   Description     :   Test New emp record
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################   
def testNewData(model):
    # ------------------------------------------------------------
    # Step 7: Test with new student data
    # Example:
    # Study Hours = 4
    # Attendance = 85
    # Assignment Score = 63
    # ------------------------------------------------------------

    new_student = [[1, 85, 63]]
    prediction = model.predict(new_student)
    print(f"Student Details")
    print(BORDER)
    print(f"Study Hours :{new_student[0][0]} \nAttendance:{new_student[0][1]}\nAssignment Score:{new_student[0][2]}")
    print(BORDER)
    if prediction[0] == 1:
        
        print(f"\nNew Student Prediction : Pass")
    else:
        print(f"\nNew Student Prediction : Fail")
    print(BORDER)  
#####################################################################################################
#   Function Name   :   SplitDataset
#   Input Params    :   xTrain,xTest
#   Output Params   :   X_train_scaled,X_test_scaled,scaler_X 
#   Description     :   Scale Data Set
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def scaleDataSet(xTrain,xTest):

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(xTrain)
    X_test_scaled  = scaler_X.transform(xTest)
    
    return X_train_scaled,X_test_scaled,scaler_X      
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
    X_train_scaled,X_test_scaled,scaler_X=scaleDataSet(xTrain,xTest)

    model=createRNNMOdel()
    predResult=BuildModel(X_train_scaled,X_test_scaled,yTrain,yTest,model)
    calculateLoss(yTest,predResult)
    testNewData(model)
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

