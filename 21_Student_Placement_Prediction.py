"""-----------------------------------------------------------------------------------------------------
                        Student Placement Prediction
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Student Placement prediction using Meural Network
--------------------------------------------------------------------------------------------------------"""
import pandas as pd,os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ( 
                             mean_absolute_error,
                             classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
###########################################################################################
#   Constants
###########################################################################################
DATASET_PATH="placement_data.csv"
MODEL_NAME="StudentPlacementModel.pkl"
SCALAR_NAME="StudentPlacementScalar.pkl"
TEST_SIZE=0.25
RANDOM_STATE=42
BORDER="-"*65

###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   5 May 2026
############################################################################################
def readCSVFile():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
        f"Dataset not found. Please keep {DATASET_PATH} in the same folder."
    )
    dFrame=pd.read_csv(DATASET_PATH)
    
    print(f"Data loaded successfully from file '{DATASET_PATH}'")
    
    print("Step 1:  Data Loading")

   
    print("\nDataset Shape:", dFrame.shape)
    print("Total Rows   :", dFrame.shape[0])
    print("Total Columns:", dFrame.shape[1])
    print("Column Names :", list(dFrame.columns))
    print("\nStatistical Summary:")
    print(dFrame.describe())

    print("\nCheck Missing Values:")
    print(dFrame.isnull().sum())
    
    return dFrame  
#####################################################################################################
#   Function Name   :   SplitDataset
#   Input Params    :   dataFrame
#   Output Params   :   xTrain,xTest,yTrain,yTest
#   Description     :   Split data set
#   Author          :   Vaishali M. Jorwekar             
#####################################################################################################
def SplitDataset(dataFrame):
    #   x = input features
    x = dataFrame[["Aptitude", "Coding", "Communication", "Academics", "Internship"]]

    # y = target output
    y = dataFrame["Placed"]

    xTrain,xTest,yTrain,yTest=train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,stratify=y)
    return xTrain,xTest,yTrain,yTest
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
#   Function Name   :   createFNNMOdel
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Create RNN model
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def createFNNMOdel():
    model = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=RANDOM_STATE
    )
    return model 
#####################################################################################################    
#   Function Name   :   BuildModel
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Build Model
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
#   Function name    :  saveModel
#   Description      :  Save the trained model
#   Input Params     :  model,modelName
#   Output           :  -
#####################################################################################################
def saveModel(model,modelName):
    joblib.dump(model,modelName)
    print(f"Model saved to path :{modelName}") 
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Description      :  Load the trained model
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#####################################################################################################
def loadTrainedModel(path):
    model = joblib.load(path)
    print(f"Model loaded from the path :{path}")
    return model        
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
    confMat = confusion_matrix(yTest, predResult)
    print(f"\nConfusion Matrix:\n{confMat}")
    ConfusionMatrixDisplay.from_predictions(yTest, predResult)
    plt.show()
    print(BORDER)
#####################################################################################################    
#   Function Name   :   testNewData
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Test New emp record
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################   
def testNewData():
    model=loadTrainedModel(MODEL_NAME)
    scalarX=loadTrainedModel(SCALAR_NAME)
    # ------------------------------------------------------------
    # Step 7: Test with new student data
    # Aptitude = 70
    # Coding = 72
    # Communication = 75
    # Academics = 74
    # Internship = 1
    # ------------------------------------------------------------

    new_student = pd.DataFrame([[70, 72, 75, 74, 1]],
                           columns=["Aptitude", "Coding", "Communication", "Academics", "Internship"])

    prediction = model.predict(new_student)
    print(f"Student Details")
    print(BORDER)
    print(new_student)
    print(BORDER)
    if prediction[0] == 1:
        
        print(f"\nNew Student Prediction : Student Placed")
    else:
        print(f"\nNew Student Prediction : Student Not Placed")
    print(BORDER)      
#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    dataFrame=readCSVFile()
    xTrain,xTest,yTrain,yTest=SplitDataset(dataFrame)
    X_train_scaled,X_test_scaled,scaler_X=scaleDataSet(xTrain,xTest)
    model=createFNNMOdel()
    predResult=BuildModel(X_train_scaled,X_test_scaled,yTrain,yTest,model)
    calculateLoss(yTest,predResult)
    saveModel(model,MODEL_NAME)
    saveModel(scaler_X,SCALAR_NAME)
    testNewData()
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
