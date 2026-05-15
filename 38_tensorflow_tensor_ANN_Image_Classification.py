"""-----------------------------------------------------------------------------------------------------
                    Tensor ANN Image Classification
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Tensor ANN Image Classification
--------------------------------------------------------------------------------------------------------"""
import tensorflow as tf
from keras import datasets
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Flatten
import numpy as np

BORDER="-"*65
CLASS_NAME = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#####################################################################################################    
#   Function Name   :   loadAndSplitDataset
#   Input Params    :   None
#   Output Params   :   Splitted data set
#   Description     :   Image load And Split Dataset
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def loadAndSplitDataset():
    fashionMnist=datasets.fashion_mnist
    print(BORDER)
    print("Fashion_mnist dataset loaded successfully")
    print(BORDER)

    (train_images, train_labels),(test_images, test_labels) = fashionMnist.load_data()
    print("Dataset Sizes:\n")
    print(f"Train Images :   {train_images.shape}")
    print(f"Train Labels :   {train_labels}")
    print(f"Test Images :   {test_images.shape}")
    print(f"Test Labels :   {test_labels}")

    return (train_images, train_labels),(test_images, test_labels)

#####################################################################################################    
#   Function Name   :   plotImages
#   Input Params    :   train_images
#   Output Params   :   Image plot
#   Description     :   Image Plot
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################   
def plotImages(train_images,train_labels,test_images):
    #fig,ax=plt.subplots(2, 5, figsize=(16, 4))
    for i in range(10):
        #plt.figure()
        plt.subplot(2,5,i+1)
        plt.imshow(train_images[i])
        #plt.colorbar()
        plt.grid(False)
        plt.xlabel(CLASS_NAME[train_labels[i]])
        #plt.title("Marvellous Infosystems : Image")
    plt.show()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.RdBu)
        plt.xlabel(CLASS_NAME[train_labels[i]])
    plt.show()  
#####################################################################################################    
#   Function Name   :   buildModel
#   Input Params    :   train_images,train_labels,test_images,test_labels
#   Output Params   :   model
#   Description     :   Image Classification
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def buildModel(train_images,train_labels,test_images,test_labels):
    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation=tf.nn.relu),
        Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', 
                  loss= 'sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)
    return model,test_acc
#####################################################################################################    
#   Function Name   :   testModel
#   Input Params    :   model,test_images,test_labels
#   Output Params   :   model
#   Description     :   Image Classification
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def testModel(model,test_images,test_labels):
    predictions = model.predict(test_images)

    print("Predicted values for first image - ")
    print(predictions[0])

    print("-------------------------------------------")
    print("Output of Image predictor after training")
    print("-------------------------------------------")
    for i in range(10):
        print("Expected image - ", CLASS_NAME[test_labels[i]])
        print("Predicted image - ", CLASS_NAME[np.argmax(predictions[i])])
        print("-------------------------------------------")
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.RdBu)
        plt.title(f"P:{CLASS_NAME[test_labels[i]]}\nT:{CLASS_NAME[np.argmax(predictions[i])]}",fontsize=9)
    plt.tight_layout()
    plt.show()  
    
    
    
#####################################################################################################    
#   Function Name   :   imageClassification
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Image Classification
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def imageClassification():
    (train_images, train_labels),(test_images, test_labels)=loadAndSplitDataset()
    plotImages(train_images,train_labels,test_images)
    model,testAcc=buildModel(train_images,train_labels,test_images,test_labels)
    testModel(model,test_images,test_labels)
#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    print(BORDER)
    print("Fashion Mnist data set")
    print("Image Classification application based on Deep Learning")
    imageClassification()
    
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
