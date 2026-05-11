"""-----------------------------------------------------------------------------------------------------
                        CNN Edge Detection
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:CNN Edge Detection
--------------------------------------------------------------------------------------------------------"""
from PIL import Image
import numpy as np
BORDER="-"*65

#####################################################################################################    
#   Function Name   :   imagePixels
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Image in Pixels format
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def Marvellous_Print_Matrix(title, matrix):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)
    print(matrix)
#####################################################################################################    
#   Function Name   :   imagePixels
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Image in Pixels format
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def imagePixels():
    image = np.array([
        [0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0],
        [255,255,255,255,255,255],
        [255,255,255,255,255,255],
        [255,255,255,255,255,255]
    ])

    print("\nOriginal 6x6 Image")
    print(image)

    return image
#####################################################################################################    
#   Function Name   :   kernelPixels
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   3x3 Kernel for Horizontal Edge Detection
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def kernelPixels():
    kernel = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ])

    print("\n3x3 Kernel")
    print(kernel)
    return kernel
#####################################################################################################    
#   Function Name   :   convolutionOperation
#   Input Params    :   img,kernel
#   Output Params   :   featureMap
#   Description     :   3x3 Kernel for Horizontal Edge Detection
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def convolutionOperation(img,kernel):
    feature_map = np.zeros((4,4))

    #   Output Size = (6-3+1) x (6-3+1) = 4x4
    row=img.shape[0]-kernel.shape[0]+1
    col=img.shape[1]-kernel.shape[1]+1
    feature_map = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            # Extract 3x3 region
            region = img[i:i+3, j:j+3]

            # Multiply and Sum
            result = np.sum(region * kernel)
            # Store result
            feature_map[i][j] = result
    return feature_map
#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    img=imagePixels()
    kernel=kernelPixels()
    featureMap=convolutionOperation(img,kernel)
    print("\nFeature Map (Detected Edge)")
    print(featureMap)
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

