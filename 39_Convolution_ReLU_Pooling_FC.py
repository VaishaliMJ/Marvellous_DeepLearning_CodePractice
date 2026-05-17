"""-----------------------------------------------------------------------------------------------------
                    Convolution ReLU Pooling Fully Connected 
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Convolution ReLU Pooling Fully Connected 
--------------------------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BORDER="-"*65
IMAGE_WIDTH=5
IMAGE_HEIGHT=5
VERTICAL="Vertical"
HORIZONTAL="Horizontal"

####################################################################################################    
#   Function Name   :   generateImage
#   Input Params    :   Image Type
#   Output Params   :   Image as per type
#   Description     :   Generate Image
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################
def generateImage(imageType):
    if imageType==VERTICAL:
        image = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=float)
    else:
        image = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=float)    
    return image
####################################################################################################    
#   Function Name   :   generateKernel
#   Input Params    :   Kernel Type
#   Output Params   :   Kernel according to line type
#   Description     :   Generate Kernel
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################
def generateKernel(imageType):
    if imageType==VERTICAL:
        kernel = np.array([
             [-1,  1, -1],
             [-1,  1, -1],
             [-1,  1, -1]
            ], dtype=float)
    else:
        kernel = np.array([
            [-1,  -1, -1],
            [1,   1, 1],
            [-1,  -1, -1]
            ], dtype=float)    
    return kernel
#####################################################################################################    
#   Function Name   :   displayMatrixResult
#   Input Params    :   matrices
#   Output Params   :   None
#   Description     :   Display matrices step wise
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def displayMatrixResult(img, kernel,feature_map):
    img_mat = str(img).split('\n')
    kernel_mat = str(kernel).split('\n')
    result_mat = str(feature_map).split('\n')
    for i, j, k in zip(img_mat, kernel_mat, result_mat):
        if i == img_mat[1]: 
             print(f"{i}   *    {j}   =   {k}")
        else:
             print(f"{i}       {j}       {k}")    
    print(BORDER)
#####################################################################################################    
#   Function Name   :   displayAnimation
#   Input Params    :   img, kernel,feature_map
#   Output Params   :   None
#   Description     :   Display matrices step wise
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def displayAnimation(img,region,kernel,featureMap,result,i,j):
    fig,ax=plt.subplots(1, 4, figsize=(16, 4))
    for a in ax: a.clear()
    rect=Rectangle((j-0.5,i-0.5),
                   kernel.shape[0],kernel.shape[1],
                   linewidth=3, 
                   edgecolor='yellow', 
                   facecolor='none')
    
    ax[0].add_patch(rect)
    showAxesImage(img, ax[0],img.shape[0],img.shape[1],"Original Image")
    # -------------------------
    # 2. Current Region
    # -------------------------
    showAxesImage(region, ax[1],region.shape[0],region.shape[1],"Region")
    # -------------------------
    # 3. Kernel
    # -------------------------
    showAxesImage(kernel, ax[2],kernel.shape[0],kernel.shape[1],"Kernel")
    # -------------------------
    # 4. Feature Map
    # -------------------------
    showAxesImage(featureMap, ax[3],featureMap.shape[0],featureMap.shape[1]," Feature Map")
    plt.suptitle(f"Convolution Step | Output at position ({i},{j}) = {result}",
                     fontsize=14, fontweight='bold')
    plt.draw()
    plt.pause(1) 
    plt.close()
#####################################################################################################    
#   Function Name   :   displayFinalMatrices
#   Input Params    :   img, kernel,feature_map
#   Output Params   :   None
#   Description     :   Display matrices step wise
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def displayFinalMatrices(image,kernel,output):
    fig,ax=plt.subplots(1, 3, figsize=(16, 4))

    for a in ax: a.clear()
    showAxesImage(image, ax[0],image.shape[0],image.shape[1],"Original Image","red")
    # -------------------------
    # 3. Kernel
    # -------------------------
    showAxesImage(kernel, ax[1],kernel.shape[0],kernel.shape[1],"Kernel","red")
    # -------------------------
    # 4. Feature Map
    # -------------------------
    showAxesImage(output, ax[2],output.shape[0],output.shape[1]," Feature Map","red")
    plt.suptitle(f"Convolution Final Output",
                     fontsize=14, fontweight='bold')
    plt.show()
#####################################################################################################    
#   Function Name   :   displayMatrices
#   Input Params    :   matrices
#   Output Params   :   None
#   Description     :   Display matrices step wise
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def displayMatrices(*matrices,titles,stepName):
    matLen=len(matrices)
    fig,ax=plt.subplots(1, matLen, figsize=(16, 4))

    #for a in ax: a.clear()
    if matLen == 1:
        ax = [ax]
        
    for cnt, a in enumerate(ax): 
        a.clear()
        showAxesImage(matrices[cnt], 
                        ax[cnt],
                        matrices[cnt].shape[0],
                        matrices[cnt].shape[1],
                        titles[cnt],
                        "red")
    # -------------------------
    # 3. Kernel
    # -------------------------
    #showAxesImage(matrices[1], ax[1],matrices[1].shape[0],matrices[1].shape[1],"Kernel","red")
    # -------------------------
    # 4. Feature Map
    # -------------------------
    #showAxesImage(matrices[2], ax[2],matrices[2].shape[0],matrices[2].shape[1]," Feature Map","red")
    plt.suptitle(f"Output after {stepName}",
                     fontsize=14, fontweight='bold')
    plt.show()    
#####################################################################################################    
#   Function Name   :   showAxesImage
#   Input Params    :   region,ax,xRange,yRange,title,text_color="black"
#   Output Params   :   None
#   Description     :   Display matrices values step wise
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def showAxesImage(region,ax,xRange,yRange,title,text_color="red"):
    #ax.imshow(region, cmap="gray", vmin=0, vmax=255)
    ax.imshow(region, cmap='gray', interpolation='nearest')

    ax.set_title(f"({xRange}*{yRange}) {title}")
    ax.set_xticks(range(xRange))
    ax.set_yticks(range(yRange))
    drawVMatrixValues(ax,region,text_color)  
#####################################################################################################    
#   Function Name   :   drawVMatrixValues
#   Input Params    :   Matrix value printing
#   Output Params   :   None
#   Description     :   Prints values of matrix 
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def drawVMatrixValues(ax, matrix, text_color="red"):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, str(matrix[i][j]),
                    ha='center', va='center',
                    fontsize=12, color=text_color, fontweight='bold')
                             
#####################################################################################################    
#   Function Name   :   convolutionOperation
#   Input Params    :   image,kernel
#   Output Params   :   Convolution Matrix
#   Description     :   Convolution Layer Calculations
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def convolutionOperation(image,kernel):
    imgRows,imgCols=image.shape
    kernelRows,kernelCols=kernel.shape
    
    outputRows=imgRows-kernelRows+1
    outputCols=imgCols-kernelCols+1
    
    output = np.zeros((outputRows, outputCols))
    
    print(BORDER)
    print("STEP 1 : Convolution Layer Processing...")
    print(BORDER)
    for i in range(outputRows):
        for j in range(outputCols):
            region = image[i:i+kernelRows, j:j+kernelCols]
            multiplication = region * kernel
            result = np.sum(multiplication)

            output[i][j] = result
            print(f"\nRegion position -> Row:{i} Column:{j}")
            print("\nSelected Region    Kernel      Output")
            print()
            displayMatrixResult(region,kernel,output)
            #   Display Graphically
            displayAnimation(image,region,kernel,output,result,i,j)
    #   Display final output        
    #displayFinalMatrices(image,kernel,output)
    return output
#####################################################################################################    
#   Function Name   :   ReLULayerOperation
#   Input Params    :   convOutput
#   Output Params   :   ReLU output
#   Description     :   ReLu Layer Calculations
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def ReLULayerOperation(convOutput):
    print(BORDER)
    print("STEP 2 : RELU ACTIVATION")
    print(BORDER)
    output = np.maximum(0, convOutput)

    print("\nInput to ReLU:")
    print(convOutput)

    print("\nRule : ReLU(x) = max(0, x)")
    print("\nOutput after ReLU:")
    print(output)

    return output
#####################################################################################################    
#   Function Name   :   poolingLayerOperation
#   Input Params    :   reluOutput
#   Output Params   :   Pooling layer output
#   Description     :   Pooling Layer Calculations
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def poolingLayerOperation(reluOutput):
    
    outputRows,outputCols=reluOutput.shape
    rows = outputRows // 2
    cols = outputCols // 2
    
    output = np.zeros((rows, cols))
    print(BORDER)
    print("STEP 3 : MAX POOLING")
    print(BORDER)
    
    r=0
    for i in range(0, outputRows, 2):
        c=0
        for j in range(0,outputCols,2):
            block = reluOutput[i:i+2, j:j+2]
            
            # Skip incomplete blocks if any
            if block.shape != (2, 2):
                continue
            max_value = np.max(block)
            output[r][c] = max_value
            print(f"\nPooling Block position -> Row:{r} Column:{c}")
            print("\nSelected 2x2 Block:")
            print(block)

            print("\nMaximum value selected =", max_value)

            c += 1
        r += 1
    print("\nFinal Pooling Output:")
    print(output)

    return output
#####################################################################################################    
#   Function Name   :   flattenLayerOperation
#   Input Params    :   poolingOutput
#   Output Params   :   Pooling layer output
#   Description     :   Flatten Layer Calculations
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def flattenLayerOperation(poolingOutput):
    print(BORDER)
    print("STEP 4 : FLATTEN LAYER")
    print(BORDER)

    flat = poolingOutput.flatten()

    print("\nInput to Flatten:")
    print(poolingOutput)

    print("\nFlattened Output:")
    print(flat)

    return flat
#####################################################################################################    
#   Function Name   :   fullyConnectedLayerOperation
#   Input Params    :   poolingOutput
#   Output Params   :   Pooling layer output
#   Description     :   Flatten Layer Calculations
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def fullyConnectedLayerOperation(flattenOP):
    print(BORDER)
    print("STEP 5 : FULLY CONNECTED LAYER")
    print(BORDER)
    print(flattenOP.shape)
    # manual weights
    weights = np.array([0.1, 0.8, 1, 1], dtype=float)
    bias = 0.0

    print("\nFlatten Input:")
    print(flattenOP)

    print("\nWeights:")
    print(weights)

    print("\nBias:")
    print(bias)

    multiplication = flattenOP * weights
    result = np.sum(multiplication) + bias

    print("\nInput * Weights:")
    print(multiplication)

    print("\nSum =", np.sum(multiplication))
    print("Final Output after adding bias =", result)

    return result

#####################################################################################################    
#   Function Name   :   CNNProcess
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Convolution Operation Function
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################
def CNNProcess():
    
    print(BORDER)
    print("Choose Input Image")
    print("1 : Vertical Line Detection")
    print("2 : Horizontal Line Detection")
    print(BORDER)
    choice = int(input("Enter your choice : "))
    if choice==1:
        image=generateImage(VERTICAL)
        kernel=generateKernel(VERTICAL)
        imageDetection=VERTICAL
    else:
        image=generateImage(HORIZONTAL)  
        kernel= generateKernel(HORIZONTAL) 
        imageDetection=HORIZONTAL
    print(BORDER)
    print(f"Image And Kernel Used for detection of {imageDetection} Line")
    print(f"Image Used:\n{image}")
    print(BORDER)
    print(f"Kernel Used:\n{kernel}")
    print(BORDER)
    
    # --------------------------------------------------------
    # Step 1 : Convolution Layer Operation
    # --------------------------------------------------------
    convOutput=convolutionOperation(image,kernel)
    displayFinalMatrices(image,kernel,convOutput)

    # --------------------------------------------------------
    # Step 2 : ReLU
    # --------------------------------------------------------
    reluOutput = ReLULayerOperation(convOutput)
    displayMatrices(image,kernel,convOutput,reluOutput,
                    titles=["Original Image","Kernel","Convolution Output","Relu Output"],
                    stepName="ReLU")
    # --------------------------------------------------------
    # Step 3 : Pooling
    # --------------------------------------------------------
    poolingOutput = poolingLayerOperation(reluOutput)
    displayMatrices(image,kernel,convOutput,reluOutput,poolingOutput,
                    titles=["Original Image","Kernel","Convolution Output","Relu Output","Pooling Output"],
                    stepName="Pooling Output")
    # --------------------------------------------------------
    # Step 4 : Flatten
    # --------------------------------------------------------
    flatOp = flattenLayerOperation(poolingOutput)
    # --------------------------------------------------------
    # Step 5 : Fully Connected Layer
    # --------------------------------------------------------
    result = fullyConnectedLayerOperation(flatOp)
    
    # --------------------------------------------------------
    # Final Prediction
    # --------------------------------------------------------
    print(BORDER)
    print("STEP 6 : FINAL PREDICTION")
    print(BORDER)

    print("\nFinal Score =", result)

    if result > 0:
        prediction = VERTICAL
    else:
        prediction = HORIZONTAL

    print(f"\nPredicted Output Line ={prediction}")
    print(f"Actual Input Line     ={imageDetection}")
    
#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    CNNProcess()
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
