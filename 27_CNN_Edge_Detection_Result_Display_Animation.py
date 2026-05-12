"""-----------------------------------------------------------------------------------------------------
                    CNN Edge Detection Result Display
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:CNN Edge Detection Result Display
--------------------------------------------------------------------------------------------------------"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
BORDER="-"*65

#####################################################################################################    
#   Function Name   :   displayMatrix
#   Input Params    :   matrices
#   Output Params   :   None
#   Description     :   Display matrices step wise
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def displayMatrix(img, kernel,feature_map):
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
#def displayAnimation(img,region,kernel,feature_map,j,i):
def displayAnimation(img,region,kernel,featureMap,result,i,j):
    fig,ax=plt.subplots(1, 4, figsize=(16, 4))

    for a in ax: a.clear()
    
    rect=Rectangle((j-0.5,i-0.5),3,3,linewidth=3, edgecolor='red', facecolor='none')
    
   
    ax[0].add_patch(rect)
    """ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Original Image")
    ax[0].set_xticks(range(6))
    ax[0].set_yticks(range(6))
    drawVMatrixValues(ax[0],img)"""
    showAxesImage(img, ax[0],3,3,"Original Image")

    # -------------------------
    # 2. Current Region
    # -------------------------
    """ax[1].imshow(region, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("Marvellous Current 3x3 Region")
    ax[1].set_xticks(range(3))
    ax[1].set_yticks(range(3))
    drawVMatrixValues(ax[1],img)"""
    showAxesImage(region, ax[1],3,3,"Region")
    # -------------------------
    # 3. Kernel
    # -------------------------
    """ax[2].imshow(kernel, cmap="gray")
    ax[2].set_title("3x3 Kernel")
    ax[2].set_xticks(range(3))
    ax[2].set_yticks(range(3))
    drawVMatrixValues(ax[2],kernel,text_color="white")"""
    showAxesImage(kernel, ax[2],3,3,"Kernel","white")
    # -------------------------
    # 4. Feature Map
    # -------------------------
    """ax[3].imshow(featureMap, cmap="gray")
    ax[3].set_title("Feature Map So Far")
    ax[3].set_xticks(range(4))
    ax[3].set_yticks(range(4))
    drawVMatrixValues(ax[3],featureMap,text_color="white")"""
    showAxesImage(featureMap, ax[3],4,4,"Feature Map","white")
    plt.suptitle(f"Convolution Step | Output at position ({i},{j}) = {result}",
                     fontsize=14, fontweight='bold')
    
    plt.draw()
    plt.pause(0.3)
#####################################################################################################    
#   Function Name   :   showAxesImage
#   Input Params    :   region,ax,xRange,yRange,title,text_color="black"
#   Output Params   :   None
#   Description     :   Display matrices values step wise
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def showAxesImage(region,ax,xRange,yRange,title,text_color="black"):
    ax.imshow(region, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Marvellous Current {xRange}*{yRange} {title}")
    ax.set_xticks(range(xRange))
    ax.set_yticks(range(yRange))
    drawVMatrixValues(ax,region,text_color)
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
#   Function Name   :   drawVMatrixValues
#   Input Params    :   Matrix value printing
#   Output Params   :   None
#   Description     :   Prints values of matrix 
#   Author          :   Vaishali M. Jorwekar              
##################################################################################################### 
def drawVMatrixValues(ax, matrix, text_color="black"):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, str(matrix[i][j]),
                    ha='center', va='center',
                    fontsize=12, color=text_color, fontweight='bold')
            
#####################################################################################################    
#   Function Name   :   convolutionOperation
#   Input Params    :   img,kernel
#   Output Params   :   featureMap
#   Description     :   img*kernel operation
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
            print("\n\n==================================================")
            print(f"Kernel Position -> Row: {i} to {i+2}, Column: {j} to {j+2}")
            print("==================================================")

            # Extract 3x3 region
            region = img[i:i+3, j:j+3]

            # Multiply and Sum
            result = np.sum(region * kernel)
            # Store result
            feature_map[i][j] = result
            print(f"Image\t* \tKernel\t=\tFeatureMap")
            print(BORDER)
            displayMatrix(region,kernel,feature_map)
            #   Display Animation
            displayAnimation(img,region,kernel,feature_map,result,i,j)
    #plt.show()    
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
    """fig,ax=plt.subplots(1, 4, figsize=(16, 4))
    ani = FuncAnimation(fig, displayAnimation, frames=16, interval=500,fargs=(img,kernel,featureMap,ax),repeat=False)
    plt.show()"""
    print("\nFeature Map (Detected Edge)")
    print(featureMap)
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

