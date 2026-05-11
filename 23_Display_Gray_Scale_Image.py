"""-----------------------------------------------------------------------------------------------------
                        Gray Scale Image Conversion 
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Gray Scale Image Conversion
--------------------------------------------------------------------------------------------------------"""
from PIL import Image
import numpy as np

#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    img=Image.open("digit_28x28.png")
    img = img.convert("L")      # grayscale
    img = img.resize((28,28))

    pixels=np.array(img)
    print(f"Image Size :{pixels.shape}")
    print("\nPixel Values:\n")
    print(pixels)

#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

