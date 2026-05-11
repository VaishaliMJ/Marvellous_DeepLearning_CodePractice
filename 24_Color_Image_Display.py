"""-----------------------------------------------------------------------------------------------------
                        Color Image Display 
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:Color Image Display
--------------------------------------------------------------------------------------------------------"""
from PIL import Image
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
    img=Image.open("color.png")
    
    img = img.resize((28,28))

    pixels=np.array(img)
    
    print(BORDER)
    print(f"Image Information")
    print(BORDER)
    print(f"Image Shape:{pixels.shape} ")
    print(f"Image Height/Rows:{pixels.shape[0]} ")
    print(f"Image Width/Columns:{pixels.shape[1]} ")
    print(f"Image Channels/(R,G,B):{pixels.shape[2]} ")
    total_pixels = pixels.shape[0] * pixels.shape[1]
    print(f"Total Pixels:{total_pixels}")
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    

