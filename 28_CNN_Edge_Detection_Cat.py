"""-----------------------------------------------------------------------------------------------------
                        CNN Edge Detection Cat
                    (Student name - Vaishali Jorwekar)
--------------------------------------------------------------------------------------------------------
Problem statement:CNN Edge Detection (Cat)
--------------------------------------------------------------------------------------------------------"""
import cv2
#####################################################################################################    
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Main entry point of the program
#   Author          :   Vaishali M. Jorwekar              
#####################################################################################################    
def main():
    # Load image
    img = cv2.imread("sample.png")
    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    # Show Original Image
    cv2.imshow("Original Image", img)

    # Show Grayscale Image
    cv2.imshow("Grayscale Image", gray)

    # Show Edge Detected Image
    cv2.imshow("Edges", edges)

    # Wait until key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#####################################################################################################    
if __name__ =="__main__":
    main()
#####################################################################################################    
