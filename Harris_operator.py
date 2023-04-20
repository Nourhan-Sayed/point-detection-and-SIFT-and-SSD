import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corner_detector(image, block_size=3, ksize=3, k=0.04, threshold=0.1):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the derivatives with respect to x and y using the Sobel operator
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Compute the second-order moments of the image
    Ixx = cv2.GaussianBlur(Ix * Ix, (block_size, block_size), 0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (block_size, block_size), 0)
    Iyy = cv2.GaussianBlur(Iy * Iy, (block_size, block_size), 0)
    
    # Compute the Harris response function using the second-order moments
    det_M = (Ixx * Iyy) - (Ixy ** 2)
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M ** 2)
    
    # Threshold the Harris response function to obtain a binary image of corner points
    corner_mask = np.zeros_like(gray)
    corner_mask[R > threshold*R.max()] = 1
    
    # Apply non-maximum suppression to eliminate multiple responses in close proximity
    corner_mask = cv2.dilate(corner_mask, None)
    corner_mask = cv2.erode(corner_mask, None)
    image[corner_mask != 0] = [0, 0, 255]  # Set corner points to blue
    plt.imshow(image,cmap='gray')
    plt.imsave("images/output.png",image,cmap='gray')
    # plt.savefig("images/output.png")
    # Return the corner mask

    return 
# img = cv2.imread('images/Sift.jpg')

# Apply the Harris corner detector
# corner_mask = harris_corner_detector(img)

# Display the corner points on the original image
# img[corner_mask != 0] = [0, 0, 255]  # Set corner points to red
# cv2.imshow('Harris Corner Detector', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imshow(img,cmap='gray')
# plt.show()

