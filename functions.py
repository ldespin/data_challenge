import numpy as np
import cv2

def get_imflow(img_0, img_1):

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros((img_0.shape[0],img_0.shape[1],3))

    # Sets image saturation to maximum
    mask[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(img_0, img_1, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    
    return flow