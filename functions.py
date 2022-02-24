import numpy as np
import cv2
import pandas as pd

def get_imflow(img_0, img_1):

    ''''
    Compute the optical flow between two images an return it
    '''

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros((img_0.shape[0],img_0.shape[1],3))

    # Sets image saturation to maximum
    mask[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(img_0, img_1, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    
    return np.array(cv2.cartToPolar(flow[..., 0], flow[..., 1]))

def mag_ang_std_mean(flows):
    mags = []
    angs = []
    for flow in flows:
        mag, ang = flow[0], flow[1]
        mags = np.concatenate((mags,mag.flatten()))
        angs = np.concatenate((angs,ang.flatten()))

    mean_mags = np.mean(mags)
    std_mags = np.std(mags)

    mean_angs = np.mean(angs)
    std_angs = np.std(angs)

    return mean_mags, std_mags, mean_angs, std_angs


    return norm_flow

def write_results(prediction, filename):
    df = pd.DataFrame(columns = ['Id', 'expression'])
    df['expression'] = prediction
    df['Id'] = df.index
    results = df.to_csv(index = False)
    f = open(filename, 'w')
    f.write(results)
    f.close()
