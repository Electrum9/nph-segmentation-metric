import numpy as np
import nibabel as nib
from skimage.morphology import flood_fill, flood

# TODO: Clear out unused imports
# import os
# from pickle import FALSE
# import cv2 as cv


def postSkullStrip(scanFile, maskFile):
    scan = nib.load(scanFile).get_fdata()
    mask = nib.load(maskFile).get_fdata()
    scan = scan.round()

    # ignore voxel values less than 0
    scan[np.where(scan < 0)] = 0

    #ignore bright boundaries
    scan[np.where(scan > 250)] = 0

    # Apply the mask
    scan[np.where(mask > 0)] = 0

    #to ignore background
    scan = flood_fill(scan, (1,1,1), 250, tolerance=0)

    #connectivity - find bright blotches where area is small
    #first find all points where low intensity
    poi = np.where(scan > 150)
    for i in range(len(poi[0])):
        if(not (scan[poi[0][i], poi[1][i], poi[2][i]] == 0)):
            print("checking pt ", poi[0][i], poi[1][i], poi[2][i])
            regMask = flood(scan, (poi[0][i], poi[1][i], poi[2][i]), tolerance=100)
            #evaluate each point to see if in a small region
            if(len(np.nonzero(regMask[0])) < 10):
                print((np.nonzero(regMask)) )
                scan = flood_fill(scan, (poi[0][i], poi[1][i], poi[2][i]), 0, tolerance=100)

    #restore background
    scan = flood_fill(scan, (1,1,1), 0, tolerance=0)

    return scan
