# TremaineConsultingGroup.com
# Brian Tremaine
# April 2018
# Course Project for OpenCV class by Satya Mallick, CV for Faces
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
""" Identify coin bounding box and ID-1 reference box
    using *** Watershed algorithm ***
    
    weakness: 
       1. Need very black background and light (high contrast) ID reference
       2. Errors when coins touch, contours cover multiple coins.  
           -- contours used to find non-circular objects          
"""

import cv2
import numpy as np
import logging, sys
import yaml
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

VIDEO = 0
camera = 0
CAL = True
  # Beaware that camera calibration can move some objects out of the frame
  # and as a result contours will be incomplete and return a very small area
  #

#qIDref = 2.00*25.4
IDref = 53.98

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def getBoxDim(box1):
    # return lenght and width of rotated rect
    L = box1[1][1]
    W = box1[1][0]
    
    return max(L,W), min(L,W)

def checkCoin(rc, Wp):
    # check if radius is valid coin
    #
    ## add code to check absolute size using L and W   
    ## dia (mm)
    ## dime:        17.91
    ## penny:       19.05
    ## nickel:      21.21
    ## quarter:     24.26
    ## fifty-cent:  30.61
    ## ID-1:        (85.6 x 53.98)
    ## business crd ( 3.5*25.4 x 2.0*25.4 )
    dia = rc * 2.0 * (IDref) / Wp
    
    # check coin dia in mm
    result = -1.0
    if dia< 16.5:
        result = -1.0
    elif dia < 18.48:
        result = 10.0
    elif dia < 20.13:
        result = 1.0
    elif dia < 22.74:
        result = 5.0
    elif dia < 27.43:
        result = 25.0
    elif dia < 32.0:
        result = 50.0
   
    return result, dia

def imfill(img_in):
    # input image is grayscale
 
    N = 235
    # Threshold.
    # Set values equal to or above N to 0.
    # Set values below N to 255.

    th, im_th = cv2.threshold(img_in, N, 255, cv2.THRESH_BINARY_INV);
 
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
 
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
 
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255); 
 
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    
    return im_out

""" ============ main ======================================
"""
if __name__ == '__main__':
    
  logging.basicConfig(filename='coins.log',filemode='w',level=logging.DEBUG)
  logging.info("Program started")    

  if VIDEO:
     cap = cv2.VideoCapture(camera)
     logging.info("video capture")
     if (cap.isOpened()== False):
        logging.info("Error opening video stream or file")
  else:
     im = cv2.imread('..\images/ID_card1.jpg')
     im = cv2.imread('..\images/ID_card2.jpg')
     im = cv2.imread('..\images/Lucky_ID.jpg')
     im = cv2.imread('..\images/ID_card1.jpg')
     im = cv2.imread('..\images/ID_card2.jpg')
     im = cv2.imread('..\images/velvBkgd.jpg')
     #im = cv2.imread('..\images/non-touch.jpg')
     #im = cv2.imread('..\images\ID-1.jpg')
     #im = cv2.imread('..\images/velvBkgd.jpg')
     logging.info("still image")
     

  while(True):
  # read image
  # ========================================================
  
     if VIDEO:
        success, im = cap.read()
        if not success:
           logging.info('Failed to read video')
           sys.exit(1)
		
     # We will run Object detection at an fixed height image
     finalHeight = 480
     # resize image to height finalHeight
     scale = finalHeight / im.shape[0]
     image = cv2.resize(im, None, fx=scale, fy=scale)
     cv2.imshow("Original",image)
     
     
     """ Apply camera calibration here, using stored matrices
     # ======================================================
     # file used is calibrate.py
     # matrices stored are:
     """
  
     if CAL:
       with open('calibration.yaml') as f:
          loadeddict = yaml.load(f)
          K = loadeddict.get('camera_matrix')
          K = np.array(K)
          d = loadeddict.get('dist_coeff')    
          d = np.array(d) 
         
       # Read an example image and acquire its size
       h, w = image.shape[:2]
       # Generate new camera matrix from parameters
       newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)
       # Generate look-up tables for remapping the camera image
       mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)
       # Remap the original image to a new image
       newimg = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

       # Display old and new image
       if(0):
          cv2.imshow("Before map", image)
          cv2.imshow("After map", newimg)
      
       imageCorr = newimg
       
     else:
       imageCorr = image
  
    
     """ alter gamma
     """
     gamma = adjust_gamma(imageCorr, 2.2)
    
    
     """ find bounding boxes for ID-1
     # ==========================================================
     # order of preprocessing:
     #   calibrate camera & distortion
     #   gamma
     #   blur before gray
     #   gray blurred image
     #   edge gray image
     #
     """ 
     output = gamma.copy()
     
     # process image
     blurred = cv2.GaussianBlur(imageCorr, (3,3), 0)
     cv2.imshow("Blurred", blurred) 
     
     gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
     cv2.imshow("gray", gray)   
 
     edged = cv2.Canny(gray, 50, 200, 10)
     cv2.imshow("Canny", edged)

     """
     # find all contours looking for ID-1
     #
     # Note: contours NOT good for detecting touching coins
     # need segmentation for that.
     """
     (_,contours,_) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
     cmax = max(contours, key = cv2.contourArea)
     rectID = cv2.minAreaRect(cmax)  # find rotated rectangle
        
     pnts = cv2.boxPoints(rectID)
     box = np.int0(pnts)
     cv2.drawContours(output,[box],0,(255,0,0),2)
     logging.info("box ID000")
     logging.info(box)
     logging.info("")
     L , W = getBoxDim(rectID)
     
     """Use Eucledian distance measurement & watershed to find coins
     """
     
     # close small gaps
     kernel = np.ones((3,3),np.uint8)
     opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
     kernel2 = np.ones((9,9),np.uint8)
     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
     
     ret2, thresh = cv2.threshold(opening,40, 255, cv2.THRESH_BINARY) # | cv2.THRESH_OTSU)
     cv2.imshow("thresh",thresh)
     
     # compute the exact Euclidean distance from every binary
     # pixel to the nearest zero pixel, then find peaks in this distance map
     D = ndimage.distance_transform_edt(thresh)
     localMax = peak_local_max(D, indices=False, min_distance= 18, labels=thresh) # 35 too big
     
     #debug
     Q = 2*D + 50
     Q = Q.astype(np.uint8)
     cv2.imshow("distance",Q)
     
     
     # perform a connected component analysis on the local peaks,
     # using 8-connectivity, then appy the Watershed algorithm
     markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
     labels = watershed(-D, markers, mask=thresh)
     print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

     #debug
     LB = 10*labels
     LB = LB.astype(np.uint8)
     cv2.imshow("labels",LB)


     # loop over the unique labels returned by the Watershed algorithm
     for label in np.unique(labels):
	     # if the label is zero, we are examining the 'background'
	     # so simply ignore it
	     if label == 0:
		     continue
         

	     # otherwise, allocate memory for the label region and draw it on the mask
	     mask = np.zeros(gray.shape, dtype="uint8")
	     mask[labels == label] = 255

	     # detect contours in the mask and grab the largest one
	     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	     c = max(cnts, key=cv2.contourArea)

	     # draw a circle enclosing the object
	     ((x, y), r) = cv2.minEnclosingCircle(c)
	     cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 2)
	     cv2.putText(output, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	     cv2.imshow("contour",mask)

     # show the output image
     cv2.imshow("Output", output)
      
     cv2.waitKey(0)  
     break

     """ check coin diamters using watershed circles
     """
     

    

     """ draw ALL circles on result then check validity of coins
     """

     coins = 0
     item = 0
     logging.info("item # aspect_ratio:")
     for contour in contours:
        rect = cv2.boundingRect(contour)
        aspect_ratio = float(rect[2]/rect[3])
        if ((aspect_ratio > 0.925) and (aspect_ratio < 1.081)):
            color = (0,255,0)
            
            coins = coins + 1
        else:
            color = (0,0,255)
            
        ## add code to check absolute size using L and W   
        ## dia (mm)
        ## penny:       19.05
        ## nickel:      21.21
        ## dime:        17.91
        ## quarter:     24.26
        ## fifty-cent:  30.61
        ## ID-1:        (85.6 x 53.98)
            
        logging.info([item, aspect_ratio])
        item = item + 1

     amount = 0.0
     cv2.putText(output, "coin count: {}".format(coins), (50, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
     cv2.putText(output, "$: {}".format(amount), (50, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


     cv2.imshow("output", output)  
     # find coin sizes using ID-1 reference
     # ==========================================================
  
  
  
     # analyze each coin for colorq
     # ==========================================================
     
     
     
     # clean up and exit on waitKey
     # ==========================================================
     Wait = True
     if VIDEO==1 & (cv2.waitKey(10) & 0xFF == ord('q')):
       logging.info("exit waitKey")
       Wait = False

     if VIDEO == 0:
        while True:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                logging.info("exit 'q' key")
                cv2.destroyAllWindows()
                Wait = False
                break
     else:
        cv2.waitKey(500)  # frame rate
                
     if not Wait:
         break
     
  # save image to disc (use for project report)
  file_path = ".\Results/"
  cv2.imwrite(file_path + "original" + ".png", image) 
  cv2.imwrite(file_path + "calibrated" + ".png", imageCorr) 
  cv2.imwrite(file_path + "gamma" + ".png", gamma)
  cv2.imwrite(file_path + "gray" + ".png", gray)
  cv2.imwrite(file_path + "blur" + ".png", blurred)
  cv2.imwrite(file_path + "edged" + ".png", edged)
  cv2.imwrite(file_path + "results" + ".png", output)      

  if VIDEO:
     cap.release()
     
  cv2.destroyAllWindows()
  