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
    using *** Hough transform ***
    
    weakness:
       1. Need very black background and light (high contrast) ID reference
       2. Errors when coins touch, contours cover multiple coins.  
           -- contours used to find non-circular objects   
"""

import cv2
import numpy as np
import logging, sys
import yaml

## =========================================================================
# select video & camera and whether to use calibration
VIDEO = 0
camera = 1
CAL = True
  # Beaware that camera calibration can move some objects out of the frame
  # and as a result contours will be incomplete and return a very small area
  #
## ==========================================================================

#qIDref = 2.00*25.4
# ID-1 85.60 x 53.98mm
IDrefW = 53.98
IDrefL = 85.60

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def getBoxDim(box1):
    # return length and width of rotated rect
    L = box1[1][1]
    W = box1[1][0]
    
    return max(L,W), min(L,W)

def checkCoinDia(rc, Wp, Lp, cent):
    # check if radius is a valid coin
    ## absolute size using L and W, dia (mm)
    ##
    ## dime:        17.91
    ## penny:       19.05
    ## nickel:      21.21
    ## quarter:     24.26
    ## fifty-cent:  30.61
    ## ID-1:        (85.6 x 53.98)
    ## business crd ( 3.5*25.4 x 2.0*25.4 )
    dia = rc * 2.0 * (IDrefW) / Wp
    
    # check coin dia in mm and color flag, penny == True
    result = -1.0
    if dia< 16.2:
        result = -1.0
    elif dia < 18.48:
        if penny == True:
           result = 1.0
        else:
            result = 10.0
    elif dia < 20.13:
        if penny == True:
           result = 1.0
        else:
           result = 10.0
    elif dia < 22.8:
        if penny == True:
            result = 1.0
        else:
            result = 5.0
    elif dia < 27.43:
        result = 25.0
    elif dia < 32.0:
        result = 50.0
   
    return result, dia

def checkCenter(img, cir, thresh = [30,15]):
    # Check if center (x,y) is greater than bkgd threshold
    cir1 = cir.copy()
    radius = np.int32(6)
    # overwrite radius
    cir1[2] = radius
    metric, _ = getCircleColor(img, cir1, 'hsv')

    #yg = (thresh[1]/(thresh[0]+1)) * metric[0] + 10
    #val = metric[0]
      
    if metric[0] < 0.4*thresh[0]:
        return True, metric
    else:
        return False, metric
    

def getCircleColor(image, circ, flag = 'hsv'):
    # return mean of colors over circle from RGB input image
    if flag == 'hsv':
        color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif flag == 'lab':
        color = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
    else:
        color = image
    # https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image
    circ = np.uint16(np.around(circ))
    mask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)   
    cv2.circle(mask, (circ[0], circ[1]), circ[2], (255, 255, 255), -1)
    
    metric = cv2.mean(color, mask)
    
    return metric, mask


def getDimePennyDecision(img, circ):
    # use hsv s-v space to discern dime from pennies
    metric, _ = getCircleColor(img, circ, flag = 'hsv')
    h = metric[0]
    s = metric[1]
    
    penny = True
    if (1.10 * h -s + 38 > 0):
    #if (6.4 * h - s - 75 > 0):
        penny = False
           
    return penny
 
def  findRangeAllContours(contours):
     # find x-y range of ALL objects & ID card
     xmin = 9999
     ymin = 9999
     xmax = 0
     ymax = 0
     
     for c in contours:       
         extLeft = tuple(c[c[:, :, 0].argmin()][0])
         extRight = tuple(c[c[:, :, 0].argmax()][0])
         extTop = tuple(c[c[:, :, 1].argmin()][0])
         extBot = tuple(c[c[:, :, 1].argmax()][0])
         
         if extLeft[0] < xmin:
             xmin = extLeft[0]
         if extRight[0] > xmax:
             xmax = extRight[0]
         if extTop[1] < ymin:
             ymin = extTop[1]
         if extBot[1] > ymax:
             ymax = extBot[1]
            
     rect = [(xmin,ymin), (xmax,ymax)]
    
     return rect
 
def  getBkgdMetric(contour, flag = 'hsv'):
     # find color of background
     #
     rectRange = findRangeAllContours(contour)
     x1 = int(rectRange[0][0]/2)
     y1 = int(rectRange[0][1]/2)
     r = int(0.5 * np.sqrt(x1**2 + y1**2))
     circ = [x1, y1, min(r, 15)]
     metric_bkgd, mask_bkgd =  getCircleColor(blurred, circ, 'hsv')
     
     return metric_bkgd, rectRange
     
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
     im = cv2.imread('..\images/new_blk.jpg')
     #im = cv2.imread('..\images/compare_blk.jpg')
     im = cv2.imread('..\images/test_final2.jpg')
     im = cv2.imread('..\images/non-touch.jpg')
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
     finalHeight = 640
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

     """ processing pipeline:
     # ==========================================================
     #   calibrate camera & distortion
     #   gamma
     #   blur before gray
     #   gray blurred image
     #   edge gray image
     #
     """     
     output = gamma.copy()
     
     # process image
     blurred = cv2.GaussianBlur(gamma, (3,3), 0)
     cv2.imshow("Blurred", blurred) 
     
     gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
     cv2.imshow("gray", gray)
   
     edged = cv2.Canny(gray, 50, 200, 10) # 50,200
     cv2.imshow("Canny", edged)

     """ find bounding boxes for coins and ID-1
     # ===========================================================
     # find all contours:
     # Pick out largest area as the ID-1 reference
     # use cv2.minAreaRect(cnt) on largest bounding box
     # standard ID-1 as reference (bank card or ID card)
     #
     # Note: contours NOT good for detecting touching coins
     # need segmentation for that.
     """
     (_,contours,_) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     
     ## debug --- draw contours
     #item1 = 0
     #for circ in contours:
        #cv2.drawContours(output, circ, -1, (0,0,255), 3)
        #item1 = item1+1
        #cv2.imshow('contours',output)
        #cv2.waitKey(2000)
        

     """ get background color
     """
     metric_bkgd, error = getBkgdMetric(contours, 'hsv')
     
     #print((metric_bkgd))
     
     if error[0][0]==0 or error[1][0]==image.shape[1]:
         print("ERROR: clipping image in x")
     if error[0][1]==0 or error[1][1]==image.shape[1]:
         print("ERROR clipping image in y")
 
     cmax = max(contours, key = cv2.contourArea)
     rectID = cv2.minAreaRect(cmax)  # find rotated rectangle
        
     pnts = cv2.boxPoints(rectID)
     box = np.int0(pnts)
     cv2.drawContours(output,[box],0,(255,0,0),2)
     logging.info("box ID1")
     logging.info(box)
     logging.info("")
            
     """ find coins and draw circle & bounding rectange using HughCircles
     # ==================================================================
     """
     HIGH = 175 # param1
     LOW = 45   # param2
     circles1 = 25  # set max upper limit for coins
     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 20, circles1, \
          param1=HIGH, param2=LOW, minRadius=15,maxRadius=40) #12,45

     # ensure circles exist
     if circles is not None:
       # convert the (x, y) coordinates and radius of the circles to integers
       circlesInt = np.round(circles[0, :]).astype("int")
    
     """ loop over the (x, y) coordinates and radius of the Hough circles and
         use float for precision 
     """
     logging.info("ref")
     L , W = getBoxDim(rectID)
     logging.info([L, W])
     logging.info("circles:")
     
     coins = 0
     item = 0
     amount = 0.0
     for circ in circlesInt:
        #logging.info([item, circles[item]])
        x = circles[0][item][0]
        y = circles[0][item][1]
        r = circles[0][item][2]
        
        # check range of x and y within imageCorr
        # --- do it here
        
        penny = getDimePennyDecision(blurred, circ)
        center, val2 = checkCenter(blurred, circ, thresh = metric_bkgd)
        
        #print([center, np.around(val2), np.around(metric_bkgd)])
        print(val2)
        
        value, dia = checkCoinDia(r, W, L, penny)        
        metric, mask_c =  getCircleColor(blurred, circ, 'hsv')

        
        logging.info([item, int(10*dia)/10, value, np.around(metric), penny, center, val2])      
        #print([item, int(10*dia)/10, value, np.around(metric), penny, center, np.around(val2)]) 
        
        if value < 0 or not center:
            color = (0,0,255)
            #cv2.putText(output, "{}".format('X'), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
           coins = coins + 1
           amount = amount + value
           color = (0,255,0)
        # draw the circle in the output image, then print the circle #
        # corresponding to the center of the circle
        cv2.circle(output, (int(x), int(y)), int(r), color, 2)
        cv2.putText(output, "{}".format(int(10*dia)/10), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)    
        # cv2.putText(output, "{}".format(int(metric[ch])), (int(x) - 10, int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1) 
        cv2.putText(output, "{}".format(value), (int(x) - 10, int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        item = item + 1
        
     logging.info("---")

     cv2.putText(output, "coin count: {}".format(coins), (30, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
     cv2.putText(output, "$: {}".format(amount/100.0), (30, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

     cv2.imshow("output", output)  
  
     # analyze each coin for color ? if needed ...
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
        cv2.waitKey(200)  # frame rate
                
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
  