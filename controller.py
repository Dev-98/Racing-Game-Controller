import cv2, imutils
import numpy as np
from keyboardkeys import PressKey, A, D, Space, ReleaseKey
from imutils.video import VideoStream


cam = VideoStream(src=0).start()

currentkey = list()

while True:
    key = False
    height = 500
    width = 640

    # Image flip
    img = cam.read()
    img = np.flip(img,axis = 1)
    # TO let the cv2 read the image we'll have to convert it to an array
    img = np.array(img)

    # Resizing and recolouring of image
    img = imutils.resize(img, width,height)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


    # Blur and masking of image to remove noise from image
    blur = cv2.GaussianBlur(hsv,(11,11),0)
    colorLower = np.array([58, 56, 209])
    colorUpper = np.array([180,255,255])
    mask = cv2.inRange(blur,colorLower,colorUpper)

    # Morphing is basically applying some transformation on images
    # Check out this link for more information : https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)) # morph_open is just another name of "erosion followed by dilation"
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))# close is reverse of opening "dilation followed by erosion"
    # cv2.imshow("morph_open",mask)


    upContour = mask[0:height//2,0:width]
    downContour = mask[3*height//4:height,2*width//5:3*width//5]

    countour_up = cv2.findContours(upContour,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    countour_up = imutils.grab_contours(countour_up)

    countour_down = cv2.findContours(downContour,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    countour_down = imutils.grab_contours(countour_down)

    if len(countour_up) > 0:

        c = max(countour_up,key = cv2.contourArea)
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])

        if cX < (width//2 - 55):
            PressKey(A)
            key = True
            currentkey.append(A)

        elif cX > (width//2 + 55):
            PressKey(D)
            key = True
            currentkey.append(D)

    if len(countour_down) > 0:
        PressKey(Space)
        key = True
        currentkey.append(Space)

    img = cv2.rectangle(img, (0,0),(width//2-55,height//2),(0,255,0),1)
    cv2.putText(img,'LEFT',(110,30),cv2.FONT_HERSHEY_DUPLEX, 1, (139,0,0))

    img = cv2.rectangle(img, (width//2+55,0),(width,height//2),(0,255,0),1)
    cv2.putText(img,'RIGHT',(440,30),cv2.FONT_HERSHEY_DUPLEX,  1, (139,0,0))

    img = cv2.rectangle(img, (2*(width//5),3*height//4),(3*width//5,height),(0,255,0),1)
    cv2.putText(img,'NITRO',(2*(width//5) + 20,height-10),cv2.FONT_HERSHEY_DUPLEX,  1, (139,0,0))


    cv2.imshow("CLOSE",img)

    if not key and len(currentkey)!=0:
        for current in currentkey:
            ReleaseKey(current)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # print("Height = ",str(height),"\n Width = "+str(width))
        break 

cv2.destroyAllWindows()