import numpy as np
import cv2
import sys
import os

bg_path = "c:/code/python/squashtracker/test5_bg.png"
frame1_path = "c:/code/python/squashtracker/test5_frames/test5_frame1.png"
video_path = os.path.join('c:/code/python/squashtracker/test5_pt1.avi')

#Set Paths
bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
bg_gray = cv2.imread("test5_bg_gray.png", cv2.COLOR_BGR2GRAY)
bg_blur = cv2.GaussianBlur(bg_gray,(5,5),0)

frame1 = cv2.imread(frame1_path, cv2.IMREAD_COLOR)
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg_blur = cv2.GaussianBlur(bg_gray,(5,5),0)

hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

#Calculate Histogram
diff = cv2.absdiff(gray1,bg_gray)    
mask = cv2.threshold(diff,20,255,cv2.THRESH_BINARY)[1]
#mask = cv2.inRange(diff, (25,25,25), (255,255,255))
mask = cv2.dilate(mask, rect)
mask = cv2.dilate(mask, rect)
mask = cv2.erode(mask, rect)
mask = cv2.erode(mask, rect)

fg = cv2.bitwise_and(frame1,frame1,mask=mask)   

_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(cnt) for cnt in contours]
players = np.argpartition(areas,-2)[-2:]
red_contour = contours[players[0]]
blue_contour = contours[players[1]]
    
#histogram of contour1
red_mask = np.zeros(gray1.shape,np.uint8)
blue_mask = np.zeros(gray1.shape,np.uint8)
cv2.drawContours(red_mask,[red_contour],0,(255,255,255),-1)
cv2.drawContours(blue_mask,[blue_contour],0,(255,255,255),-1)

red_hist = cv2.calcHist([hsv1],[0,1,2],red_mask,[45,32,32],[0,180,0,256,0,256])
blue_hist = cv2.calcHist([hsv1],[0,1,2],blue_mask,[45,32,32],[0,180,0,256,0,256])      

######################     
      
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit('Could not open ' + video_path)

results = open('c:/code/python/squashtracker/test5_pt1_results.txt', 'w')
results.write("frame,player1x,player1y,player2x,player2y,ballx,bally\n")  
    
cv2.namedWindow('Squash Tracker')

framenum=60
while(cap.isOpened() and cv2.getWindowProperty('Squash Tracker', 0) >= 0):
   ret, frame = cap.read()    
   if ret==True:
      
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray,(5,5),0)
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      diff = cv2.absdiff(blur,bg_blur)    
      mask = cv2.threshold(diff,20,255,cv2.THRESH_BINARY)[1]
      #mask = cv2.inRange(diff, (25,25,25), (255,255,255))
      mask = cv2.dilate(mask, rect)
      mask = cv2.dilate(mask, rect)
      mask = cv2.erode(mask, rect)
      mask = cv2.erode(mask, rect)
      
      fg = cv2.bitwise_and(frame,frame,mask=mask)   
      newframe = frame.copy()
      
      plx = -1
      ply = -1
      p2x = -1
      p2y = -1
      bx = -1
      by = -1
      
      _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      if len(contours) >= 3:
          areas = [cv2.contourArea(cnt) for cnt in contours]
          players = np.argpartition(areas,-2)[-2:]       
          contour1 = contours[players[0]]
          contour2 = contours[players[1]]
          contour1_mask = np.zeros(gray1.shape,np.uint8)
          contour2_mask = np.zeros(gray1.shape,np.uint8)
          cv2.drawContours(contour1_mask,[contour1],0,(255,255,255),-1)
          cv2.drawContours(contour2_mask,[contour2],0,(255,255,255),-1)
          contour1_hist = cv2.calcHist([hsv],[0,1,2],contour1_mask,[45,32,32],[0,180,0,256,0,256])
          contour2_hist = cv2.calcHist([hsv],[0,1,2],contour2_mask,[45,32,32],[0,180,0,256,0,256])
          
          red1 = cv2.compareHist(contour1_hist, red_hist, cv2.HISTCMP_CORREL )
          blue1 = cv2.compareHist(contour1_hist, blue_hist, cv2.HISTCMP_CORREL )
          red2 = cv2.compareHist(contour2_hist, red_hist, cv2.HISTCMP_CORREL )
          blue2 = cv2.compareHist(contour2_hist, blue_hist, cv2.HISTCMP_CORREL )
          
          if (red1 + blue2 >= red2 + blue1):
              player1 = contour1.copy()
              player2 = contour2.copy()
          else:
              player1 = contour2.copy()
              player2 = contour1.copy()
              
          cv2.drawContours(newframe, [player1], 0, (255,0,0), 3)
          cv2.drawContours(newframe, [player2], 0, (0,255,255), 3)
          
          moments1 = cv2.moments(player1)
          moments2 = cv2.moments(player2)
          if (moments1['m00'] > 0):
              p1x = int(moments1['m10']/moments1['m00'])
              p1y = int(moments1['m01']/moments1['m00'])
          else:
              p1x=player1[0][0][0]
              p1y=player1[0][0][1]
          if (moments2['m00'] > 0):
              p2x = int(moments2['m10']/moments2['m00'])
              p2y = int(moments2['m01']/moments2['m00'])
          else:
              p2x=player2[0][0][0]
              p2y=player2[0][0][1]
          
          #Ball
          other_contours = [cnt for i,cnt in enumerate(contours) if i not in [players[0],players[1]]]
          #circularity = (4*pi*area)/(perimeter^2)
          numerators = [np.nan_to_num(4*3.14159*cv2.contourArea(cnt)) for cnt in other_contours]
          denominators = [cv2.arcLength(cnt,True)*cv2.arcLength(cnt,True) for cnt in other_contours]
          nonzeroes = [i!=0 for i in denominators]
          roundnesses=np.divide(numerators,denominators,None,where=nonzeroes)
          ball = np.argpartition(roundnesses,-1)[-1:]
          contour3 = other_contours[ball[0]]
          cv2.drawContours(newframe, [contour3], 0, (255,255,255), 3)
          
          moments3 = cv2.moments(contour3)
          if (moments3['m00'] > 0):
              bx = int(moments3['m10']/moments3['m00'])
              by = int(moments3['m01']/moments3['m00'])
          else:
              bx=contour3[0][0][0]
              by=contour3[0][0][1]          
          
          
      #results.write("{},{},{},{},{},{},{}\n".format(framenum,p1x,p1y,p2x,p2y,bx,by)) 
      cv2.putText(newframe,str(framenum),(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)      
      cv2.imshow('Squash Tracker',newframe)
      framenum+=1

   else:
       break
    
   k = cv2.waitKey(1) & 0xFF
   if k == ord('q'):
       break
   #pause
   if k == ord('p'):
       print("Paused")
       if cv2.waitKey(-1) & 0xFF == ord('p'):
           continue
       
 
cap.release()
results.close()
cv2.destroyAllWindows()