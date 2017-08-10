import numpy as np
import cv2
import sys
import os
import math

#frame = cv2.imread("c:/code/python/squashtracker/test5_frames/test5_frame110.png", cv2.IMREAD_COLOR)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray,(5,5),0)
bg = cv2.imread("c:/code/python/squashtracker/test5_bg.png", cv2.IMREAD_COLOR)
bg_gray = cv2.imread("test5_bg_gray.png", cv2.COLOR_BGR2GRAY)
bg_blur = cv2.GaussianBlur(bg_gray,(5,5),0)
rect55 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
rect33 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

#Remove reflective glass
blackout1 = np.array([[638,755],[597,867],[627,1079],[680,1079]], np.int32)
blackout2 = np.array([[1205,764],[1234,863],[1214,1077],[1171,1074]], np.int32)
blackout3 = np.array([[1865,768],[1919,816],[1919,999],[1892,1079],[1762,1079]], np.int32)
blackout4 = np.array([[670,745],[680,745],[710,1079],[700,1079]], np.int32)

path = os.path.join('c:/code/python/squashtracker/test5_pt1.avi')
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    sys.exit('Could not open ' + path)

predictions = open('c:/code/python/squashtracker/test5_pt1_predictions.csv', 'w')
prediction_output_strings = ["framenum","area","roundness","gdiff","model_x","model_y","model_score","model_ball_detected"]
predictions.write(','.join(prediction_output_strings))
predictions.write("\n")

cv2.namedWindow('Squash Tracker')
framenum=1
while(cap.isOpened() and cv2.getWindowProperty('Squash Tracker', 0) >= 0):
    ret, frame = cap.read()
    
    if ret==True:
        height, width, channels = frame.shape 
        fg=frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        diff = cv2.absdiff(blur,bg_blur)    
        mask = np.ones((height,width),np.uint8) * 255
        cv2.fillPoly(mask, [blackout1,blackout2,blackout3,blackout4], 0)

        diff_blur = cv2.absdiff(blur,bg_blur)    
        mask_blur = cv2.threshold(diff_blur,15,255,cv2.THRESH_BINARY)[1]
        mask_blur = cv2.dilate(mask_blur, rect55)
        mask_blur = cv2.dilate(mask_blur, rect55)
        
        #first detect players and remove them
        _, contours, _ = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            if (area > 1000):
                cv2.drawContours(mask,[cnt],-1,(0,0,0), -1)
        
        #detect ball from remaining contours
        gray_masked = cv2.bitwise_and(blur,blur,mask=mask)   
        bg_gray_masked = cv2.bitwise_and(bg_blur,bg_blur,mask=mask)   
        diff = cv2.absdiff(gray_masked,bg_gray_masked)    
        ball_mask = cv2.threshold(diff,15,255,cv2.THRESH_BINARY)[1]
        ball_mask = cv2.dilate(ball_mask, rect55)
        ball_mask = cv2.dilate(ball_mask, rect55)
        ball_mask = cv2.erode(ball_mask, rect55)
        ball_mask = cv2.erode(ball_mask, rect55)
        
        _, contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        model_probs = [0] * num_contours
        xs = [-999] * num_contours
        ys = [-999] * num_contours
        areas = [-999] * num_contours
        roundnesses = [-999] * num_contours
        gdiffs = [-999] * num_contours
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            
            moments = cv2.moments(cnt)
            perimeter = cv2.arcLength(cnt,True)
            if perimeter > 0:
                roundness = (4*3.14159*area)/(perimeter**2)
            else:
                roundness = -1
            radius = math.sqrt(area/3.14159)
            if (moments['m00'] > 0):
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
            else:
                cx=cnt[0][0][0]
                cy=cnt[0][0][1]
                               
            gcol = gray[cy][cx]            
            gcol2 = gray[min(height-1,max(0,cy+int(radius)))][min(width-1,max(0,cx+int(radius)))].astype(int)
            gcol3 = gray[min(height-1,max(0,cy+int(radius)))][min(width-1,max(0,cx-int(radius)))].astype(int)
            gcol4 = gray[min(height-1,max(0,cy-int(radius)))][min(width-1,max(0,cx+int(radius)))].astype(int)
            gcol5 = gray[min(height-1,max(0,cy-int(radius)))][min(width-1,max(0,cx-int(radius)))].astype(int)
         
            gavg = (gcol2 + gcol3 + gcol4 + gcol5) / 4
            gdiff = gcol - gavg
            
            total_effect = 4.05233001 * roundness - 0.04492482 * gdiff - 2.44888384
            model_probs[i] = 1 / (1 + np.exp(-total_effect))
            if (area < 20 or area > 300):
                model_probs[i] = 0
            
            areas[i] = area
            roundnesses[i] = roundness
            gdiffs[i] = gdiff
            xs[i] = cx
            ys[i] = cy
        
        # Compare the scores
        best_x = -999
        best_y = -999        
        best_area = -999
        best_roundness = -999
        best_gdiff = -999
        best_score = 0 
        ball_detected = 0
        if (num_contours > 0):
            best_cnt = np.argmax(model_probs)                
            best_area = areas[best_cnt]
            best_roundness = roundnesses[best_cnt]
            best_gdiff = gdiffs[best_cnt]
            best_score = model_probs[best_cnt]
            if (best_score > 0.5):
                best_x = xs[best_cnt]
                best_y = ys[best_cnt]
                ball_detected = 1
                cv2.drawContours(fg, contours, best_cnt, (0,255,0), 3)   
          
        predictions.write("{},{},{},{},{},{},{},{}\n".format(framenum,best_area,round(best_roundness,2),round(best_gdiff,2),best_x,best_y,round(best_score,2),ball_detected))  
               
        #cv2.imshow('ball_mask',ball_mask)
        cv2.putText(fg,str(framenum),(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('Squash Tracker',fg)
        #print(framenum)
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
       
predictions.close()
cap.release()
cv2.destroyAllWindows()