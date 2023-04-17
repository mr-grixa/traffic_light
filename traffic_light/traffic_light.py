from asyncio.windows_events import NULL
from pickle import FALSE
import cv2
import os
import numpy as np
from mss import mss
from sklearn.cluster import DBSCAN

def find_convex_contours(mask, rectMin=25, rectMax=200,indent=7):
    contours, hierarchy = cv2.findContours (mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    # поиск выпуклых контуров
    rectangles = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        #convex_contours.append(hull)
        x, y, w, h = cv2.boundingRect(hull)
        if w >rectMin and w <rectMax and h >rectMin and h <rectMax:
            rectangles.append((max(0, x-indent),max(0, y-indent),x + w+indent, y + h+indent))
    return rectangles




print("Start")
# Создаем объект захвата кадров с камеры
#cap = cv2.VideoCapture(1)
#dbscan = DBSCAN(eps=10, min_samples=200,algorithm="kd_tree")
sct = mss()
monitor = {"top": 0, "left": 0, "width": 1280, "height": 960}

orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, 
                     firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31)

dir_path = os.path.dirname(os.path.realpath(__file__))
folder_path = dir_path+'/signs/red'
sign_red = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        sign=cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        mask = sign[:, :, 3]
        kp, des = orb.detectAndCompute(sign, mask)
        sign_red.append((file_name.replace(".png", ""),kp,des))
        print(file_name)
folder_path = dir_path+'/signs/blue'
sign_blue = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        sign=cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        mask = sign[:, :, 3]
        kp, des = orb.detectAndCompute(sign, mask)
        sign_red.append((file_name.replace(".png", ""),kp,des))
        print(file_name)



rectMax=200
rectMin=25

while True:
    # Захватываем кадр с камеры
    #ret, frame = cap.read()
    frame = sct.grab(monitor)
    frame = np.array(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #canny = cv2.Canny(frame,100,200)
    #frame =cv2.resize(frame,[640,480])
    #frame = cv2.GaussianBlur(frame, (3,3), 0)
    prop = cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE)
    if prop < 0:
        break
     # Проверяем, что кадр успешно прочитан
    if 1: #ret: 
        # Переводим изображение в цветовое пространство HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Задаем диапазон цветов, который хотим выделить
        lower_green = (40, 50, 50)
        upper_green = (80, 255, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        lower_red = (0, 150, 100)
        upper_red = (10, 255, 255)
        mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = (170, 150, 100)
        upper_red = (180, 255, 255)
        mask_red2 = cv2.inRange(hsv, lower_red, upper_red)
        mask_red = mask_red1 + mask_red2
        lower_yellow = (20, 50, 50)
        upper_yellow = (40, 255, 255)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        lower_blue = (100, 50, 50)
        upper_blue = (140, 255, 255)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        rectangles = find_convex_contours(mask_red, rectMin,rectMax)
        for x1, y1, x2, y2 in rectangles:
            roi = frame[y1:y2, x1:x2]

            kp, des = orb.detectAndCompute(roi, None)
            matches = {}
            if des is not None and kp is not NULL:
                for file_name,kpR,desR in sign_red:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches[file_name] = bf.match(desR, des)
            # Находим наиболее подходящий объект на изображении
            best_match = None
            distCoef = 100
            best_match_object = None
            for object_name, match in matches.items():
                good_matches = []
                for m in match:
                    #print(f"Для обьекта {object_name} совпадение равняется {m.distance}")
                    if m.distance < 58:
                        good_matches.append(m)
                if len(good_matches) > 10 and (best_match is None or len(good_matches) > len(best_match)):
                    best_match = good_matches
                    best_match_object = object_name
                    distCoef = m.distance

            roi = cv2.drawKeypoints(roi, kp, None)
            frame[y1:y2, x1:x2] =roi

            roiGray=mask_red[y1:y2, x1:x2]
            circles = cv2.HoughCircles(roiGray, cv2.HOUGH_GRADIENT, 1, 10, param1=80, param2=80, minRadius=5, maxRadius=100)
            if circles is not None:
                for (x, y, r)  in circles[0]:
                    circle_area = np.pi * r**2
                    circle_mask = np.zeros(roiGray.shape, dtype=np.uint8)
                    cv2.circle(circle_mask, (x, y),r, 255, -1)
                    white_pixels = np.sum(circle_mask == 255)
                    fill_ratio = white_pixels / circle_area
                    if fill_ratio >0.75:
                        cv2.circle(roi, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 0, 255), 1)
            
            if best_match_object is not None:
                cv2.putText(frame,best_match_object,(x1, y2),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA,bottomLeftOrigin=False)
        rectangles = find_convex_contours(mask_green, rectMin,rectMax)
        for x1, y1, x2, y2 in rectangles:
            roiGray=mask_green[y1:y2, x1:x2]
            circles = cv2.HoughCircles(roiGray, cv2.HOUGH_GRADIENT, 1, 10, param1=80, param2=80, minRadius=5, maxRadius=100)
            if circles is not None:
                for (x, y, r)  in circles[0]:
                    circle_area = np.pi * r**2
                    circle_mask = np.zeros(roiGray.shape, dtype=np.uint8)
                    cv2.circle(circle_mask, (x, y),r, 255, -1)
                    white_pixels = np.sum(circle_mask == 255)
                    fill_ratio = white_pixels / circle_area
                    if fill_ratio >0.75:
                        cv2.circle(roi, (int(x), int(y)), int(r), (0, 255, 0), 2)    
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 1)

        rectangles = find_convex_contours(mask_yellow, rectMin,rectMax)
        for x1, y1, x2, y2 in rectangles:
            roiGray=mask_yellow[y1:y2, x1:x2]
            circles = cv2.HoughCircles(roiGray, cv2.HOUGH_GRADIENT, 1, 10, param1=80, param2=80, minRadius=5, maxRadius=100)
            if circles is not None:
                for (x, y, r)  in circles[0]:
                    circle_area = np.pi * r**2
                    circle_mask = np.zeros(roiGray.shape, dtype=np.uint8)
                    cv2.circle(circle_mask, (x, y),r, 255, -1)
                    white_pixels = np.sum(circle_mask == 255)
                    fill_ratio = white_pixels / circle_area
                    if fill_ratio >0.75:
                        cv2.circle(roi, (int(x), int(y)), int(r), (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 255), 1)
            

        rectangles = find_convex_contours(mask_blue, rectMin,rectMax)
        for x1, y1, x2, y2 in rectangles:
            roi = frame[y1:y2, x1:x2]
            kp, des = orb.detectAndCompute(roi, None)
            matches = {}
            if des is not None and kp is not NULL:
                for file_name,kpR,desR in sign_blue:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches[file_name] = bf.match(desR, des)
            # Находим наиболее подходящий объект на изображении
            best_match = None
            distCoef = 100
            best_match_object = None
            for object_name, match in matches.items():
                good_matches = []
                for m in match:
                    #print(f"Для обьекта {object_name} совпадение равняется {m.distance}")
                    if m.distance < 58:
                        good_matches.append(m)
                if len(good_matches) > 10 and (best_match is None or len(good_matches) > len(best_match)):
                    best_match = good_matches
                    best_match_object = object_name
                    distCoef = m.distance

            roi = cv2.drawKeypoints(roi, kp, None)
            frame[y1:y2, x1:x2] =roi
            cv2.rectangle(frame, (x1, y1), (x2,y2), (255, 0, 0), 1)
            if best_match_object is not None:
                cv2.putText(frame,best_match_object,(x1, y2),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2,cv2.LINE_AA,bottomLeftOrigin=False)
              



        # Отображаем изображение с камеры
        cv2.imshow('frame', frame)
        #cv2.imshow('canny', canny)
        cv2.imshow('mask_blue', mask_blue)
        cv2.imshow('mask_yellow', mask_yellow)
        cv2.imshow('mask_red', mask_red)
        cv2.imshow('mask_green', mask_green)
        #height, width, channels = frame.shape

        ## Выводим размеры
        #print(f"width: {width}")
        #print(f"height: {height}")
        #print(f"channels: {channels}")
    # Если пользователь нажимает клавишу 'q', то выходим из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
#cap.release()
#cv2.destroyAllWindows()
sct.close()
cv2.destroyAllWindows()


