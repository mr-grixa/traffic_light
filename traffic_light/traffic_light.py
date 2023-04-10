#import cv2
#image = cv2.imread('dorozhnye-svetofory-0003012342-preview.jpg')

#blur = cv2.GaussianBlur(image, (3, 3), 0)
#gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=80, minRadius=5, maxRadius=100)

#hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#lower_green = (40, 50, 50)
#upper_green = (80, 255, 255)
#mask_green = cv2.inRange(hsv, lower_green, upper_green)
#lower_red = (0, 50, 50)
#upper_red = (10, 255, 255)
#mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
#lower_red = (170, 50, 50)
#upper_red = (180, 255, 255)
#mask_red2 = cv2.inRange(hsv, lower_red, upper_red)
#mask_red = mask_red1 + mask_red2
#lower_yellow = (20, 50, 50)
#upper_yellow = (40, 255, 255)
#mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
#lower_blue = (110, 50, 50)
#upper_blue = (130, 255, 255)
#mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)


#cv2.imshow("mask_green", mask_green)
#cv2.imshow("mask_red", mask_red)
#cv2.imshow("mask_yellow", mask_yellow)
#cv2.imshow("mask_blue", mask_blue)



#for circle in circles[0]:
#    x, y, r = circle
#    roi = mask_green[int(y-r):int(y+r), int(x-r):int(x+r)]
#    cv2.circle(image, (int(x), int(y)), int(r), (100, 100, 100), 2)
#    if cv2.countNonZero(roi) > 0.5 * roi.size:
#        cv2.circle(blur, (int(x), int(y)), int(r), (0, 255, 0), 2)

#    roi = mask_red[int(y-r):int(y+r), int(x-r):int(x+r)]
#    if cv2.countNonZero(roi) > 0.5 * roi.size:
#        cv2.circle(blur, (int(x), int(y)), int(r), (0, 0, 255), 2)

#    roi = mask_yellow[int(y-r):int(y+r), int(x-r):int(x+r)]
#    if cv2.countNonZero(roi) > 0.5 * roi.size:
#        cv2.circle(blur, (int(x), int(y)), int(r), (0, 155, 255), 2)

#cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
#cv2.imshow("Result", blur)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

rectMax=200
rectMin=25

# ������� ������ ������� ������ � ������
cap = cv2.VideoCapture(0)
dbscan = DBSCAN(eps=10, min_samples=200)
while True:
    # ����������� ���� � ������
    ret, frame = cap.read()
    #frame = cv2.GaussianBlur(frame, (3, 3), 0)

     # ���������, ��� ���� ������� ��������
    if ret:
        # ��������� ����������� � �������� ������������ HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ������ �������� ������, ������� ����� ��������
        lower_green = (40, 50, 50)
        upper_green = (80, 255, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        lower_red = (0, 50, 50)
        upper_red = (10, 255, 255)
        mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = (170, 50, 50)
        upper_red = (180, 255, 255)
        mask_red2 = cv2.inRange(hsv, lower_red, upper_red)
        mask_red = mask_red1 + mask_red2
        lower_yellow = (20, 50, 50)
        upper_yellow = (40, 255, 255)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        lower_blue = (110, 50, 50)
        upper_blue = (130, 255, 255)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # ������� ������� ��������� �������� �� �����
        indices = np.nonzero(mask_blue)

        # ����������� ������� � ������ numpy
        pixels = np.array(indices).T

        if len(pixels) > 0:
            # ��������� �������� DBSCAN ��� ������������� �������� �� �����
            labels = dbscan.fit_predict(pixels)
          
            # ������� ������ ������� ��� �������������
            clusters = {}
            
            # �������� �� ������� ���������������
            for i in range(len(labels)):
                # ���� ������������� ��� ���� � �������, ��������� ���������� � ������������� �������
                if labels[i] != -1:
                    if labels[i] in clusters:
                        clusters[labels[i]].append(pixels[i])
                    # ���� �������������� ��� ��� � �������, ������� ����� ������
                    else:
                        clusters[labels[i]] = [pixels[i]]

            # �������� ���������� ��������� � �� �����
            #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            colors = np.random.randint(0, 255, size=(len(clusters), 3))

            rectangles = []
            # �������� �� ���� ���������
            for cluster_id in clusters:
                # �������� ������� ��������, ������������� �������� ��������
                #cluster_indices = np.where(labels == i)

                coord_array = np.array(clusters[cluster_id], dtype=np.int32)
                # �������� ���������� x � y ���� ��������
                x = coord_array[:, 1]
                y = coord_array[:, 0]

                # ������� ����������� � ������������ �������� x � y
                x_min = np.min(x)
                x_max = np.max(x)
                y_min = np.min(y)
                y_max = np.max(y)
                rectMax=200
                rectMin=25
                delta_x=x_max-x_min
                delta_y=y_max-y_min
                if delta_x >rectMin and delta_x <rectMax and delta_y >rectMin and delta_y <rectMax:
                    # ��������� ���������� �������������� � ������
                    rectangles.append((x_min, y_min, x_max, y_max))

            for x1, y1, x2, y2 in rectangles:
                # ������ ������������� �� �������� ����� � ����� ������ � �������� 2 �������
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
            # ������� ������ ������ ��� �������� ��������������� �����������
            #result = np.nonzero(frame)

            # �������� �� ���� �������� � �� ������
            for pixel, label in zip(pixels, labels):
                # ���� ����� �� ����� -1 (�� ���� �� ���)
                if label != -1:
                    # �������� ���� �������� �� �����
                    color = colors[label]

                    # ���������� ���� �������� � ��������������� ������� ��������������� �����������
                    frame[pixel[0], pixel[1]] = color


        # ���������� ����������� � ������
        cv2.imshow('frame', frame)
        cv2.imshow('mask_blue', mask_blue)
        #height, width, channels = frame.shape

        ## ������� �������
        #print(f"width: {width}")
        #print(f"height: {height}")
        #print(f"channels: {channels}")

    # ���� ������������ �������� ������� 'q', �� ������� �� �����
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ����������� �������
cap.release()
cv2.destroyAllWindows()
