#import cv2
#import numpy as np

#image = cv2.imread('Jmk2AjJdG3g.jpg')

##kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
##image = cv2.filter2D(image, -1, kernel)

#blur = cv2.GaussianBlur(image, (3, 3), 0)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
#    if cv2.countNonZero(roi) > 0.3 * roi.size:
#        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)

#    roi = mask_red[int(y-r):int(y+r), int(x-r):int(x+r)]
#    if cv2.countNonZero(roi) > 0.3 * roi.size:
#        cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 2)

#    roi = mask_yellow[int(y-r):int(y+r), int(x-r):int(x+r)]
#    if cv2.countNonZero(roi) > 0.3 * roi.size:
#        cv2.circle(image, (int(x), int(y)), int(r), (0, 155, 255), 2)

#cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
#cv2.imshow("Result", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

#def search(mask):
#        # ������� ������� ��������� �������� �� �����
#    indices = np.nonzero(mask)
    
#    # ����������� ������� � ������ numpy
#    pixels = np.array(indices).T
    
#    if len(pixels) > 0:
#        # ��������� �������� DBSCAN ��� ������������� �������� �� �����
#        labels = dbscan.fit_predict(pixels)
      
        ## ������� ������ ������� ��� �������������
        #clusters = {}
        
        ## �������� �� ������� ���������������
        #for i in range(len(labels)):
        #    # ���� ������������� ��� ���� � �������, ��������� ���������� � ������������� �������
        #    if labels[i] != -1:
        #        if labels[i] in clusters:
        #            clusters[labels[i]].append(pixels[i])
        #        # ���� �������������� ��� ��� � �������, ������� ����� ������
        #        else:
        #            clusters[labels[i]] = [pixels[i]]
    
        ## �������� ���������� ��������� � �� �����
        ##n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #colors = np.random.randint(0, 255, size=(len(clusters), 3))
    
        #rectangles = []
        ## �������� �� ���� ���������
        #for cluster_id in clusters:
        #    # �������� ������� ��������, ������������� �������� ��������
        #    #cluster_indices = np.where(labels == i)
        #    coord_array = np.array(clusters[cluster_id], dtype=np.int32)
        #    # �������� ���������� x � y ���� ��������
        #    x = coord_array[:, 1]
        #    y = coord_array[:, 0]
    
        #    # ������� ����������� � ������������ �������� x � y
        #    x_min = np.min(x)
        #    x_max = np.max(x)
        #    y_min = np.min(y)
        #    y_max = np.max(y)
        #    rectMax=200
        #    rectMin=25
        #    delta_x=x_max-x_min
        #    delta_y=y_max-y_min
        #    if delta_x >rectMin and delta_x <rectMax and delta_y >rectMin and delta_y <rectMax:
        #        # ��������� ���������� �������������� � ������
        #        rectangles.append((x_min, y_min, x_max, y_max))
        #return rectangles
        #for x1, y1, x2, y2 in rectangles:
        #    # ������ ������������� �� �������� ����� � ����� ������ � �������� 2 �������
        #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
        ## ������� ������ ������ ��� �������� ��������������� �����������
        ##result = np.nonzero(frame)
    
        ## �������� �� ���� �������� � �� ������
        #for pixel, label in zip(pixels, labels):
        #    # ���� ����� �� ����� -1 (�� ���� �� ���)
        #    if label != -1:
        #        # �������� ���� �������� �� �����
        #        color = colors[label]
    
        #        # ���������� ���� �������� � ��������������� ������� ��������������� �����������
        #        frame[pixel[0], pixel[1]] = color

def find_convex_contours(mask, min_area=100, max_area=500):
    ## �������������� ����������� � ������� ������
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ## ����������� �����������
    #ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # ����� �������� �� �������������� �����������
    contours, hierarchy = cv2.findContours (mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # ����� �������� ��������
    convex_contours = []
    for contour in contours:
        # ���������� ������� �������
        area = cv2.contourArea(contour)
        if area > min_area and area <max_area:
            hull = cv2.convexHull(contour)
            convex_contours.append(hull)
    return convex_contours

rectMax=200
rectMin=25

# ������� ������ ������� ������ � ������
cap = cv2.VideoCapture(0)
dbscan = DBSCAN(eps=10, min_samples=200,algorithm="kd_tree")
while True:
    # ����������� ���� � ������
    ret, frame = cap.read()
    frame =cv2.resize(frame,[640,480])
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

     # ���������, ��� ���� ������� ��������
    if ret:
        # ��������� ����������� � �������� ������������ HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ������ �������� ������, ������� ����� ��������
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
        lower_blue = (110, 50, 50)
        upper_blue = (130, 255, 255)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)


        convex_contours = find_convex_contours(mask_green, min_area=500,max_area=5000)
        cv2.drawContours(frame, convex_contours, -1, (0, 255, 0), 3)
        convex_contours = find_convex_contours(mask_red, min_area=500,max_area=5000)
        cv2.drawContours(frame, convex_contours, -1, (255, 0, 0), 3)
        convex_contours = find_convex_contours(mask_yellow, min_area=500,max_area=5000)
        cv2.drawContours(frame, convex_contours, -1, (0, 255, 255), 3)
        convex_contours = find_convex_contours(mask_blue, min_area=500,max_area=5000)
        cv2.drawContours(frame, convex_contours, -1, (0, 255, 0), 3)

        # ���������� ����������� � ������
        cv2.imshow('frame', frame)
        cv2.imshow('mask_blue', mask_blue)
        cv2.imshow('mask_yellow', mask_yellow)
        cv2.imshow('mask_red', mask_red)
        cv2.imshow('mask_green', mask_green)
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


