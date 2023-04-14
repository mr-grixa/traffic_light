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
#        # Находим индексы ненулевых пикселей на маске
#    indices = np.nonzero(mask)
    
#    # Преобразуем индексы в массив numpy
#    pixels = np.array(indices).T
    
#    if len(pixels) > 0:
#        # Применяем алгоритм DBSCAN для кластеризации пикселей на маске
#        labels = dbscan.fit_predict(pixels)
      
        ## Создаем пустой словарь для сопоставления
        #clusters = {}
        
        ## Проходим по массиву идентификаторов
        #for i in range(len(labels)):
        #    # Если идентификатор уже есть в словаре, добавляем координаты к существующему массиву
        #    if labels[i] != -1:
        #        if labels[i] in clusters:
        #            clusters[labels[i]].append(pixels[i])
        #        # Если идентификатора еще нет в словаре, создаем новую запись
        #        else:
        #            clusters[labels[i]] = [pixels[i]]
    
        ## Получаем количество кластеров и их цвета
        ##n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #colors = np.random.randint(0, 255, size=(len(clusters), 3))
    
        #rectangles = []
        ## Проходим по всем кластерам
        #for cluster_id in clusters:
        #    # Получаем индексы пикселей, принадлежащих текущему кластеру
        #    #cluster_indices = np.where(labels == i)
        #    coord_array = np.array(clusters[cluster_id], dtype=np.int32)
        #    # Получаем координаты x и y этих пикселей
        #    x = coord_array[:, 1]
        #    y = coord_array[:, 0]
    
        #    # Находим минимальные и максимальные значения x и y
        #    x_min = np.min(x)
        #    x_max = np.max(x)
        #    y_min = np.min(y)
        #    y_max = np.max(y)
        #    rectMax=200
        #    rectMin=25
        #    delta_x=x_max-x_min
        #    delta_y=y_max-y_min
        #    if delta_x >rectMin and delta_x <rectMax and delta_y >rectMin and delta_y <rectMax:
        #        # Добавляем координаты прямоугольника в список
        #        rectangles.append((x_min, y_min, x_max, y_max))
        #return rectangles
        #for x1, y1, x2, y2 in rectangles:
        #    # Рисуем прямоугольник на исходном кадре с белым цветом и толщиной 2 пикселя
        #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
        ## Создаем пустой массив для хранения результирующего изображения
        ##result = np.nonzero(frame)
    
        ## Проходим по всем пикселям и их меткам
        #for pixel, label in zip(pixels, labels):
        #    # Если метка не равна -1 (то есть не шум)
        #    if label != -1:
        #        # Получаем цвет кластера по метке
        #        color = colors[label]
    
        #        # Записываем цвет кластера в соответствующий пиксель результирующего изображения
        #        frame[pixel[0], pixel[1]] = color

def find_convex_contours(mask, min_area=100, max_area=500):
    ## преобразование изображения в оттенки серого
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ## бинаризация изображения
    #ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # поиск контуров на бинаризованном изображении
    contours, hierarchy = cv2.findContours (mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # поиск выпуклых контуров
    convex_contours = []
    for contour in contours:
        # вычисление площади контура
        area = cv2.contourArea(contour)
        if area > min_area and area <max_area:
            hull = cv2.convexHull(contour)
            convex_contours.append(hull)
    return convex_contours

rectMax=200
rectMin=25

# Создаем объект захвата кадров с камеры
cap = cv2.VideoCapture(0)
dbscan = DBSCAN(eps=10, min_samples=200,algorithm="kd_tree")
while True:
    # Захватываем кадр с камеры
    ret, frame = cap.read()
    frame =cv2.resize(frame,[640,480])
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

     # Проверяем, что кадр успешно прочитан
    if ret:
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

        # Отображаем изображение с камеры
        cv2.imshow('frame', frame)
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
cap.release()
cv2.destroyAllWindows()


