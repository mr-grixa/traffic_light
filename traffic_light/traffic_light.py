import cv2
image = cv2.imread('dorozhnye-svetofory-0003012342-preview.jpg')

blur = cv2.GaussianBlur(image, (3, 3), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=80, minRadius=5, maxRadius=100)

hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
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


cv2.imshow("mask_green", mask_green)
cv2.imshow("mask_red", mask_red)
cv2.imshow("mask_yellow", mask_yellow)
cv2.imshow("mask_blue", mask_blue)



for circle in circles[0]:
    x, y, r = circle
    roi = mask_green[int(y-r):int(y+r), int(x-r):int(x+r)]
    cv2.circle(image, (int(x), int(y)), int(r), (100, 100, 100), 2)
    if cv2.countNonZero(roi) > 0.5 * roi.size:
        cv2.circle(blur, (int(x), int(y)), int(r), (0, 255, 0), 2)

    roi = mask_red[int(y-r):int(y+r), int(x-r):int(x+r)]
    if cv2.countNonZero(roi) > 0.5 * roi.size:
        cv2.circle(blur, (int(x), int(y)), int(r), (0, 0, 255), 2)

    roi = mask_yellow[int(y-r):int(y+r), int(x-r):int(x+r)]
    if cv2.countNonZero(roi) > 0.5 * roi.size:
        cv2.circle(blur, (int(x), int(y)), int(r), (0, 155, 255), 2)

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()