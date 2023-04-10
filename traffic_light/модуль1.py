# Создаем объект захвата кадров с камеры
cap = cv2.VideoCapture(0)

while True:
    # Захватываем кадр с камеры
    ret, frame = cap.read()

    # Переводим изображение в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Задаем диапазон цветов, который хотим выделить
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])

    # Применяем маску, чтобы выделить объекты, соответствующие заданному диапазону цветов
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Находим контуры объектов на изображении
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Находим центры и радиусы описывающих кругов для каждого контура
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles.append((center, radius))

    # Рисуем круги на изображении
    for center, radius in circles:
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

    # Отображаем изображение с камеры
    cv2.imshow('frame', frame)

    # Если пользователь нажимает клавишу 'q', то выходим из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()

