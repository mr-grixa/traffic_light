# ������� ������ ������� ������ � ������
cap = cv2.VideoCapture(0)

while True:
    # ����������� ���� � ������
    ret, frame = cap.read()

    # ��������� ����������� � �������� ������������ HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ������ �������� ������, ������� ����� ��������
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])

    # ��������� �����, ����� �������� �������, ��������������� ��������� ��������� ������
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # ������� ������� �������� �� �����������
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ������� ������ � ������� ����������� ������ ��� ������� �������
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles.append((center, radius))

    # ������ ����� �� �����������
    for center, radius in circles:
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

    # ���������� ����������� � ������
    cv2.imshow('frame', frame)

    # ���� ������������ �������� ������� 'q', �� ������� �� �����
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ����������� �������
cap.release()
cv2.destroyAllWindows()

