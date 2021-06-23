import cv2


def detect_face(img):
    # Преобразование изображения в изображение в градациях серого, потому что детектор лиц OpenCV требует изображения в градациях серого
    gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

    # Загрузите распознаватель лиц OpenCV, обратите внимание, что путь здесь - это место, которое вы сохранили при загрузке распознавателя ранее
    face_cascade = cv2.CascadeClassifier(r'C:\Users\cat10\Downloads\lbpcascade_frontalface.xml')

    #scaleFactor представляет собой отношение уменьшения размера каждого изображения, minNeighbors представляет минимальное количество смежных прямоугольников, которые составляют цель обнаружения.
    # Здесь выберите размер изображения, которое нужно уменьшить в 1,2 раза. Чем больше minNeighbors, тем точнее распознается лицо, но его также легко пропустить
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)

    # Если на картинке нет лица, картинка не будет участвовать в обучении, верните None
    if len(faces) == 0:
        return None, None

    # Извлечь область лица
    (x, y, w, h) = faces[0]

    # Вернуться к лицу и его области
    return gray[y:y + w, x:x + h], faces[0]

