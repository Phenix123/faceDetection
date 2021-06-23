import os

# Представьте только что написанный исходный файл
from detect_face import detect_face
import cv2


def prepare_training_data():
    # Прочтите название изображения в папке обучения
    dirs = os.listdir(r'./bigimg_train')
    faces = []
    labels = []
    for image_path in dirs:
        # Если название картинки начинается со слова happy, метка 1l; sad начинается с метки 2
        if image_path[0] == 'h':  ## happy
            label = 1
        elif image_path[0] == 's':  ## sad
            label = 2
        elif image_path[0] == 'a':  ##angry
            label = 3
        elif image_path[0] == 'd':  ## disgust
            label = 4
        elif image_path[0] == 'f':  ## fear
            label = 5
        elif image_path[0] == 'n':  ## neutral
            label = 6
        else:
            label = 7

        # Получить путь к изображению
        image_path = './bigimg_train/' + image_path

        # Вернуться к оттенкам серого, вернуться к объекту Mat
        image = cv2.imread(image_path, 0)

        # Отображать изображение в форме окна, отображать 100 миллисекунд
        # cv2.imshow("Training on image...", image)
        # cv2.waitKey(100)

        # Вызов функции, которую мы написали ранее
        face, rect = detect_face(image)
        if face is not None:
            faces.append(face)
            labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels
