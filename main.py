from prepare_training_data import prepare_training_data
import cv2
import numpy as np
# Файл, который мы напишем позже
from predict import predict

if __name__ == '__main__':
    print("Preparing data...")
    # Вызвать функцию, написанную ранее, чтобы получить последовательность, содержащую несколько матриц лиц и их метки
    faces, labels = prepare_training_data()
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    # Получить (LBPH) распознаватель лиц
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Применить данные, обучить
    face_recognizer.train(faces, np.array(labels))
    print("Predicting images...")

    # Загрузите предсказанное изображение, здесь я просто, просто напишите путь напрямую
    test_img1 = cv2.imread(r"./img_predict/happy1.jpg",0)
    test_img2 = cv2.imread(r"./img_predict/sad1.jpg",0)

    # Делать предсказания
    # Обратите внимание, эта функция еще не написана! ! !
    predicted_img1 = predict(test_img1, face_recognizer)
    predicted_img2 = predict(test_img2, face_recognizer)
    print("Prediction complete")

    # Показать результаты прогноза
    cv2.imshow('Happy', predicted_img1)
    cv2.imshow('Sad', predicted_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

