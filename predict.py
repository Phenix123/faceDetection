from detect_face import detect_face

# Мы не подготовили эти два файла
from draw_rectangle import draw_rectangle
from draw_text import draw_text


def predict(test_img, face_recognizer):
    # Преобразование тегов 1, 2 в текст
    subjects = ['', 'Happy', 'Sad', 'Angry', 'Disgust', 'Fear', 'Neutral', 'Surprise']

    # Получить копию изображения
    img = test_img.copy()

    # Определить лицо по изображению
    face, rect = detect_face(img)

    # Используйте наш распознаватель лиц, чтобы предугадать изображение
    label = face_recognizer.predict(face)
    # Получить имя соответствующего тега, возвращаемое распознавателем лиц
    label_text = subjects[label[0]]

    # Обратите внимание, что мы еще не написали следующие две функции! ! !
    # Нарисуйте прямоугольник вокруг обнаруженного лица
    draw_rectangle(img, rect)
    # Обозначьте эмоции на лице вокруг прямоугольника
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img
