import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
# from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

# Загрузка предобученной модели MobileNetV2
# model = keras.applications.MobileNetV2(weights='imagenet')
# Загрузка предобученной модели InceptionV3
# model = keras.applications.InceptionV3(weights='imagenet')

# Загрузка предобученной модели ResNet50
model = keras.applications.ResNet50(weights='imagenet')

# Список классов продуктов питания
food_classes = {'crayfish': 82, 'king_crab': 130, 'corn': 90, 'wine_bottle': 83, 'ice_cream': 276, 'ice_lolly': 51,
                'French_loaf': 260, 'bagel': 257, 'pretzel': 397, 'cheeseburger': 263, 'hotdog': 247, 'mashed_potato': 106,
                'head_cabbage': 22, 'cauliflower': 25, 'zucchini': 19, 'spaghetti_squash': 31, 'acorn_squash': 40,
                'butternut_squash': 43, 'cucumber': 14, 'artichoke': 53, 'bell_pepper': 26, 'cardoon': 17, 'Granny_Smith': 54,
                'strawberry': 32, 'lemon': 29, 'fig': 74, 'pineapple': 50, 'jackfruit': 94, 'custard_apple': 101,
                'pomegranate': 83, 'carbonara': 174, 'chocolate_sauce': 220, 'dough': 250, 'meat_loaf': 207, 'potpie': 204,
                'burrito': 230, 'red_wine': 85, 'espresso': 9, 'apple': 52, 'banana': 89, 'orange': 47, 'pizza': 292,
                'hamburger': 264, 'sushi': 143, 'chocolate_cake': 371, 'carrot': 37, 'broccoli': 28}

# Функция для предсказания классов продуктов на входной картинке
def predict_product(image):
    # Загрузка и предобработка входной картинки
    img = image
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(np.expand_dims(img, axis=0))

    # Получение предсказаний классов продуктов
    predictions = model.predict(img)
    top_predictions = decode_predictions(predictions, top=5)[0]

    t = ""

    # Вывод классов продуктов со степенью уверенности выше 85%
    for _, class_name, confidence in top_predictions:
        if class_name in food_classes and confidence > 0.85:
            t = t + f"Предсказанный класс: {class_name}, уверенность: {int (confidence * 100)}%. Калории в 100 г данного продукта: {food_classes[class_name]}. "
            print(f"Предсказанный класс: {class_name}, уверенность: {int (confidence * 100)}%.")
            print(f"Калории в 100 г данного продукта: {food_classes[class_name]}.")

    if (t == ""):
        t = "Еда на фотографии не обнаружена, либо степень уверенности нейросети в определении еды невысока."
    return t