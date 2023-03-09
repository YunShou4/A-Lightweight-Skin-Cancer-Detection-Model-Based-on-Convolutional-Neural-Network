import numpy as np
from PIL import Image
from keras.models import load_model


def predict(model, inputImgPath):
    # process inputdata
    input = []
    image = Image.open(inputImgPath)
    image = image.resize([28, 28], Image.ANTIALIAS)
    npVector = np.array(image) / 255.0
    input.append(npVector)

    # predict
    predict = model.predict(np.array(input))
    label = np.argmax(predict)

    if label == 0:
        return "Melanocytic nevi"
    if label == 1:
        return "Melanoma"
    if label == 2:
        return "Benign keratosis-like lesions"
    if label == 3:
        return "Basal cell carcinoma"
    if label == 4:
        return "Actinic keratoses"
    if label == 5:
        return "Vascular lesions"
    if label == 6:
        return "Dermatofibroma"



if __name__ == "__main__":
    model = load_model("./model")

    # image path
    inputImgPath = "./data/HAM10000_images/ISIC_0024306.jpg"
    predict = predict(model, inputImgPath)
    print("--------------------")
    print("Your type is: ", predict)
    print("--------------------")