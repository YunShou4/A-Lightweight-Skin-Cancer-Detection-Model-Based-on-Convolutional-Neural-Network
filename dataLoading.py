import os
import numpy as np
from PIL import Image


def loadImages(imgDirPath):
    # datas: [(npVector, label), (npVector, label), ...]
    datas = []

    # load info of each image
    imagesInfo = open("./data/HAM10000_metadata.csv", "r")
    lines = imagesInfo.readlines()
    lines.remove(lines[0])

    for line in lines:  
        strList = line.split(",")

        imgPath = imgDirPath + strList[1] + ".jpg"
        image = Image.open(imgPath)
        # image resize: (450, 600, 3) -> (28, 28, 3) 不然内存炸了，或者太慢了
        image = image.resize([28, 28], Image.ANTIALIAS)

        npVector = np.array(image) / 255.0
        label = strList[2]

        datas.append((npVector, label))
    
    path = "./data/benign_images"
    files = os.listdir(path)
    for benignImgPath in files:
        # add benign images
        image = Image.open(path + "/" + benignImgPath)
        image = image.resize([28, 28], Image.ANTIALIAS)
        npVector = np.array(image) / 255.0
        datas.append((npVector, "benign"))
    
    return datas



def dataSave(listToSave, nameToSave):
    np.save(nameToSave, np.array(listToSave))



if __name__ == "__main__":
    datas = loadImages("./data/HAM10000_images/")
    dataSave(datas, "./data/loading_datas.npy")
