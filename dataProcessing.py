import os
import numpy as np
from PIL import Image
from dataLoading import dataSave
from dataAnalysing import dataLoad, dataVisualize
from imblearn.combine import SMOTEENN


def getOneHotLabel(labels):
    oneHotLabel = []
    for i in labels:
        sample = [0, 0, 0, 0, 0, 0, 0, 0]
        if i == "nv":
            sample[0] = 1
        if i == "mel":
            sample[1] = 1
        if i == "bkl":
            sample[2] = 1
        if i == "bcc":
            sample[3] = 1
        if i == "akiec":
            sample[4] = 1
        if i == "vasc":
            sample[5] = 1
        if i == "df":
            sample[6] = 1
        if i == "benign":
            sample[7] = 1
        oneHotLabel.append(np.array(sample))
    
    return np.array(oneHotLabel)



if __name__ == "__main__":
    datas = dataLoad("./data/loading_datas.npy")

    # balance dataset
    X = []
    y = []
    for item in datas:
        X.append(item[0].reshape(-1))
        y.append(item[1])

    X = np.array(X)
    y = np.array(y)
    print(X)
    print(y)
    
    smote_enn = SMOTEENN(random_state = 0)
    dataBalanced, labelBalanced = smote_enn.fit_resample(X, y)
    # print(dataBalanced)
    # print(dataBalanced.shape)
    # print(labelBalanced)
    # print(labelBalanced.shape) 

    # balanced dataset reshape
    temp = []
    for i in range(len(dataBalanced)):
        temp.append(dataBalanced[i].reshape(28, 28, 3))
    
    """
    path = "./data/benign_images_add"
    files = os.listdir(path)
    ii = 0
    for benignImgPath in files:
        # add benign images
        print(ii)
        ii += 1
        image = Image.open(path + "/" + benignImgPath)
        image = image.resize([28, 28], Image.ANTIALIAS)
        npVector = np.array(image) / 255.0
        temp.append(npVector)
        labelBalanced = np.append(labelBalanced, "benign")    
    """

    dataBalanced = np.array(temp)

    # save balanced dataset
    dataSave(dataBalanced, "./data/data_balanced.npy")
    dataSave(getOneHotLabel(labelBalanced), "./data/label_balanced.npy")

    nv = 0
    mel = 0
    bkl = 0
    bcc = 0
    akiec = 0
    vasc = 0
    df = 0
    benign = 0

    # Count the number of different labels
    for item in labelBalanced:
        if item == "nv":
            nv = nv + 1
        if item == "mel":
            mel = mel + 1
        if item == "bkl":
            bkl = bkl + 1
        if item == "bcc":
            bcc = bcc + 1
        if item == "akiec":
            akiec = akiec + 1
        if item == "vasc":
            vasc = vasc + 1
        if item == "df":
            df = df + 1
        if item == "benign":
            benign = benign + 1

    dataNum = len(dataBalanced)
    dataShape = dataBalanced[0].shape
    print("--------------------------------------------------")
    print("the number of images: %d" % (dataNum))
    print("the shape of each pieces of the data: ", end = "")
    print(dataShape)
    print("--------------------------------------------------")

    # visualize
    X = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df", "benign"]
    Y = [nv, mel, bkl, bcc, akiec, vasc, df, benign]  
    title = "the Number of Different Labels - Balenced"
    dataVisualize(X, Y, title)
