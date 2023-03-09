import numpy as np
import matplotlib.pyplot as plt 


def dataLoad(dataPath):
    # load data (.npz)
    datas = np.load(dataPath, allow_pickle = True)

    return datas 



def dataVisualize(X, Y, title):
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.bar(x = X,
           height = Y,
           width = 0.6,
           align = "center",
           color = "red")
    ax.set_title(title, fontsize = 15)
    
    for a,b,i in zip(X, Y, range(len(X))):
        plt.text(a, b + 50, "%d"%Y[i], ha='center', fontsize = 10)

    plt.show()



if __name__ == "__main__":
    datas = dataLoad("./data/loading_datas.npy")

    # label classes: 8
    # nv: Melanocytic nevi
    # mel: Melanoma
    # bkl: Benign keratosis-like lesions
    # bcc: Basal cell carcinoma
    # akiec: Actinic keratoses
    # vasc: Vascular lesions
    # df: Dermatofibroma
    # benign
    nv = 0
    mel = 0
    bkl = 0
    bcc = 0
    akiec = 0
    vasc = 0
    df = 0
    benign = 0

    # Count the number of different labels
    for item in datas:
        if item[1] == "nv":
            nv = nv + 1
        if item[1] == "mel":
            mel = mel + 1
        if item[1] == "bkl":
            bkl = bkl + 1
        if item[1] == "bcc":
            bcc = bcc + 1
        if item[1] == "akiec":
            akiec = akiec + 1
        if item[1] == "vasc":
            vasc = vasc + 1
        if item[1] == "df":
            df = df + 1
        if item[1] == "benign":
            benign = benign + 1

    dataNum = len(datas)
    dataShape = datas[0][0].shape
    print("--------------------------------------------------")
    print("the number of images: %d" % (dataNum))
    print("the shape of each pieces of the data: ", end = "")
    print(dataShape)
    print("--------------------------------------------------")

    # visualize
    X = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df", "benign"]
    Y = [nv, mel, bkl, bcc, akiec, vasc, df, benign]  
    title = "the Number of Different Labels"
    dataVisualize(X, Y, title)
