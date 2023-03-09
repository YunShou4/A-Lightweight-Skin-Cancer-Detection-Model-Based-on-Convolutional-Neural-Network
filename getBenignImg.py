import cv2


def copyImages(imgDirPath):
    # load info of each image
    imagesInfo = open("./data/ISBI2016_ISIC_Part3_Training_GroundTruth.csv", "r")
    lines = imagesInfo.readlines()


    for line in lines:  
        strList = line.split(",")

        if strList[1] == "benign\n":
            imgPath = imgDirPath + strList[0] + ".jpg"
            img = cv2.imread(imgPath)
            copyPath = "./data/benign_images/" + strList[0] + ".jpg"
            cv2.imwrite(copyPath, img)



if __name__ == "__main__":
    copyImages("./data/ISBI2016_ISIC_Part3_Training_data/")
