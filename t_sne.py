import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataAnalysing import dataLoad
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


# load data
X = dataLoad("./data/data_balanced.npy")
y = dataLoad("./data/label_balanced.npy")
X = X.reshape(len(X), 28*28*3)

# 洗乱数据
index = [i for i in range(len(X))] 
random.shuffle(index)
X = X[index]
y = y[index]

y_new = []
y_list = []
for i in y:
    if i[0] == 1:
        y_list.append(0)
        y_new.append(0)
    if i[1] == 1:
        y_list.append(0)
        y_new.append(1)
    if i[2] == 1:
        y_list.append(0)
        y_new.append(2)
    if i[3] == 1:
        y_list.append(0)
        y_new.append(3)
    if i[4] == 1:
        y_list.append(0)
        y_new.append(4)
    if i[5] == 1:
        y_list.append(0)
        y_new.append(5)
    if i[6] == 1:
        y_list.append(0)
        y_new.append(6)
    if i[7] == 1:
        y_list.append(7)
        y_new.append(7)


# t-sne
# d = TSNE(n_components=3, perplexity=30, init='pca', learning_rate=1000, random_state=0).fit_transform(X[:5000])
# LDA
d = LinearDiscriminantAnalysis(n_components=3).fit_transform(X, y_new)
# PCA
# pca = PCA(n_components=2)
# d = pca.fit_transform(X)

"""
# 2D figure
plt.figure(figsize=(12, 12))
plt.scatter(d[:,0], d[:,1], marker='*', c=y_list, s=0.1)
plt.colorbar()
plt.show() 

"""
# 3D figure
plt.figure("3D Scatter", facecolor="lightgray")
ax3d = plt.gca(projection="3d")
ax3d.scatter(d[:, 0], d[:, 1],d[:, 2], c=y_list[:], s=1)
plt.show()

