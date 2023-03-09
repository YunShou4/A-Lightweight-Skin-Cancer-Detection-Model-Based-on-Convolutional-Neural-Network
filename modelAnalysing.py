import random
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from dataAnalysing import dataLoad
from sklearn.manifold import TSNE
from dataAnalysing import dataLoad
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


# load model
# 目标模型的路径
model = load_model("./model/data_balanced/vgg16_half_fire")

# -------------------------------------------------------------------------------------
# 重点，去掉最后一层，新模型popLayerModel的输出就是倒数第二层的输出向量
model.summary()
# popLayerName修改成想得到输出结果的层的名称
# 层的名称可以用model.summary()查看
popLayerName = "global_average_pooling2d_2"
popLayerModel = Model(inputs=model.input, outputs=model.get_layer(popLayerName).output)
popLayerModel.summary()
# -------------------------------------------------------------------------------------

# load data
X = dataLoad("./data/data_balanced.npy")
y = dataLoad("./data/label_balanced.npy")

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


out = popLayerModel.predict(X)
# t-sne
d = TSNE(n_components=3, perplexity=30, init='pca', learning_rate=1000, random_state=0).fit_transform(out[:5000])
# LDA
# d = LinearDiscriminantAnalysis(n_components=2).fit_transform(out, y_new)
# PCA
# pca = PCA(n_components=2)
# d = pca.fit_transform(out)

"""
# 2D figure
plt.figure(figsize=(12, 12))
plt.scatter(d[:5000,0], d[:5000,1], marker='*', c=y_new[:5000], s=0.1)
plt.colorbar()
plt.show() 
"""

# 3D figure
# 经过测试，t-sne方法降到3维效果最好
plt.figure("3D Scatter", facecolor="lightgray")
ax3d = plt.gca(projection="3d")
ax3d.scatter(d[:5000, 0], d[:5000, 1],d[:5000, 2], c=y_new[:5000], s=1)
plt.show()
