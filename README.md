# A-Lightweight-Skin-Cancer-Detection-Model-Based-on-Convolutional-Neural-Network

构建了一个轻量化的基于CNN和SqueezeNet的模型，对于8分类的皮肤癌图像分类任务，用1/4的参数量，达到了97.8%的测试集准确率。  

项目成员：Yizhou Li, Hongxi Mao, Zhiran Wang

· dataLoading.py装载数据，得到loading_datas.npy  
· dataAnalysing.py，分析数据集分布和格式  
· dataProcessing.py，平衡数量不同的种类，得到data_balanced.npy和label_balanced.npy  
· train.py训练模型  
· predict.py，利用模型进行预测  

数据平衡方法：SMOTE-ENN（过采样-欠采样结合）  
数据降维与可视化：t-SNE、LDA
模型原理：深度可分离卷积  
