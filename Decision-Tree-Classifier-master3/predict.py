import pandas as pd
import joblib  # 用于加载模型
from sklearn.preprocessing import StandardScaler

# 加载保存的模型
classifier = joblib.load('decision_tree_model.pkl')

# 加载新的测试集
new_test_data = pd.read_csv('E:/D/1.csv')

# 分离特征
new_X = new_test_data.iloc[:, 1:]  # 假设第一列以外的是特征

# 特征缩放
scaler = StandardScaler()
new_X = scaler.fit_transform(new_X)

# 进行预测
new_Y_pred = classifier.predict(new_X)

for pred in new_Y_pred:
    if pred == 0:
        print("正常")
    elif pred == 1:
        print("牙周炎")