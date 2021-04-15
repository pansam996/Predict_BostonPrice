import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

np.random.seed(7)  # 指定亂數種子
# 載入波士頓房屋資料集
df = pd.read_csv("./boston_housing.csv")
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:, 0:13]
Y = dataset[:, 13]
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割訓練和測試資料集
X_train, Y_train = X[:404], Y[:404]     # 訓練資料前404筆
X_test, Y_test = X[404:], Y[404:]       # 測試資料後102筆
# 定義模型
model = Sequential()
model.add(Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))
# 編譯模型
model.compile(loss="mse", optimizer="adam", 
              metrics=["mae"])
# 訓練模型
model.fit(X_train, Y_train, epochs=80, batch_size=16, verbose=0)
# 使用測試資料評估模型
mse, mae = model.evaluate(X_test, Y_test, verbose=0)    
print("MSE_test: ", mse)
print("MAE_test: ", mae)

