import warnings
# 有時compile 會報告warning訊息（提醒使用者某些function在最新版不在支援等等，或是他建議的寫法）
# 增加這一行是將warning的訊息給關閉（前提是你看過warning訊息並且了解後再關閉），
# 你可以試著將warning 這一段code給註解，看看compile提供給你的warning訊息是什麼
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold

# 載入波士頓房屋資料集
df = pd.read_csv("./boston_housing.csv")
dataset = df.values

X = dataset[:, 0:13] # 房價特徵
Y = dataset[:, 13] # 房價

# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)

# 分割訓練和測試資料集
X_train, Y_train = X[:400], Y[:400]     # 訓練資料前400筆
X_test, Y_test = X[400:], Y[400:]       # 測試資料後107筆

# k-fold, k =10
# 這一行主要是宣告使用KFold 要切割成幾份
# 用法如 kf.split(X, Y) <--- 你可以試著 print(kf.split(X, Y)) 這樣子的東西叫做generator
# 關於 generator 的概念，這邊拿常見的例子來說明
# for i in range(10) 當中的 range(10) ，就是一個generator，他會隨著for迴圈的進行由 0 ~ 9 每次吐出一個數字
# 所以 kf.split(X,Y) 也是會執行一樣的動作(隨著for迴圈吐出東西) ，不過他是會吐出兩個東西分別是 train 與 val
# 所以底下才會這樣子寫 for train, val in kf.split(X,Y): <--- 由於kf.split(X,Y) 是吐出兩個東西所以我們用兩個變數去承接
# 除此之外你可以試著單獨執行下面這段程式碼

# for train, val in kf.split(X,Y):
#   print(train)
#   print(val)

# 你就會發現原來kf.split(X,Y)已經將資料切成k 份，當中k-1份是train ，另外1份是val，並且每次都不同，重複10次

kf = KFold(n_splits = 10)

# 拿來紀錄 val_accuracy的list
val_acc = []

# 定義模型
model = Sequential()
model.add(Dense(12, input_shape=(X_train.shape[1],), activation="sigmoid"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="relu")) # 輸出層中的activation原為 sigmoid，但是sigmoid的輸出為0~1，與預測房價不符，所以改為relu
# 編譯模型
model.compile(loss="mse", optimizer="sgd", metrics=["mae"])
# 顯示模型架構
model.summary()

# 利用 k-fold 切分10份資料，其中 9份拿去train，剩下的 1份拿去 val
for train, val in kf.split(X_train, Y_train):
    # 訓練模型
    model.fit(X_train[train], Y_train[train], epochs=120, batch_size=20, verbose=0)

    # 計算驗證組準確度
    predict_price = model.predict(X_train[val]).squeeze().tolist()
    predict_price = [round(p, 1) for p in predict_price]

    error_range = 2 # 「預測出的房價」與「真實房價」兩者間可以接受的誤差
    result = abs(predict_price - Y_train[val])
    correct = len([x for x in result if x <= error_range])
    accuracy = correct / len(Y_train[val])
    # 紀錄每次驗證組的準確度
    val_acc.append(accuracy)

print("驗證組平均預測準確度:{:.2f}".format(sum(val_acc)/len(val_acc)))


# 計算測試組準確度
predict_price = model.predict(X_test).squeeze().tolist()
# 這種形式的for in 用法等同於以下這段 code:
# x = []
# for p in predict_price:
#   p = round(p, 1)
#   x.append(p)

# 可以發現 這樣子的寫法 1行可以抵 4行來用
# 並且這樣子的寫法可以比較簡潔乾淨，之後可以試著往這方向前進
predict_price = [round(p, 1) for p in predict_price]

error_range = 2 # 可以接受的誤差範圍
result = abs(predict_price - Y_test)
correct = len([x for x in result if x <= error_range])
accuracy = correct / len(Y_test)
print("測試組平均預測準確度:{:.2f}".format(accuracy))



