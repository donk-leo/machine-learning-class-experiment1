import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

X, y =load_svmlight_file("housing",n_features=13)
feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

X_train, X_test,y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



if hasattr(X_train,"toarray"):
    X_train=X_train.toarray()
    X_test=X_test.toarray()

train_df=pd.DataFrame(X_train,columns=feature_names)
test_df=pd.DataFrame(X_test,columns=feature_names)


imputer=SimpleImputer(strategy="median")

train_clean=pd.DataFrame(imputer.fit_transform(train_df),columns=feature_names)

test_clean=pd.DataFrame(imputer.transform(test_df),columns=feature_names)

#skewness中文为偏度
skewness=train_clean.skew()

log_columns =[
    column
    for column in feature_names
    if skewness[column]>1 and (train_clean[column]>=0).all and train_clean[column].nunique()>2
]

for column in log_columns:
    train_clean[column+"log"]=np.log1p(train_clean[column])
    test_clean[column+"log"]=np.log1p(test_clean[column])

#完成了缺失值填充，对于偏态分布明显的特征进行了处理以及表头增加，下面进行归一化处理


scaler=MinMaxScaler(feature_range=(-1,1))

X_train_final=scaler.fit_transform(train_clean)
X_test_final=scaler.transform(test_clean)#这里为什么不写fit，因为测试集直接用训练集的最大最小防止泄漏

np.savez(
    "processed_housing.npz",
    X_train=X_train_final,
    X_test=X_test_final,
    y_train=y_train,
    y_test=y_test

)

#一个打包器
joblib.dump(
    {
        "Imputer":imputer,
        "scaler":scaler,
        "log_columns":log_columns,
        "feature_names":list(train_clean.columns)
    },
    "preprocess_objects.joblib"
)