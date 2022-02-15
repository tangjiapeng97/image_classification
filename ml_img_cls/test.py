import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import classification_report
import joblib


def test(X_train, X_test, y_train, y_test):
    # save_model保存模型: model.save_model('model.bin')
    # sklearn接口加载模型
    # model = XGBClassifier()
    # booster = xgb.Booster()
    # booster.load_model('model.bin')
    # model._Booster = booster
    # y_pred = model.predict(xgb.DMatrix(X_test))
    # y_pred = [round(value) for value in y_pred]
    # xgbooost原生接口加载模型
    # model = xgb.Booster()
    # model.load_model('model.bin')
    # y_pred = model.predict(X_test)

    #joblib保存模型: joblib.dump(model, 'model.bin')
    #joblib加载模型
    model = joblib.load('model.bin')
    y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)
    # print(y_prob[:,1])
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == '__main__':
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    test(X_train, X_test, y_train, y_test)