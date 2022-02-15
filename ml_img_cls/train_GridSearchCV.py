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

def select_clfs(X_train, X_test, y_train, y_test):
    clfs = {
        'svm': SVC(),
        'rf': RandomForestClassifier(),
        'adaboost': AdaBoostClassifier(),
        'gbdt': GradientBoostingClassifier(),
        'xgboost': XGBClassifier(verbosity=0, use_label_encoder=False)
    }

    for clf in clfs.keys():
        print('classifier: ', clf)
        model = clfs[clf]
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        best_parameters = model.get_params()
        for param_name in best_parameters.keys():
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        print('score: ', score)
        print(classification_report(y_test, y_pred, digits=4))


def params_optim(X_train, X_test, y_train, y_test):
    other_params = {
                'learning_rate': 0.3, # default: 0.3
                'n_estimators': 100, # default: 100
                'max_depth': 6, # default: 6
                'min_child_weight': 1, # default: 1
                'gamma': 0, # default: 0
                'subsample': 1, # default: 1
                'colsample_bytree': 1, # default: 1
                'reg_alpha': 0, # default: 0
                'reg_lambda': 1, # default: 1
                'scale_pos_weight': 1, # default: 1
                'random_state': 0, # default: 0
                'verbosity': 0, # 0 (silent), 1 (warning), 2 (info), 3 (debug)
                }
    
    cv_params = [
            {'n_estimators': range(50,160,10)},
            {'max_depth': range(3,11,1), 'min_child_weight': range(1,6,1)},
            {'gamma': [i/10.0 for i in range(0,6)]},
            {'subsample': [i/10.0 for i in range(5,11)], 'colsample_bytree': [i/10.0 for i in range(5,11)]},
            {'reg_alpha': [0, 0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]},
            {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]},
            ]

    for cv_param in cv_params:
        model = XGBClassifier(**other_params, use_label_encoder=False)
        estimators = GridSearchCV(estimator=model, param_grid=cv_param, cv=5)
        estimators.fit(X_train, y_train)
        print('best score: ', estimators.best_score_)
        print('best params: ', estimators.best_params_)
        best_model = estimators.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)
        # print(y_prob[:,1])
        print(classification_report(y_test, y_pred, digits=4))

        other_params.update(estimators.best_params_)
    
    print(other_params)
    model = XGBClassifier(**other_params, use_label_encoder=False)
    model.fit(X_train, y_train)
    # model.save_model('model.bin')
    joblib.dump(model, 'model.bin')

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    # print(y_prob[:,1])
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # select_clfs(X_train, X_test, y_train, y_test)
    params_optim(X_train, X_test, y_train, y_test)