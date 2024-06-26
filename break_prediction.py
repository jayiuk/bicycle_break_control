import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

r_data = pd.read_csv('./km_break.csv')
data = r_data.drop(labels = ['고장구분', 'pca_x', 'pca_y'], axis = 1)
x_data = data.iloc[:, 0:3]
y_data = data.iloc[:, -1]
train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size = 0.2, random_state = 42)

params = {'n_estimators' : [10, 500],
          'max_depth' : [1, 100],
          'random_state' : [20, 70],
          'criterion' : ["gini", "entropy", "log_loss"]}

rf = RandomForestClassifier()
gv = GridSearchCV(estimator = rf, param_grid = params, scoring = 'accuracy', refit = True)
gv.fit(train_x, train_y)
print(gv.best_params_)
rf_clf = RandomForestClassifier(n_estimators = 10, max_depth = 100, criterion = 'gini', random_state = 20)

def train(model, train, target):
    result = model.fit(train, target)
    return result

def predict(model, test):
    pred = model.predict(test)
    return pred


accuracy = accuracy_score(test_y, predict(train(rf_clf, train_x, train_y), test_x))
print(accuracy)