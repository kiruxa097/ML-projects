from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=33)

pipe = Pipeline([
    ('scl', StandardScaler()),
    ('lr', LogisticRegression(solver='liblinear'))
])

params = {
    'lr__penalty' : ['l2', 'l1'],
    'lr__C' : [0.1, 1, 10, 50, 100]
}

grid = GridSearchCV(
    estimator=pipe, 
    param_grid=params,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

pred = grid.best_estimator_.predict(X_test)

print(accuracy_score(y_test, pred))