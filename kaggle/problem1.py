import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
import pandas as pd

train = pd.read_csv('C:\\ML\\ML-projects\\kaggle\\train.csv')
test = pd.read_csv('C:\\ML\\ML-projects\\kaggle\\test.csv')
X_train = train.drop(columns=['Survived', 'Name', 'Cabin', 'Ticket'])
y_train = train['Survived']
X_test = test.drop(columns=['Name', 'Cabin', 'Ticket'])
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())
embarked_mode = X_train['Embarked'].mode()[0]
X_train['Embarked'] = X_train['Embarked'].fillna(embarked_mode)
X_test['Embarked'] = X_test['Embarked'].fillna(embarked_mode)

encoder = LabelEncoder()
X_train['Sex'] = encoder.fit_transform(X_train['Sex'])
X_test['Sex'] = encoder.transform(X_test['Sex'])

preprocessor = ColumnTransformer(
    transformers=[
        ('embarked', OneHotEncoder(handle_unknown='ignore'), ['Embarked'])
    ],
    remainder='passthrough'
)

pipe = Pipeline([
    ('prep', preprocessor),
    ('xgb', xgb.XGBClassifier())
])

param = {
    "xgb__n_estimators" : [200],
    "xgb__max_depth" : [15],
    "xgb__learning_rate" : [0.01],
}

model = GridSearchCV(
    estimator=pipe,
    cv=5,
    param_grid=param
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': pred
})

submission.to_csv(
    r'C:\\ML\\ML-projects\\kaggle\\submission.csv',
    index=False
)

print(submission.head())