from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support
import numpy as np

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=43)

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
predict = model.predict(X_test)

p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(y_test, predict, average="micro")
p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_test, predict, average="macro")

print(f"Macro F1: {f_macro:.3f}") 
print(f"Micro F1: {f_micro:.3f}")