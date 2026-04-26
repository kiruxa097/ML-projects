from sklearn.datasets import load_iris

iris = load_iris()

print("Названия признаков: ", iris.feature_names)
print("Названия сортов: ", iris.target_names)
print("Количество цветков: ", len(iris.data))
print("Признаки первого цветка: ", iris.data[0])
print("Сорт первого цветка(число): ", iris.target[0])
print("Сорт первого цветка(строка(название)): ", iris.target_names[iris.target[0]])