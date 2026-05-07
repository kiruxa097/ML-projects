from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X, y = bc.data, bc.target

print("======= ДАТАСЕТ BREAST CANCER (Рак груди) =======")
print()
print(f"Количество образцов: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Классы: {bc.target_names}")