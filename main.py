import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from linear_classifier import linear_classifier
from sklearn.decomposition import PCA

# Jakość predykcji
def accuracy (y, y_pred):
    # Werydikacja poprawności rozmiarów etykiet i predykcji
    if (len(y) != len(y_pred)):
        raise ValueError("Incorrect size of y")

    # Wyliczenie jakości predykcji
    points = 0
    for index, label in enumerate(y_pred):
        if label == y[index]:
            points += 1
    
    return points / len(y_pred)

# Importowanie zbioru danych, wydzielenie etykiet i wzorców
dataset = np.genfromtxt("datasets/australian.csv", delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

# Przeporwadzenie standaryzacji zbioru
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Stworzenie obiektu do obsługi foldów
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1410)

# Inicjalizowanie nowego obiektu
lc = linear_classifier()

scores = []
# Podział zbiorów wg stratyfikowanej k-foldowej walidacji z powtórzeniami
for train_i, test_i in rskf.split(X, y):
    X_train = X[train_i]
    y_train = y[train_i]
    X_test = X[test_i]
    y_test = y[test_i]

    # Uczenie
    lc.fit(X_train, y_train)

    # Predykcja
    y_pred = lc.predict(X_test)

    # Obliczenie jakości dla uzyskanej predykcji
    score = accuracy(y_test, y_pred)
    scores.append(score)

# Wyznaczenie średniej jakości
mean_score = sum(scores) / len(scores)
print(mean_score)

# Przeprowadzenie PCA, ANOVA
# Obliczenie jakości
# Porównanie jakości dla PCA, ANOVA (testy parowe, testy globalne na 7 rangach)