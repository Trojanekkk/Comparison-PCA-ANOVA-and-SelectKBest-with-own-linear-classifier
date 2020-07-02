import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.stats import ttest_ind
from linear_classifier import linear_classifier

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

# Stworzenie słownika klas redukcji dla zadanej liczby wymiarów
dimension = 3
reducers = {
    "PCA" : PCA(n_components=dimension),
    "ANOVA" : SelectKBest(score_func=f_classif, k=dimension),
    "SelectKBest" : SelectKBest(score_func=mutual_info_classif, k=dimension)
}

# Lista danych do importowania
datasets = ['haberman']
d = ['monk-2', 'caesarian', 'australian', 'ring', 'phoneme', 'heart', 'titanic', 'haberman']
# Stworzenie obiektu do obsługi foldów
folds = 5
repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=1410)

# Inicjalizowanie nowego obiektu
lc = linear_classifier()

# Inicjalizacja tablicy wyników
if (len(datasets) > 1):
    scores = np.zeros((len(reducers), len(datasets), folds * repeats))
else:
    scores = np.zeros((len(reducers), folds * repeats))

for data_index, dataset in enumerate(datasets):

    # Importowanie zbioru danych, wydzielenie etykiet i wzorców
    dataset = np.genfromtxt("datasets/{}.csv".format(dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    # Przeporwadzenie standaryzacji zbioru
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Podział zbiorów wg stratyfikowanej k-foldowej walidacji z powtórzeniami
    for fold_i, (train_i, test_i) in enumerate(rskf.split(X, y)):
        
        #Wybór naprzemiennie PCA i ANOVA do redukcji kolejnych foldów
        for reducer_i, reducer in enumerate(reducers):
            reducer_obj = reducers[reducer]
            X_tmp = reducer_obj.fit_transform(X, y)
            X_train = X_tmp[train_i]
            y_train = y[train_i]
            X_test = X_tmp[test_i]
            y_test = y[test_i]

            # Uczenie
            lc.fit(X_train, y_train)

            # Predykcja
            y_pred = lc.predict(X_test)

            # Obliczenie jakości dla uzyskanej predykcji
            score = accuracy(y_test, y_pred)
            if (len(datasets) > 1):
                scores[reducer_i][data_index][fold_i] = score
            else:
                scores[reducer_i][fold_i] = score

# Zapisanie wyników do pliku
np.save("results", scores)

# Wyznaczenie średniej i wariancji jakości dla metod redukcji
for reducer_score_i, reducer_score in enumerate(scores):
    mean_score = np.mean(np.round(sum(reducer_score) / len(reducer_score), 3))
    std_score = np.round(np.std(reducer_score), 3)
    print("średnia: {}, std: {} <- {}".format(mean_score, std_score, list(reducers.keys())[reducer_score_i]))

# Testy parowe
# Porównanie jakości dla PCA, ANOVA (testy parowe, testy globalne na 7 rangach)