import numpy as np
from scipy.stats import rankdata
from scipy.stats import ranksums
from tabulate import tabulate

# Wczytywanie wyników eksperymentu
scores = np.load("results.npy")
print(scores)

# Lista nazw testowanych metod redukcji danych
reducers = ["PCA", "ANOVA", "SelectKBest"]

# Uśrednianie wyników po foldach
mean_scores = np.mean(scores, axis=2).T
print("Mean scores:\n", mean_scores, "\n")

# Przyporządkowanie rang
ranks = []
for score in mean_scores:
    ranks.append(rankdata(score).tolist())
ranks = np.array(ranks)
print("Ranks:\n", ranks, "\n")

# Uśrednienie rang
mean_ranks = np.mean(ranks, axis=0)
for index, reducer in enumerate(reducers):
    print(reducer + ": ", mean_ranks[index])

# Wstępne parametry dla statystyki wilcoxona
alfa = .05
w_statistic = np.zeros((len(reducers), len(reducers)))
p_value = np.zeros((len(reducers), len(reducers)))

# Obliczenie wartości statystyki 
for i in range(len(reducers)):
    for j in range(len(reducers)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

# Utworzenie kolumny nazw testowanych metod, sformatowanie i wyświetlenie wyników
column_header = np.array([[reducers[0]], [reducers[1]], [reducers[2]]])

w_statistic_table = np.concatenate((column_header, w_statistic), axis=1)
p_value_table = np.concatenate((column_header, p_value), axis=1)

print("w_statistic\n" + tabulate(w_statistic_table, reducers, floatfmt=".2f") + "\n")
print("p_value\n" + tabulate(p_value_table, reducers, floatfmt=".2f") + "\n")

# Wyznaczenie i wyświetlenie macierzy przewagi
advantage = np.zeros((len(reducers), len(reducers)))
advantage[w_statistic > 0] = 1

print("advantage\n" + tabulate(np.concatenate((column_header, advantage), axis=1), reducers) + "\n")

# Wyznaczenie i wyświetlenie macierzy istotności
significance = np.zeros((len(reducers), len(reducers)))
significance[p_value <= alfa] = 1

print("significance\n" + tabulate(np.concatenate((column_header, significance), axis=1), reducers) + "\n")