import numpy as np
import math

class linear_classifier:
    # Zainicjowanie obiektu
    def __init__ (self):
        pass

    # Uczenie
    def fit (self, X, y):
        # Sprawdzenie wymiarów zbiorów
        if (len(X) != len(y)):
            raise ValueError("Incorrect size of X and y")

        self.X = X
        self.y = y

        # Ekstrakcja unikatowych klas
        self.unique_classes = []
        for label in self.y:
            if label not in self.unique_classes:
                self.unique_classes.append(label)
        if (len(self.unique_classes) != 2):
            raise ValueError("Classifier is for datasets with 2-classes only")

        # Obliczenie centroidów dla każdej z klas
        self.centroids = []
        for cls in self.unique_classes:
            tmp = []
            for index, element in enumerate(self.X):
                if (self.y[index] == cls):
                    tmp.append(element)
            self.centroids.append(np.mean(tmp, axis=0))

        # Obliczenie punktu środkowego między centroidami oraz wektora normalnego
        self.centroid_center = np.mean((self.centroids), axis=0)
        self.centroid_vector = np.sum((np.divide((self.centroids[0]), -1), self.centroid_center), axis=0)
        
        return self

    # Testowanie
    def predict(self, X):
        # Weryfikacja czy wykonano proces uczenia
        if not self.centroids:
            raise ValueError("Complete the 'Fit' function first")

        # Sprawdzenie rozmiaru zbioru testowego
        for i in X:
            if (len(i) != len(self.centroids[0])):
                raise ValueError("Incorrect size of X")

        # Predykcja poprzez obliczanie odległości od centroidów każdej z klas
        # y_pred = []
        # for i in range(len(X)):
        #     d = []
        #     for cls_i, cls in enumerate(self.unique_classes):
        #         sqr_d = 0
        #         for j in range(len(X[0])):
        #             sqr_d += (X[i][j] - self.centroids[cls_i][j]) ** 2
        #         d.append(math.sqrt(sqr_d))
        #     y_pred.append(d.index(min(d)))

        # Predykcja poprzez określenie położenia próbki względem hiperpłaszczyzny
        y_pred = []
        for i in range(len(X)):
            g = 0
            for index, j in enumerate(X[i]):
                g += self.centroid_vector[index] * (j - self.centroid_center[index])
            if (g > 0):
                y_pred.append(1)
            else:
                y_pred.append(0)

        return y_pred


                


        

