import numpy as np
import matplotlib.pyplot as plt


class ThresholdFunction:

    DomainDimension = 0
    DomainRings = None
    ImageDimension = None
    LinFormCoeff = []
    Borders = []

    MaxLinForm = 0
    MinLinForm = 0
    __F = None
    Grid = None

    def __init__(self, dimension_im, rings_dim=None, coefficients=None, borders=None, cube=None):
        if cube is None:
            self.DomainRings = np.asarray(rings_dim)
        else:
            self.DomainRings = np.asarray([cube[0]] * cube[1])
        self.ImageDimension = dimension_im
        self.DomainDimension = np.prod(self.DomainRings)

        if coefficients is None:
            self.LinFormCoeff = np.random.randint(-10000, 10000, len(self.DomainRings))
        else:
            self.LinFormCoeff = np.asarray(coefficients)

        self.MaxLinForm = np.dot(self.DomainRings - 1, np.where(self.LinFormCoeff < 0, 0, self.LinFormCoeff))
        self.MinLinForm = np.dot(self.DomainRings - 1, np.where(self.LinFormCoeff > 0, 0, self.LinFormCoeff))

        if borders is None:
            self.Borders = np.random.randint(self.MinLinForm, self.MaxLinForm + 1, self.ImageDimension - 1)
        else:
            self.Borders = np.asarray(borders)
        self.Borders = np.sort(self.Borders)

        self.Grid = np.indices(self.DomainRings).reshape((len(self.DomainRings), -1)).T
        self.__F = self.get_value(self.Grid).reshape(self.DomainRings)

    def write_options(self, namefile=None):
        if namefile is None:
            print("LinFormCoeff = ", self.LinFormCoeff)
            print("DomainRings = ", self.DomainRings)
            print("DomainDimension = ", self.DomainDimension)
            print("MaxLinForm = ", self.MaxLinForm)
            print("MinLinForm = ", self.MinLinForm)
            print("Borders = ", self.Borders)
            print("ImageDimension = ", self.ImageDimension)
            print("F = \n", self.__F)
        else:
            with open(namefile, 'w') as f:
                f.write("LinFormCoeff = " + str(self.LinFormCoeff) + "\n")
                f.write("DomainRings = " + str(self.DomainRings) + "\n")
                f.write("DomainDimension = " + str(self.DomainDimension) + "\n")
                f.write("MaxLinForm = " + str(self.MaxLinForm) + "\n")
                f.write("MinLinForm = " + str(self.MinLinForm) + "\n")
                f.write("Borders = " + str(self.Borders) + "\n")
                f.write("ImageDimension = " + str(self.ImageDimension) + "\n")
                f.write("F = \n"+ str(self.__F) + "\n")
            f.close()

    def normalization(self):
        eps = np.max(np.abs(self.LinFormCoeff))
        self.LinFormCoeff = 1.*self.LinFormCoeff / eps
        self.Borders = 1.*self.Borders / eps
        self.MaxLinForm /= eps
        self.MinLinForm /= eps

    def get_value(self, x):
        # На вход: набор векторов, на выходе: набор значений фунцкции в кольце
        scalar = np.dot(x, self.LinFormCoeff)
        R = np.zeros(scalar.shape)
        for i, border in enumerate(self.Borders):
            R[scalar >= border] = i + 1
        return R

    def calculate_increase_coeff_type1(self, ind):
        sum = 0
        k = self.DomainRings[ind]
        tempGrid = self.Grid.copy()
        for e in range(0, k-1 + 1):
            tempGrid[:, ind] = e
            sum += (2*e + 1 - k) * self.__F[tuple(zip(*tempGrid))]
        return np.sum(sum)/k

    def geometric_algoritm(self):
        pass

    def check(self):
        l = list(map(lambda i: self.calculate_increase_coeff_type1(i), range(len(self.DomainRings))))
        F = list(map(lambda i: self.Grid[self.get_value(self.Grid) == i], range(self.ImageDimension)))
        F = [F[i] for i in range(len(F)) if len(F[i]) != 0]
        maxF = []
        minF = []
        argMax = []
        argMin = []
        for i in range(0, len(F)):
            maxF.append(np.max(np.dot(F[i], l)))
            argMax.append(np.argmax(np.dot(F[i], l)))
            minF.append(np.min(np.dot(F[i], l)))
            argMin.append(np.argmin(np.dot(F[i], l)))

        for i in range(0, len(F)-1):
            if minF[i+1] <= maxF[i]:
                return False
        return True

    @staticmethod
    def correction(l, u, v):
        l = l - u + v
        return l

    def draw2d(self, name=None):
        colors = "rbgcmyk"
        markers = "xo"
        F = list(map(lambda i: self.Grid[self.get_value(self.Grid) == i], range(self.ImageDimension)))
        F = [F[i] for i in range(len(F)) if len(F[i]) != 0]
        for i in range(0,len(F)):
            plt.plot(F[i][:, 0], F[i][:, 1], markers[i % 2], color=colors[i % len(colors)], markersize=4)

        for border in self.Borders:
            X, Y = self.line1d(border, self.LinFormCoeff)
            plt.plot(X, Y, linewidth=1, color='black')

        l = list(map(lambda i: self.calculate_increase_coeff_type1(i), range(len(self.DomainRings))))
        for i in range(0, len(F)-1):
            b = (np.max(np.dot(F[i], l)) + np.min(np.dot(F[i + 1], l)))/2
            X, Y = self.line1d(b, l)
            plt.plot(X, Y, linewidth=1, color='red')


        plt.grid()
        if name is None:
            plt.show()
        else:
            plt.savefig(name)
        plt.close()

    def line1d(self, b, a):
        # a1x1+a2x2+b=0   =>  x1 = (b+a2x2)/-a1  x2 = (b+a1x1)/-a2
        X = []
        Y = []
        k1, k2 = self.DomainRings
        a1, a2 = a
        if a1 == 0:
            a1 = 0.00000001
        if a2 == 0:
            a2 = 0.00000001

        if 0 <= b/a1 <= k1 - 1:
            X.append(b/a1)
            Y.append(0)
        if 0 <= b/a2 <= k2 - 1:
            X.append(0)
            Y.append(b/a2)
        if 0 <= (b - a2 * (k2 - 1)) / a1 <= k1 - 1:
            X.append((b - a2 * (k2 - 1)) / a1)
            Y.append(k2 - 1)
        if 0 <= (b - a1 * (k1 - 1)) / a2 <= k2 - 1:
            X.append(k1 - 1)
            Y.append((b - a1 * (k1 - 1)) / a2)
        return X, Y
