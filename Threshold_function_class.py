import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ThresholdFunction:

    DomainDimension = 0
    DomainRings = None
    ImageDimension = None
    LinFormCoeff = []
    Borders = []

    MaxLinForm = 0
    MinLinForm = 0
    F = None
    Grid = None

    methodType = None

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
        self.F = self.get_value(self.Grid).reshape(self.DomainRings)

        self.methodType = 'old'

    def write_options(self, filename=None):
        if filename is None:
            print("LinFormCoeff = ", self.LinFormCoeff)
            print("DomainRings = ", self.DomainRings)
            print("DomainDimension = ", self.DomainDimension)
            print("MaxLinForm = ", self.MaxLinForm)
            print("MinLinForm = ", self.MinLinForm)
            print("Borders = ", self.Borders)
            print("ImageDimension = ", self.ImageDimension)
            print("F = \n", self.F)
        else:
            with open(filename, 'w') as f:
                f.write("LinFormCoeff = " + str(self.LinFormCoeff) + "\n")
                f.write("DomainRings = " + str(self.DomainRings) + "\n")
                f.write("DomainDimension = " + str(self.DomainDimension) + "\n")
                f.write("MaxLinForm = " + str(self.MaxLinForm) + "\n")
                f.write("MinLinForm = " + str(self.MinLinForm) + "\n")
                f.write("Borders = " + str(self.Borders) + "\n")
                f.write("ImageDimension = " + str(self.ImageDimension) + "\n")
                f.write("F = \n" + str(self.F) + "\n")
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

    def geometric_algorithm(self):
        # Todo: Алгоритм должен работать с нормализованными данными
        pass

    def check(self):
        l = list(map(lambda i: self.calculate_increase_coeff(i), range(len(self.DomainRings))))
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

    def setMethodType(self, type='old'):
        self.methodType = type

    def getMethodType(self):
        return self.methodType

    def calculate_increase_coeff(self,ind):
        if self.methodType is 'old':
            return self.old_method_increase_coeff(ind)
        if self.methodType is 'new':
            return self.new_method_increase_coeff(ind)

        return None

    def new_method_increase_coeff(self, ind):
            sum = 0
            k = self.DomainRings[ind]
            tempGrid = self.Grid.copy()
            for e in range(0, k):
                tempGrid[:, ind] = e
                sum += (2*e + 1 - k) * self.F[tuple(zip(*tempGrid))]
            return np.sum(sum) / k

    def old_method_increase_coeff(self, ind):
            sum = 0
            k = self.DomainRings[ind]
            tempGrid = self.Grid.copy()
            for e in range(0, k):
                tempGrid[:, ind] = e
                sum += (2 * e + 1 - k) * self.F[tuple(zip(*tempGrid))]
            return np.sum(sum) / k


class ThresholdFunction2D(ThresholdFunction):

    def draw2d(self, name=None):
        colors = "rbgcmyk"
        markers = "xo"
        F = list(map(lambda i: self.Grid[self.get_value(self.Grid) == i], range(self.ImageDimension)))
        F = [F[i] for i in range(len(F)) if len(F[i]) != 0]
        for i in range(0, len(F)):
            plt.plot(F[i][:, 0], F[i][:, 1], markers[i % 2], color=colors[i % len(colors)], markersize=4)

        for border in self.Borders:
            X, Y = self.border_line(border, self.LinFormCoeff)
            plt.plot(X, Y, linewidth=1, color='black')

        l = list(map(lambda i: self.calculate_increase_coeff(i), range(len(self.DomainRings))))
        for i in range(0, len(F) - 1):
            b = (np.max(np.dot(F[i], l)) + np.min(np.dot(F[i + 1], l))) / 2
            X, Y = self.border_line(b, l)
            plt.plot(X, Y, linewidth=1, color='red')
        plt.grid()
        if name is None:
            plt.show()
        else:
            plt.savefig(name)
        plt.close()

    def border_line(self, b, a):
        # a1x1+a2x2+b=0   =>  x1 = (b+a2x2)/-a1  x2 = (b+a1x1)/-a2
        X = []
        Y = []
        k1, k2 = self.DomainRings
        a1, a2 = a
        if a1 == 0:
            a1 = 0.00000001
        if a2 == 0:
            a2 = 0.00000001

        if 0 <= b / a1 <= k1 - 1:
            X.append(b / a1)
            Y.append(0)
        if 0 <= b / a2 <= k2 - 1:
            X.append(0)
            Y.append(b / a2)
        if 0 <= (b - a2 * (k2 - 1)) / a1 <= k1 - 1:
            X.append((b - a2 * (k2 - 1)) / a1)
            Y.append(k2 - 1)
        if 0 <= (b - a1 * (k1 - 1)) / a2 <= k2 - 1:
            X.append(k1 - 1)
            Y.append((b - a1 * (k1 - 1)) / a2)
        return X, Y


class ThresholdFunction3D(ThresholdFunction):

    def draw3d(self, name=None):
        colors = "rbgcmyk"
        markers = "xo"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0, self.ImageDimension):
            x, y, z = (self.F == i).nonzero()
            ax.scatter(x, y, z, c=colors[i % len(colors)])

        for border in self.Borders:
            x, y, z = self.border_plane(border, self.LinFormCoeff)
            ax.plot_surface(x, y, z, color='black', alpha=0.1)

        ax.set_xlim3d([0, self.DomainRings[0] - 1])
        ax.set_ylim3d([0, self.DomainRings[1] - 1])
        ax.set_zlim3d([0, self.DomainRings[2] - 1])

        plt.draw()
        if name is None:
            plt.show()
        else:
            plt.savefig(name)
        plt.close()

    def border_plane(self, d, a):
        a1, a2, a3 = a
        k1, k2, k3 = self.DomainRings
        X = range(0, k1)
        Y = range(0, k2)
        X , Y = np.meshgrid(X, Y)
        Z = (d - a1*X - a2*Y)/a3
        return X, Y, Z
