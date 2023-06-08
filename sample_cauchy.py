import numpy

import seaborn as sns
import matplotlib.pyplot as plt


def create_cauchy_distribution(mu, gamma, size):
    u = numpy.random.uniform(0, 1, size)

    return mu + gamma * numpy.tan(numpy.pi * (u - 0.5))


def plot_cauchy_distribution(mu, gamma, size):
    x = create_cauchy_distribution(mu, gamma, size)

    sns.distplot(x, hist=False)
    plt.show()


if __name__ == '__main__':
    plot_cauchy_distribution(0, 1, 100)
