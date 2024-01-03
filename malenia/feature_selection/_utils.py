import matplotlib.pyplot as plt
import numpy as np


def get_pdfs_from_hist(df, feature, target, plot=True):
    target_uniques = df[target].unique()
    pdfs = []

    for target_unique in target_uniques:
        hist, bin_edges = np.histogram(
            df[df[target] == target_unique][feature].values, bins="auto", density=True
        )

        pdf = hist / np.sum(hist)
        pdfs.append(pdf)

        if plot:
            plt.plot(bin_edges[1:], pdf, label=target_unique)

    if plot:
        plt.xlabel("Value")
        plt.ylabel("Probability Distribution")
        plt.title("Probability Distribution From Histogram")
        plt.legend()

    return pdfs


def get_pdfs_from_distribution(df, feature, target, distribution, plot=True):
    target_uniques = df[target].unique()

    pdfs = dict()

    for target_unique in target_uniques:
        feature_values = df[df[target] == target_unique][feature].values

        mu, sigma = np.mean(feature_values), np.std(feature_values)

        x_min, x_max = np.min(feature_values), np.max(feature_values)
        x = np.linspace(x_min, x_max, 100)

        pdf = distribution.pdf(x, mu, sigma)
        pdfs[target_unique] = pdf

        if plot:
            plt.plot(x, pdf, label=target_unique)

    if plot:
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title("Normal Probability Density Function")
        plt.title("Probability Density Function")
        plt.legend()
        plt.show()

    return pdfs
