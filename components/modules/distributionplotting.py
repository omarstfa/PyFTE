import matplotlib.pyplot as plt
import math
from scipy.stats import expon, norm, weibull_min, lognorm
from modules.faultTreeReconstruction import get_object_name
import os
import numpy as np
import seaborn as sns


EXPORT_PNG = True


EMPTY_LIST = []

TOLERANCE = 0.000001


def get_index_of_first_zero_in_array(array):
    for index in range(0, len(array)):
        if array[index] < TOLERANCE:
            return index


def differentiate(x, y):
    if type(y) is list:
        dy = np.zeros(len(y), float)
    else:
        dy = np.zeros(y.shape, float)
    dy[0:-1] = np.diff(y) / np.diff(x)
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dy


def mean_squared_error(y, y_estimated):
    return np.square(y - y_estimated).mean()


def mse_exp(theoretical_distribution, estimated_distribution):
    theoretical_lambda= theoretical_distribution[1]
    theoretical_scale = 1/theoretical_lambda

    estimated_lambda = estimated_distribution[1]
    estimated_scale = 1 / estimated_lambda

    linspace = np.linspace(expon.ppf(0.001, scale=theoretical_scale), expon.ppf(0.999, scale=theoretical_scale), 1000)
    theoretical_pdf = expon.pdf(linspace, scale=theoretical_scale)
    estimated_pdf = expon.pdf(linspace, scale=estimated_scale)

    mse_pdf = mean_squared_error(theoretical_pdf, estimated_pdf)

    theoretical_cdf = expon.cdf(linspace, scale=theoretical_scale)
    estimated_cdf = expon.cdf(linspace, scale=estimated_scale)

    mse_cdf = mean_squared_error(theoretical_cdf, estimated_cdf)

    theoretical_reliability = 1 - expon.cdf(linspace, scale=theoretical_scale)
    estimated_reliability = 1 - expon.cdf(linspace, scale=estimated_scale)

    mse_reliability = mean_squared_error(theoretical_reliability, estimated_reliability)

    return [mse_pdf, mse_cdf, mse_reliability]


def setup_fig_subplots(metric):
    # Depending on Reliability or Maintainability the number of plots changes
    # Reliability has 3, Maintainability has only 2
    num_of_subplots = 0
    figure_width = 0

    if metric == 'Reliability':
        num_of_subplots = 4
        figure_width = 15
    if metric == 'Maintainability':
        num_of_subplots = 3
        figure_width = 12

    fig, subplots = plt.subplots(1, num_of_subplots, figsize=(figure_width, 4))
    return fig, subplots


def name_of_distribution(distribution):
    if distribution[0] == 'EXP':
        return 'Exponential'

    if distribution[0] == 'WEIBULL':
        return 'Weibull'

    if distribution[0] == 'NORMAL':
        return 'Normal'

    if distribution[0] == 'LOGNORM':
        return 'Lognormal'


def calculate_pdf(distribution, linspace):
    if distribution[0] == 'EXP':
        lambda_ = distribution[1]
        scale_ = 1 / lambda_
        return expon.pdf(linspace, scale=scale_)

    if distribution[0] == 'WEIBULL':
        scale = distribution[1]
        shape = distribution[2]
        return weibull_min.pdf(linspace, shape, loc=0, scale=scale)

    if distribution[0] == 'NORMAL':
        mu = distribution[1]
        sigma = distribution[2]
        return norm.pdf(linspace, loc=mu, scale=sigma)

    if distribution[0] == 'LOGNORM':
        mu = distribution[1]
        sigma = distribution[2]
        scale = math.exp(mu)
        return lognorm.pdf(linspace, sigma, loc=0, scale=scale)


def calculate_cdf(distribution, linspace):
    if distribution[0] == 'EXP':
        lambda_ = distribution[1]
        scale_ = 1 / lambda_
        return expon.cdf(linspace, scale=scale_)

    if distribution[0] == 'WEIBULL':
        scale = distribution[1]
        shape = distribution[2]
        return weibull_min.cdf(linspace, shape, loc=0, scale=scale)

    if distribution[0] == 'NORMAL':
        mu = distribution[1]
        sigma = distribution[2]
        return norm.cdf(linspace, loc=mu, scale=sigma)

    if distribution[0] == 'LOGNORM':
        mu = distribution[1]
        sigma = distribution[2]
        scale = math.exp(mu)
        return lognorm.cdf(linspace, sigma, loc=0, scale=scale)


def calculate_reliability(distribution, linspace):
    if distribution[0] == 'EXP':
        lambda_ = distribution[1]
        scale_ = 1 / lambda_
        return 1 - expon.cdf(linspace, scale=scale_)

    if distribution[0] == 'WEIBULL':
        scale = distribution[1]
        shape = distribution[2]
        return 1 - weibull_min.cdf(linspace, shape, loc=0, scale=scale)

    if distribution[0] == 'NORMAL':
        mu = distribution[1]
        sigma = distribution[2]
        return 1 - norm.cdf(linspace, loc=mu, scale=sigma)

    if distribution[0] == 'LOGNORM':
        mu = distribution[1]
        sigma = distribution[2]
        scale = math.exp(mu)
        return 1 - lognorm.cdf(linspace, sigma, loc=0, scale=scale)


def calculate_linspace(distribution):
    if distribution[0] == 'EXP':
        lambda_ = distribution[1]
        scale_ = 1 / lambda_
        return np.linspace(expon.ppf(0.001, scale=scale_), expon.ppf(0.999, scale=scale_), 1000)

    if distribution[0] == 'WEIBULL':
        scale = distribution[1]
        shape = distribution[2]
        return np.linspace(weibull_min.ppf(0.001, shape, loc=0, scale=scale),
                           weibull_min.ppf(0.999, shape, loc=0, scale=scale), 1000)

    if distribution[0] == 'NORMAL':
        mu = distribution[1]
        sigma = distribution[2]
        return np.linspace(norm.ppf(0.001, loc=mu, scale=sigma), norm.ppf(0.999, loc=mu, scale=sigma), 1000)

    if distribution[0] == 'LOGNORM':
        mu = distribution[1]
        sigma = distribution[2]
        scale = math.exp(mu)
        return np.linspace(lognorm.ppf(0.001, sigma, loc=0, scale=scale),
                           lognorm.ppf(0.999, sigma, loc=0, scale=scale), 1000)
    else:
        return np.linspace(0, 100, 1000)


def plot_identified_distribution_comparison(name, metric, distribution, times, theoretical_distribution):
    fig, subplots = setup_fig_subplots(metric)
    #fig.suptitle(name + '\n\n', fontsize=16)

    linspace = calculate_linspace(distribution)
    pdf = calculate_pdf(distribution, linspace)
    cdf = calculate_cdf(distribution, linspace)

    theoretical_pdf = calculate_pdf(theoretical_distribution, linspace)
    theoretical_cdf = calculate_cdf(theoretical_distribution, linspace)

    # First plot PDF
    subplots[0].plot(linspace, pdf, 'b-', lw=1, alpha=0.6, label='Reconstructed')
    subplots[0].plot(linspace, theoretical_pdf, 'r-', lw=1, alpha=0.6, label='Theoretical')
    subplots[0].set_xlabel('time (t)')
    subplots[0].set_ylabel('P(t)')
    subplots[0].legend()

    if times != EMPTY_LIST:
        if metric == 'Reliability':
            subplots[0].hist(times, bins=20, density=True, histtype='stepfilled', alpha=0.2, label='Time to failures')
        if metric == 'Maintainability':
            subplots[0].hist(times, bins=20, density=True, histtype='stepfilled', alpha=0.2, label='Time to repairs')
        subplots[0].legend()
    subplots[0].set_title('PDF')

    # Second plot CDF and/or Maintainability
    subplots[1].plot(linspace, cdf, 'b-', lw=1, alpha=0.6, label='Reconstructed')
    subplots[1].plot(linspace, theoretical_cdf, 'r-', lw=1, alpha=0.6, label='Theoretical')
    subplots[1].set_xlabel('time (t)')
    subplots[1].set_ylabel('P(T\u2264t)')
    subplots[1].legend()

    if metric == 'Maintainability':
        subplots[1].set_title('CDF (Maintainability)')
        subplots[2].plot(linspace, pdf / (1 - cdf), 'b-', lw=1, alpha=0.6, label='Reconstructed')
        subplots[2].plot(linspace, theoretical_pdf / (1 - theoretical_cdf), 'r-', lw=1, alpha=0.6, label='Theoretical')
        subplots[2].set_title('Repair Rate')
        subplots[2].set_xlabel('time (t)')
        subplots[2].set_ylabel('\u03BC(t)')
        subplots[2].legend()
        if distribution[0] == 'EXP':
            subplots[2].set_ylim([0, 1])

    # Third plot Reliability
    if metric == 'Reliability':
        subplots[1].set_title('CDF')
        reliability = calculate_reliability(distribution, linspace)
        theoretical_reliability = calculate_reliability(theoretical_distribution, linspace)
        subplots[2].plot(linspace, reliability, 'b-', lw=1, alpha=0.6, label='Reconstructed')
        subplots[2].plot(linspace, theoretical_reliability, 'r-', lw=1, alpha=0.6, label='Theoretical')
        subplots[2].set_xlabel('time (t)')
        subplots[2].set_ylabel('R(t)')
        subplots[2].legend()
        subplots[2].set_title(metric)

        subplots[3].plot(linspace, pdf/reliability, 'b-', lw=1, alpha=0.6, label='Reconstructed')
        subplots[3].plot(linspace, theoretical_pdf/theoretical_reliability, 'r-', lw=1, alpha=0.6, label='Theoretical')
        subplots[3].set_title('Failure Rate')
        subplots[3].set_xlabel('time (t)')
        subplots[3].set_ylabel('\u03BB(t)')
        subplots[3].legend()
        if distribution[0] == 'EXP':
            subplots[3].set_ylim([0, 1])


    # plt.show()
    plt.tight_layout()
    #if EXPORT_PNG is True:
    fig.savefig(os.getcwd() + '/static/images/' + get_object_name(name) + '_' + metric + '.png')
    #else:
    plt.show(block=False)


def plot_unidentified_distribution_comparison(name, metric, times, theoretical_distribution):
    # Maybe fix inconsistencies with the numbers on the axis
    fig, subplots = setup_fig_subplots(metric)

    linspace = calculate_linspace(theoretical_distribution)
    theoretical_pdf = calculate_pdf(theoretical_distribution, linspace)
    theoretical_cdf = calculate_cdf(theoretical_distribution, linspace)
    theoretical_reliability = calculate_reliability(theoretical_distribution, linspace)

    if metric == 'Reliability':
        sns.distplot(times, hist=True, ax=subplots[0], label='Time to failures')
    if metric == 'Maintainability':
        sns.distplot(times, hist=True, ax=subplots[0], label='Time to repairs')

    x, pdf = subplots[0].lines[0].get_data()
    subplots[0].plot(x, pdf, 'b-', lw=1, alpha=0.6, label='Reconstructed')
    subplots[0].set_xlabel('time (t)')

    subplots[0].plot(linspace, theoretical_pdf, 'r-', lw=1, alpha=0.6, label='Theoretical')
    subplots[0].set_title('PDF')
    subplots[0].set_xlabel('time (t)')
    subplots[0].set_ylabel('P(t)')
    subplots[0].legend()

    # CDF
    sns.kdeplot(times, cumulative=True, ax=subplots[1])
    _, cdf = subplots[1].lines[0].get_data()
    # Second plot CDF and/or Maintainability
    subplots[1].plot(x, cdf, 'b-', lw=1, alpha=0.6, label='Reconstructed')
    subplots[1].plot(linspace, theoretical_cdf, 'r-', lw=1, alpha=0.6, label='Theoretical')
    subplots[1].set_xlabel('time (t)')
    subplots[1].set_ylabel('P(T\u2264t)')
    subplots[1].legend()

    if metric == 'Maintainability':
        subplots[1].set_title('CDF (Maintainability)')
        subplots[2].plot(x, pdf/(1 - cdf), 'b-', lw=1.5, alpha=0.6, label='Reconstructed')
        subplots[2].plot(linspace, theoretical_pdf/theoretical_reliability, 'r-', lw=1, alpha=0.6, label='Theoretical')
        subplots[2].set_title('Repair Rate')
        subplots[2].set_xlabel('time (t)')
        subplots[2].set_ylabel('\u03BC(t)')
        subplots[2].set_ylim([0, 5])
        subplots[2].legend()

    if metric == 'Reliability':
        reliability = 1 - cdf
        subplots[1].set_title('CDF')

        subplots[2].plot(x, reliability, 'b-', lw=1.5, alpha=0.6, label='Reconstructed')
        subplots[2].set_ylim([0, 1.05])
        # Get length of this and see if its the same length as the reliability of the top event
        # Better yet when calculating relability function use the same number of elements as the
        # length of top events time of failure or time of repair.
        subplots[2].plot(linspace, theoretical_reliability, 'r-', lw=1, alpha=0.6, label='Theoretical')
        # For comparing graphs
        # subplots[2].set_xlim([0, 100])
        subplots[2].set_title('Reliability')
        subplots[2].set_xlabel('time (t)')
        subplots[2].set_ylabel('R(t)')
        subplots[2].legend()

        subplots[3].plot(x, pdf/reliability, 'b-', lw=1.5, alpha=0.6, label='Reconstructed')
        subplots[3].plot(linspace, theoretical_pdf/theoretical_reliability, 'r-', lw=1, alpha=0.6, label='Theoretical')
        subplots[3].set_title('Failure Rate')
        subplots[3].set_xlabel('time (t)')
        subplots[3].set_ylabel('\u03BB(t)')
        subplots[3].legend()

    # plt.show()
    plt.tight_layout()
    #if EXPORT_PNG is True:
    fig.savefig(os.getcwd() + '/static/images/' + get_object_name(name) + '_' + metric + '.png')
    plt.show(block=False)


def plot_distributions_comparison(name, metric, distribution, times, theoretical_distribution):
    if distribution[0] == 'UNIDENTIFIED DISTRIBUTION':
        plot_unidentified_distribution_comparison(name, metric, times, theoretical_distribution)
    else:
        plot_identified_distribution_comparison(name, metric, distribution, times, theoretical_distribution)


def plot_identified_distribution_no_comparison(name, metric, distribution, times):
    fig, subplots = setup_fig_subplots(metric)
    fig.suptitle(name)

    linspace = calculate_linspace(distribution)
    pdf = calculate_pdf(distribution, linspace)
    cdf = calculate_cdf(distribution, linspace)
    distribution_name = name_of_distribution(distribution)

    # First plot PDF
    subplots[0].plot(linspace, pdf, 'b-', lw=1, alpha=0.6)
    if times != EMPTY_LIST:
        if metric == 'Reliability':
            subplots[0].hist(times, bins=20, normed=True, histtype='stepfilled', alpha=0.2, label='Time to failures')
        if metric == 'Maintainability':
            subplots[0].hist(times, bins=20, normed=True, histtype='stepfilled', alpha=0.2, label='Time to repairs')
        subplots[0].legend()
    subplots[0].set_title(distribution_name + ' PDF')

    # Second plot CDF
    subplots[1].plot(linspace, cdf, 'b-', lw=1, alpha=0.6)

    if metric == 'Maintainability':
        subplots[1].set_title(distribution_name + ' CDF (Maintainability)')
        subplots[2].plot(linspace, pdf / (1 - cdf), 'b-', lw=1, alpha=0.6)
        subplots[2].set_title('Repair Rate')
        if distribution[0] == 'EXP':
            subplots[2].set_ylim([0, 1])

    # Third plot Reliability
    if metric == 'Reliability':
        subplots[1].set_title(distribution_name + ' CDF')
        reliability = calculate_reliability(distribution, linspace)
        subplots[2].plot(linspace, reliability, 'b-', lw=1, alpha=0.6)
        subplots[2].set_title(metric)
        subplots[3].plot(linspace, pdf / reliability, 'b-', lw=1, alpha=0.6)
        subplots[3].set_title('Failure Rate')
        if distribution[0] == 'EXP':
            subplots[3].set_ylim([0, 1])

    plt.show(block=False)


def plot_unidentified_distribution_no_comparison(name, metric, times):
    fig, subplots = setup_fig_subplots(metric)
    fig.suptitle(name)

    times.sort()
    samples = len(times)
    one_minus_cdf = [1 - (x / samples) for x in range(1, samples + 1)]

    # PDF
    sns.distplot(times, hist=True, ax=subplots[0])
    subplots[0].set_title('PDF')

    # CDF
    sns.kdeplot(times, cumulative=True, ax=subplots[1])
    if metric == 'Maintainability':
        subplots[1].set_title('CDF (Maintainability)')

    if metric == 'Reliability':
        subplots[1].set_title('CDF')

        # Get length of this and see if its the same length as the reliability of the top event
        # Better yet when calculating relability function use the same number of elements as the
        # length of top events time of failure or time of repair.
        subplots[2].plot(times, one_minus_cdf)
        subplots[2].set_ylim([0, 1.05])
        # For comparing graphs
        # subplots[2].set_xlim([0, 100])
        subplots[2].set_title('Reliability')

    # plt.show()
    plt.show(block=False)


def plot_distributions_no_comparison(name, metric, distribution, times):
    if distribution[0] == 'UNIDENTIFIED DISTRIBUTION':
        plot_unidentified_distribution_no_comparison(name, metric, times)
    else:
        plot_identified_distribution_no_comparison(name, metric, distribution, times)


def plot_distributions(name, metric, distribution, times, theoretical_distribution=None):
    if theoretical_distribution is not None:
        plot_distributions_comparison(name, metric, distribution, times, theoretical_distribution)
    else:
        plot_distributions_no_comparison(name, metric, distribution, times)


def plot_arbitrary__distribution_no_compare(name, metric, times):
    # Reliability for now
    # Maybe fix inconsistencies with the numbers on the axis
    fig, subplots = setup_fig_subplots(metric)
    #fig.suptitle(name + '\n\n', fontsize=16)

    # PDF
    subplots[0].set_title('PDF')

    theoretical_cdf = 0
    if metric == 'Reliability':
        sns.distplot(times, hist=True, ax=subplots[0], label='Time to failures')
    if metric == 'Maintainability':
        sns.distplot(times, hist=True, ax=subplots[0], label='Time to repairs')

    x, pdf = subplots[0].lines[0].get_data()
    subplots[0].plot(x, pdf, 'b-', lw=1, alpha=0.6, label='Reconstructed')
    subplots[0].set_xlabel('time (t)')
    subplots[0].set_ylabel('P(t)')
    subplots[0].legend()
    #subplots[0].set_xlim([0, 30])

    # CDF
    sns.kdeplot(times, cumulative=True, ax=subplots[1])
    _, cdf = subplots[1].lines[0].get_data()
    subplots[1].plot(x, cdf, 'b-', lw=1, alpha=0.6, label='Reconstructed')
    subplots[1].set_xlabel('time (t)')
    subplots[1].set_ylabel('P(T\u2264t)')
    subplots[1].legend()

    if metric == 'Maintainability':
        subplots[1].set_title('CDF (Maintainability)')

        subplots[2].plot(x, pdf/(1 - cdf), 'b-', lw=1.5, alpha=0.6, label='Reconstructed')
        subplots[2].set_title('Repair Rate')
        subplots[2].set_xlabel('time (t)')
        subplots[2].set_ylabel('\u03BC(t)')
        subplots[2].set_ylim([0, 5])
        subplots[2].legend()

    if metric == 'Reliability':
        reliability = 1 - cdf
        subplots[1].set_title('CDF')

        # Get length of this and see if its the same length as the reliability of the top event
        # Better yet when calculating relability function use the same number of elements as the
        # length of top events time of failure or time of repair.
        # Reliability
        subplots[2].plot(x, reliability, 'b-', lw=1.5, alpha=0.6, label='Reconstructed')
        subplots[2].set_ylim([0, 1.05])
        subplots[2].legend()
        # For comparing graphs
        # subplots[2].set_xlim([0, 100])
        subplots[2].set_title('Reliability')
        subplots[2].set_xlabel('time (t)')
        subplots[2].set_ylabel('R(t)')

        subplots[3].plot(x, pdf/reliability, 'b-', lw=1.5, alpha=0.6, label='Reconstructed')
        #subplots[2].set_ylim([0, 0.01])

        subplots[3].set_title('Failure Rate')
        subplots[3].set_xlabel('time (t)')
        subplots[3].set_ylabel('\u03BB(t)')
        #subplots[3].set_ylim([0, 0.1])
        subplots[3].legend()
        #subplots[3].set_xlim([300, 325])

    # plt.show()
    plt.tight_layout()
    #if EXPORT_PNG is True:
    fig.savefig(os.getcwd() + '/static/images/' + get_object_name(name) + '_' + metric + '.png')
    #else:
    plt.show(block=False)


def plot_arbitrary_distribution_compare(name, metric, times, linspace, theoretical):
    # Reliability for now
    # Maybe fix inconsistencies with the numbers on the axis
    fig, subplots = setup_fig_subplots(metric)
    #fig.suptitle(name)

    # PDF
    subplots[0].set_title('PDF')

    theoretical_cdf = 0
    if metric == 'Reliability':
        theoretical_cdf = 1 - theoretical
        sns.distplot(times, hist=True, ax=subplots[0], label='Time to failures')
    if metric == 'Maintainability':
        theoretical_cdf = theoretical
        sns.distplot(times, hist=True, ax=subplots[0], label='Time to repairs')
    index = get_index_of_first_zero_in_array(1 - theoretical_cdf)
    theoretical_pdf = differentiate(linspace, theoretical_cdf)

    x, pdf = subplots[0].lines[0].get_data()
    subplots[0].plot(x, pdf, 'b-', lw=1, alpha=0.6, label='Reconstructed')
    subplots[0].plot(linspace[:index], theoretical_pdf[:index], 'r-', lw=1.5, alpha=0.6, label='Theoretical')
    subplots[0].set_xlabel('time (t)')
    subplots[0].set_ylabel('P(t)')
    subplots[0].legend()
    #subplots[0].set_xlim([0, 30])

    # CDF
    sns.kdeplot(times, cumulative=True, ax=subplots[1], label='Reconstructed')
    subplots[1].plot(linspace[:index], theoretical_cdf[:index], 'r-', lw=1.5, alpha=0.6, label='Theoretical')
    subplots[1].legend()
    subplots[1].set_xlabel('time (t)')
    subplots[1].set_ylabel('P(T\u2264t)')
    #subplots[1].set_xlim([0, 30])
    _, cdf = subplots[1].lines[0].get_data()

    if metric == 'Maintainability':
        subplots[1].set_title('CDF (Maintainability)')


        subplots[2].plot(x, pdf/(1 - cdf), 'b-', lw=1.5, alpha=0.6, label='Reconstructed')

        subplots[2].plot(linspace[:index], theoretical_pdf[:index]/(1 - theoretical_cdf[:index]),
                         'r-', lw=1.5, alpha=0.6, label='Theoretical')
        subplots[2].set_title('Repair Rate')
        subplots[2].set_title('Repair Rate')
        subplots[2].set_xlabel('time (t)')
        subplots[2].set_ylabel('\u03BC(t)')
        #subplots[2].set_xlim([0, 50])
        subplots[2].set_ylim([0, 5])
        subplots[2].legend()
        #subplots[3].set_xlim([0, 320])
    if metric == 'Reliability':
        theoretical_reliability = theoretical
        reliability = 1 - cdf
        subplots[1].set_title('CDF')

        #times.sort()
        #samples = len(times)
        #one_minus_cdf = [1 - (x / samples) for x in range(1, samples + 1)]
        #cdf = [(x / samples) for x in range(1, samples + 1)]

        # Get length of this and see if its the same length as the reliability of the top event
        # Better yet when calculating relability function use the same number of elements as the
        # length of top events time of failure or time of repair.
        # Reliability
        subplots[2].plot(x, reliability, 'b-', lw=1.5, alpha=0.6, label='Reconstructed')
        subplots[2].plot(linspace[:index], theoretical_reliability[:index],
                         'r-', lw=1.5, alpha=0.6, label='Theoretical')
        subplots[2].set_ylim([0, 1.05])
        # For comparing graphs
        subplots[2].set_title('Reliability')
        subplots[2].set_xlabel('time (t)')
        subplots[2].set_ylabel('R(t)')
        subplots[2].legend()
        subplots[3].plot(x, pdf/reliability, 'b-', lw=1.5, alpha=0.6, label='Reconstructed')
        #subplots[2].set_ylim([0, 0.01])

        subplots[3].plot(linspace[:index], theoretical_pdf[:index]/theoretical_reliability[:index],
                         'r-', lw=1.5, alpha=0.6, label='Theoretical')
        subplots[3].set_title('Failure Rate')
        subplots[3].set_xlabel('time (t)')
        subplots[3].set_ylabel('\u03BB(t)')
        #subplots[3].set_ylim([0, 0.1])
        subplots[3].legend()
        #subplots[3].set_xlim([300, 325])

    plt.tight_layout()
    #if EXPORT_PNG is True:
    fig.savefig(os.getcwd() + '/static/images/' + get_object_name(name) + '_' + metric + '.png')
    #else:
    plt.show(block=False)


def plot_arbitrary_distribution(name, metric, times, linspace=None, theoretical=None):
    if theoretical is not None:
        plot_arbitrary_distribution_compare(name, metric, times, linspace, theoretical)
    else:
        plot_arbitrary__distribution_no_compare(name, metric, times)


def plot_():
    fig, subplot = plt.subplots(1, 1)
    #lambda_ = distribution[1]
    #scale_ = 1 / lambda_
    linspace = np.linspace(0, 10, 1000)

    rel = (1 - expon.cdf(linspace, expon.pdf(linspace, scale=5)))
    print(rel)

    subplot.plot(linspace, rel)

    plt.show()


def plot_probability_of_failure(time_series, probability_of_failure):
    fig, subplot = plt.subplots(1, 1)
    subplot.plot(time_series, probability_of_failure)
    #subplot.set_title('Probability of component in FAIL state')
    subplot.set_xlabel('time (t)')
    subplot.set_ylabel('Probability')
    #subplot.set_ylim([0, 1])

    plt.show(block=False)


def plot_probability_of_ok(time_series, probability_of_ok):
    fig, subplot = plt.subplots(1, 1)
    subplot.plot(time_series, probability_of_ok)

    #subplot.set_title('Probability of component in OK state (Availability)')
    subplot.set_xlabel('time (t)')
    subplot.set_ylabel('Probability')
    #subplot.set_ylim([0, 1])

    plt.show(block=False)
