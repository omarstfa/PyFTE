from scipy.stats import expon, norm, weibull_min, lognorm
import math
from anytree import NodeMixin
from modules import timeseries, distributionfitting as DF
import proxel
import numpy as np


DISPLAY_UP_TO = 4
# 'distributions', 'states', 'time_series'
EVENT_PRINT = 'time_series'


class Event(NodeMixin):
    def __init__(self, name, reliability_distribution=None, maintainability_distribution=None, parent=None):
        self.name = name
        self.reliability_distribution = reliability_distribution
        self.maintainability_distribution = maintainability_distribution
        self.reliability_function = None
        self.maintainability_function = None
        self.MTTF = 0
        self.MTTR = 0
        self.availability_inherent = 0
        self.availability_operational = 0
        self.parent = parent
        self.time_series = []
        self.state = None
        self.proxel_time_series = []
        self.proxel_probability_of_failure = []
        self.proxel_probability_of_ok = []

    def generate(self, size):
        """
        Generate time series based on distribution of events.
        :param size: Size of time series to generate; Size right now generates a time series 2 x size,
        since generates both failure times and repair times.
        :return:
        """
        self.time_series = timeseries.generate_time_series(self.reliability_distribution,
                                                           self.maintainability_distribution,
                                                           size)

    def calculate_MTTF_from_time_series(self):
        self.MTTF = timeseries.calculate_mean_time_to_failure(self.time_series)

    def calculate_MTTR_from_time_series(self):
        self.MTTR = timeseries.calculate_mean_time_to_repair(self.time_series)

    def calculate_MTTF_from_distribution(self):
        self.MTTF = DF.calculate_mttf_or_mttr_from_distribution(self.reliability_distribution)

    def calculate_MTTR_from_distribution(self):
        self.MTTR = DF.calculate_mttf_or_mttr_from_distribution(self.maintainability_distribution)

    def calculate_inherent_availability(self):
        self.availability_inherent = self.MTTF/(self.MTTF + self.MTTR)

    def calculate_operational_availability(self, operating_cycle):
        self.availability_operational = timeseries.calculate_operational_availability(self.time_series, operating_cycle)

    def determine_reliability_distribution(self):
        time_of_failures = timeseries.calculate_time_to_failures(self.time_series)
        self.reliability_distribution = DF.determine_distribution(time_of_failures)

    def calculate_reliability_function(self, linspace):
        rel_dist = self.reliability_distribution
        if rel_dist[0] == 'EXP':
            lambda_ = rel_dist[1]
            scale_ = 1 / lambda_
            self.reliability_function = 1 - expon.cdf(linspace, scale=scale_)
        if rel_dist[0] == 'WEIBULL':
            scale = rel_dist[1]
            shape = rel_dist[2]
            self.reliability_function = 1 - weibull_min.cdf(linspace, shape, loc=0, scale=scale)
        if rel_dist[0] == 'NORMAL':
            mu = rel_dist[1]
            sigma = rel_dist[2]
            self.reliability_function = 1 - norm.cdf(linspace, loc=mu, scale=sigma)
        if rel_dist[0] == 'LOGNORM':
            mu = rel_dist[1]
            sigma = rel_dist[2]
            scale = math.exp(mu)
            self.reliability_function = 1 - lognorm.cdf(linspace, sigma, loc=0, scale=scale)

    def determine_maintainability_distribution(self):
        time_of_repairs = timeseries.calculate_time_to_repairs(self.time_series)
        self.maintainability_distribution = DF.determine_distribution(time_of_repairs)

    def calculate_maintainability_function(self, linspace):
        main_dist = self.maintainability_distribution
        if main_dist[0] == 'EXP':
            lambda_ = main_dist[1]
            scale_ = 1 / lambda_
            self.maintainability_function = expon.cdf(linspace, scale=scale_)
        if main_dist[0] == 'WEIBULL':
            scale = main_dist[1]
            shape = main_dist[2]
            self.maintainability_function = weibull_min.cdf(linspace, shape, loc=0, scale=scale)
        if main_dist[0] == 'NORMAL':
            mu = main_dist[1]
            sigma = main_dist[2]
            self.maintainability_function = norm.cdf(linspace, loc=mu, scale=sigma)
        if main_dist[0] == 'LOGNORM':
            mu = main_dist[1]
            sigma = main_dist[2]
            scale = math.exp(mu)
            self.maintainability_function = lognorm.cdf(linspace, sigma, loc=0, scale=scale)

    def calculate_probability_of_failure_using_proxel_based_method(self, delta_time, simulation_time):
        pn = proxel.ProxelNetwork(delta_time, simulation_time, self.reliability_distribution,
                                  self.maintainability_distribution)
        pn.expand_network()
        self.proxel_time_series = np.asarray(pn.time_series)
        self.proxel_probability_of_failure = np.asarray(pn.probability_of_failure)
        self.proxel_probability_of_ok = np.asarray(pn.probability_of_OK)
        print('Proxel time series: ' + str(self.proxel_time_series))
        print('Proxel prob of failure: ' + str(self.proxel_probability_of_failure))

    def __repr__(self):
        if EVENT_PRINT == 'time_series':
            return self.name + ' : ' + str(self.time_series[:DISPLAY_UP_TO])
        if EVENT_PRINT == 'distributions':
            return self.name + ' : ' + str(self.reliability_distribution) + ' : '\
                   + str(self.maintainability_distribution)
        if EVENT_PRINT == 'states':
            return self.name + ' : ' + str(self.state)
        else:
            return self.name
