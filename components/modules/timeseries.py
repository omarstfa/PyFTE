import random


is_EVEN = lambda i: i % 2 == 0
is_ODD = lambda i: i % 2 == 1


def _generate_numbers(distribution, length):
    """
    Generate random numbers of length according to the distribution.
    :param distribution: A list which holds information about a distribution and its parameters,
    first element is always the name, the rest depends on the distribution.
    Ex: exponential distribution = distribution = ['EXP', lambda]
    :param length: The length of random numbers to be generated
    :return: A list of random numbers
    """
    name, parameters = distribution  # Unpack distribution name and parameters
    random_numbers = []

    # Handle each distribution type
    if name == 'EXP':
        lambda_ = parameters if isinstance(parameters, (int, float)) else parameters[0]
        for _ in range(length):
            num = random.expovariate(lambda_)
            random_numbers.append(num)
    elif name == 'WEIBULL':
        scale_, shape_ = parameters  # Unpack scale and shape
        for _ in range(length):
            num = random.weibullvariate(scale_, shape_)
            random_numbers.append(num)
    elif name == 'LOGNORM':
        mu_, sigma_ = parameters  # Unpack mu and sigma
        for _ in range(length):
            num = random.lognormvariate(mu_, sigma_)
            random_numbers.append(num)
    elif name == 'NORMAL':
        mu_, sigma_ = parameters  # Unpack mu and sigma
        for _ in range(length):
            num = random.normalvariate(mu_, sigma_)
            random_numbers.append(num)
    else:
        raise ValueError(f"Unsupported distribution: {name}")
    
    return random_numbers



# def _generate_numbers(distribution, length):
#     """
#     Generate random numbers of length according to the distribution.
#     :param distribution: A list which holds information about a distribution and its parameters,
#     first element is always the name, the rest depends on the distribution.
#     Ex: exponential distribution = distribution = ['EXP', lambda]
#     :param length: The length of random numbers to be generated
#     :return: A list of random numbers
#     """
#     name, parameters = distribution
#     random_numbers = []
#     if name == 'EXP':
#         lambda_, = parameters
#         for i in range(length):
#             num = random.expovariate(lambda_)
#             random_numbers.append(num)
#     if name == 'WEIBULL':
#         scale_, shape_ = parameters
#         for i in range(length):
#             num = random.weibullvariate(scale_, shape_)
#             random_numbers.append(num)
#     if name == 'LOGNORM':
#         mu_, sigma_ = parameters
#         for i in range(length):
#             num = random.lognormvariate(mu_, sigma_)
#             random_numbers.append(num)
#     if name == 'NORMAL':
#         mu_, sigma_ = parameters
#         for i in range(length):
#             num = random.normalvariate(mu_, sigma_)
#             random_numbers.append(num)
#     return random_numbers


def _merge_streams(stream1, stream2):
    """
    Merges two streams of data into one, by taking the first element from the first stream,
    then first element from second stream, then second element from first, second from second,
    and so on.
    :param stream1: First stream of data
    :param stream2: Second stream of data
    :return: Returns merged stream of the two streams of data
    """
    stream_total = []
    if len(stream1) == len(stream2):
        for i in range(len(stream1)):
            stream_total.append(stream1[i])
            stream_total.append(stream2[i])
    return stream_total


def _create_time_series(stream):
    """
    Create time series from stream of data by keeping the first element the same,
    but afterwards the next elements will also have the time from the previous
    elements added to it.
    Example: stream = [4, 3, 1, 6]
             time series = [4, 7, 8, 14]
    :param stream: Stream of data
    :return: Time series data
    """
    time_series = [stream[0]]
    for i in range(len(stream) - 1):
        time_series.append(time_series[i] + stream[i + 1])
    return time_series


def generate_time_series(reliability_dist, maintainability_dist, size):
    """
    Generate time series according to distributions
    :param reliability_dist:
    :param maintainability_dist:
    :param size:
    :return:
    """

    time_to_failure = _generate_numbers(reliability_dist, size)
    time_to_repair = _generate_numbers(maintainability_dist, size)

    times = _merge_streams(time_to_failure, time_to_repair)

    time_series = _create_time_series(times)

    return time_series


def calculate_time_differences(time_series):
    """
    Calculate time differences from the time series to get the times. Basically subtracts the previous time from
    the next time to get the difference between the time series times.
    :param time_series: The time series.
    :return: The time differences extracted from the time series.
    """
    time_difference = 0
    times = []
    for time in time_series:
        times.append(time - time_difference)
        time_difference = time
    return times


def calculate_time_to_failures(time_series):
    """
    Calculate the time of failures from the time series.
    :param time_series: The time series.
    :return: List of time of failures.
    """
    times = calculate_time_differences(time_series)
    time_to_failures = []
    for i in range(len(times)):
        if is_EVEN(i):
            time_to_failures.append(times[i])

    return time_to_failures


def calculate_time_to_repairs(time_series):
    """
    Calculate the time of repairs from the time series.
    :param time_series: The time series.
    :return: List of time of repairs.
    """
    times = calculate_time_differences(time_series)
    time_to_repairs = []
    for i in range(len(times)):
        if is_ODD(i):
            time_to_repairs.append(times[i])

    return time_to_repairs


def calculate_mean_time_to_failure(time_series):
    """
    Calculate mean time to failure from the time series.
    :param time_series: The time series.
    :return: Mean time to failure
    """
    time_to_failures = calculate_time_to_failures(time_series)
    return sum(time_to_failures) / len(time_to_failures)


def calculate_mean_time_to_repair(time_series):
    """
    Calculate mean time to repair from the time series.
    :param time_series: The time series.
    :return: Mean time to failure
    """
    time_to_repairs = calculate_time_to_repairs(time_series)
    return sum(time_to_repairs) / len(time_to_repairs)


def get_time_series_up_to(time_series, up_to_time):
    time_series_up_to = []
    for i in range(0, len(time_series)):
        if time_series[i] < up_to_time:
            time_series_up_to.append(time_series[i])
        else:
            break
    return time_series_up_to


def calculate_up_time(time_series):
    return calculate_time_to_failures(time_series)


def calculate_remaining_time(time_series, up_to_time):
    if is_EVEN(len(time_series)):
        return up_to_time - time_series[-1]
    else:
        return 0


def calculate_total_up_time_up_to(time_series, up_to_time):
    if up_to_time <= time_series[-1]:
        time_series_up_to = get_time_series_up_to(time_series, up_to_time)
        up_times = calculate_up_time(time_series_up_to)

        total_up_time = sum(up_times)
        total_up_time += calculate_remaining_time(time_series_up_to, up_to_time)

        return total_up_time
    else:
        return 0


def calculate_operational_availability(time_series, operating_cycle):
    if operating_cycle <= time_series[-1]:
        up_time = calculate_total_up_time_up_to(time_series, operating_cycle)
        return up_time / operating_cycle
    else:
        return 0
