import collections
import itertools


HIGH = 'HIGH'
LOW = 'LOW'

is_EVEN = lambda i: i % 2 == 0
is_ODD = lambda i: i % 2 == 1


def _get_index_of_number_before(queue, number):
    """
    Gets the index of the number that is right before the number passed in as an argument
    in the method.
    :param queue: List of numbers
    :param number: The number to find the index of its lower neighbor, so the index of the
    number before number
    :return: The index of the previous number or the index of number if it matches a number
    in the queue
    """
    index = -1
    # First checks if number is smaller then the first number in series
    if number < queue[0]:
        index = -1
    # Second checks if number is greater then the last number in series
    elif number > queue[-1]:
        index = len(queue) - 1
    # Then checks the rest of the numbers
    else:
        for i in range(len(queue)):
            if number < queue[i]:
                index = i - 1
                break
    return index


def _get_state_of_event(queue, time):
    """
    Get the state of the event at a certain time. Basically just means
    if the index is odd then the state of the event is UP and if the
    index is even then the state is DOWN.
    :param queue: List of numbers
    :param time: A transition time, technically a number
    :return:
    """
    index = _get_index_of_number_before(queue, time)
    if is_ODD(index):
        return HIGH
    else:
        return LOW


def _get_state_of_basic_events(basic_events, time):
    """
    Get the state of all the basic event at a certain time, basically just
    puts the get_state_of_event method into a for loop to find it for all
    basic events.
    :param basic_events:
    :param time:
    :return: A list indicating what events were UP or DOWN
    """
    status_of_events = []
    for i in range(len(basic_events)):
        basic_event = basic_events[i]
        status_of_events.append(_get_state_of_event(basic_event, time))
    return status_of_events


def _get_all_cut_sets(top_event, basic_events):
    """
    Get all the cut sets (the basic events that cause failures) by checking the state of the basic
    events at all the time of failures of the top event
    :param top_event: Time series indicating the top event's failure and repair times
    :param basic_events: List of time series indicating each basic event's failure and repair times
    :return:
    """
    all_cut_sets = {}
    for i in range(len(top_event)):
        if is_EVEN(i):
            time_of_failure = top_event[i]
            all_cut_sets[time_of_failure] = _get_state_of_basic_events(basic_events, time_of_failure)
    all_cut_sets = collections.OrderedDict(sorted(all_cut_sets.items()))
    return all_cut_sets


def _calculate_unique_cut_sets(all_cut_sets):
    """
    Calculate the unique cut sets from all the cut sets.
    :param all_cut_sets: A dictionary with all cut sets. Keys are the times of failure and the
    values
    :return:
    """
    unique_cut_sets = []
    for time, cut_set in all_cut_sets.items():
        if cut_set not in unique_cut_sets:
            unique_cut_sets.append(cut_set)
    return unique_cut_sets


def _convert_cut_set(symbol_cut_set):
    """
    Converts the cut set which is indicated by HIGH and LOW states at each basic events to
    a list which holds the index of basic events which caused that failure.
    Ex: [HIGH, LOW, LOW, LOW] -> [2, 3, 4]
    :param symbol_cut_set: Cut set indicated by HIGH and LOW states
    :return: List of basic events that caused failure of top event
    """
    cut_set = []
    for i in range(len(symbol_cut_set)):
        if symbol_cut_set[i] == LOW:
            cut_set.append(i + 1)
    return cut_set


def calculate_cut_sets(top_event, basic_events):
    """
    Calculate cut sets from the failure times of top events and the state of basic events
    :param top_event: Time series indicating the top event's failure and repair times
    :param basic_events: List of time series indicating each basic event's failure and repair times
    :return: The cut sets of the fault tree
    """
    cut_sets = []
    all_cut_sets = _get_all_cut_sets(top_event, basic_events)
    symbol_cut_sets = _calculate_unique_cut_sets(all_cut_sets)
    for symbol_cut_set in symbol_cut_sets:
        cut_sets.append(_convert_cut_set(symbol_cut_set))
    return cut_sets


def _is_cut_set_reducible(cut_set, cut_set_under_test):
    """
    Decides if the cut_set_under_test is reducible into the cut_set. Looks if all the basic events in cut_set are
    also in cut_set_under_test. If all of them are inside cut_set_under_test then cut_set_under_test is reducible.
    :param cut_set: Cut set
    :param cut_set_under_test: Cut set under test to see if it can be reduced to cut_set
    :return: If cut_set_under_test is reducible or not
    """
    counter = 0
    for basic_event in cut_set:
        if basic_event in cut_set_under_test:
            counter += 1
    if counter == len(cut_set):
        return True
    else:
        return False


def calculate_minimal_cut_sets(cut_sets):
    """
    Calculates the minimal cut sets from the cut sets.
    :param cut_sets: Cut sets
    :return: Minimal cut sets
    """
    cut_sets_for_removal = []
    # Get cut_set from cut_sets
    # Get another cut_set(_under_test) from cut_sets
    # Check if the two cut_set are not the same
    for cut_set, cut_set_under_test in itertools.permutations(cut_sets, 2):
        # Check if cut_set_under_test can be reduced to cut_set
        if _is_cut_set_reducible(cut_set, cut_set_under_test):
            # If can be reduced set up for removal
            # Check if cut_set_under_test is not marked for removal yet
            if cut_set_under_test not in cut_sets_for_removal:
                # If not marked for removal yet, added it for removal
                cut_sets_for_removal.append(cut_set_under_test)
    # Remove the cut_set from cut_sets that are marked for removal
    for cut_set in cut_sets_for_removal:
        cut_sets.remove(cut_set)
    return cut_sets
