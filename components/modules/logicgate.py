from numbers import Number
import collections


HIGH = 'HIGH'
LOW = 'LOW'

is_EVEN = lambda i: i % 2 == 0
is_ODD = lambda i: i % 2 == 1


def _get_subsequent_edge_type(data_stream, time):
    """
    Get a three element list, showing if the transition of the time is from up to down or down to up.
    The first element is the edge before the time transition, the second element is the time transition,
    the third element is the edge after the time transition. Function finds index of of the time, and if
    it's in an even index its U then D, if odd index its D then U
    :param data_stream: Stream of data with transition times
    :param time: The time of transition where the type of transition is needed
    :return: A list of three elements, the first elements is the edge before transition, second is the the
    transition time, third is the edge after transition. Ex: ['U', 1.5, 'D']
    """
    edges = []
    index = 0
    for i in range(len(data_stream)):
        if data_stream[i] == time:
            index = i
            break
    if is_EVEN(index):
        edges.append(HIGH)
        edges.append(time)
        edges.append(LOW)
    else:
        edges.append(LOW)
        edges.append(time)
        edges.append(HIGH)
    return edges


def _create_data_stream_dict(a_data_streams):
    """
    Create a sorted dictionary of all the transition times in all the data streams.
    :param a_data_streams: List of data streams, one data stream is a list of time transitions.
    :return: A sorted dictionary where the keys are the transition times the values are the index
    of which data stream it belongs to.
    """
    data_dict = {}
    for i in range(len(a_data_streams)):
        length = len(a_data_streams[i])
        data_stream = a_data_streams[i]
        for j in range(length):
            data_dict[data_stream[j]] = i
    ordered_dict = collections.OrderedDict(sorted(data_dict.items()))
    return ordered_dict


def _initialize_coded_data_streams(size, num_of_streams):
    """
    Create empty data_streams which hold enough space for all the transition times plus the
    corresponding edge transitions. The first element in each stream is initialized to 'U',
    since all the streams start with from high.
    :param size: Size of data stream dictionary, which holds all the time transitions from
    all the data streams
    :param num_of_streams: Number of data streams
    :return: Empty list of streams which are initialized to hold the time transitions and
    edge type
    """
    coded_data_streams = []
    for i in range(num_of_streams):
        i_coded_data_stream = [None] * (2 * size + 1)
        i_coded_data_stream[0] = HIGH
        coded_data_streams.append(i_coded_data_stream)
    return coded_data_streams


def _initialize_result_data_stream(size):
    """
    Initialize empty data stream to save result
    :param size: Size of data stream dictionary, which holds all the time transitions from
    all the data streams
    :return: Return empty list to hold results
    """
    empty_list = [None] * (size * 2 + 1)
    return empty_list


def _fill_out_missing_entries(coded_data_streams):
    """
    Fill out missing entries in edge_transition_coded_data_streams by looking at previous element.
    :param coded_data_streams:
    :return: Nothing, since the input is changed
    """
    for i in range(len(coded_data_streams)):
        coded_data_stream_i = coded_data_streams[i]
        length = len(coded_data_stream_i)
        for j in range(length):
            if coded_data_stream_i[j] is None:
                coded_data_stream_i[j] = coded_data_stream_i[j - 1]


def _is_all_up(slice_):
    """
    Check if all elements in the slice are UP, in the high state
    :param slice_: A 'slice' of data holding the state of the data streams at a certain time
    :return: Boolean indicating if all elements are in the high state.
    """
    decision = False
    counter = 0
    for element in slice_:
        if element == HIGH:
            counter += 1
    if counter == len(slice_):
        decision = True
    return decision


def _is_all_down(slice_):
    """
    Check if all elements in the slice are DOWN, in the low state.
    :param slice_: A 'slice' of data holding the state of the data streams at a certain time
    :return: Boolean indicating if all elements are in the low state.
    """
    decision = False
    counter = 0
    for element in slice_:
        if element == LOW:
            counter += 1
    if counter == len(slice_):
        decision = True
    return decision


def _is_at_least_k_down(slice_, k):
    """
    Check if there is at least k number of LOW states in slice_
    :param slice_: A 'slice' of data holding the state of the data streams at a certain time
    :param k: There must be at least k number of LOW states in this slice
    :return: Boolean indicating if there is at least k number of LOW states
    """
    decision = False
    counter = 0
    for element in slice_:
        if element == LOW:
            counter += 1
    if counter >= k:
        decision = True
    return decision


def _is_number_in_slice(slice_):
    """
    Function to check if at least one element is a number, meaning there is a transition there
    :param slice_: A 'slice' of data holding the state of the data streams at a certain time
    :return: Boolean indicating if there is a number in the slice.
    """
    decision = False
    for element in slice_:
        if isinstance(element, Number):
            decision = True
            break
    return decision


def _get_number_in_slice(slice_):
    """
    Function to get number, the transition time, in that slice.
    :param slice_: A 'slice' of data holding the state of the data streams at a certain time
    :return: The number, the transition time, in the slice
    """
    number = 0
    for element in slice_:
        if isinstance(element, Number):
            number = element
            break
    return number


def _and_evaluate_slice(slice_):
    """
    Evaluate slice according to AND gate
    :param slice_: A 'slice' of data holding the state of the data streams at a certain time
    :return: Result of slice according to AND gate
    """
    if _is_all_up(slice_):
        element = HIGH
    elif _is_number_in_slice(slice_):
        element = _get_number_in_slice(slice_)
    else:
        element = LOW
    return element


def _or_evaluate_slice(slice_):
    """
    Evaluate slice according to AND gate
    :param slice_: A 'slice' of data holding the state of the data streams at a certain time
    :return: Result of slice according to OR gate
    """
    if _is_all_down(slice_):
        element = LOW
    elif _is_number_in_slice(slice_):
        element = _get_number_in_slice(slice_)
    else:
        element = HIGH
    return element


def _k_voting_evaluate_slice(slice_, k):
    """
    Evaluate slice according to k/N VOTING
    :param slice_: A 'slice' of data holding the state of the data streams at a certain time
    :param k: There must be at least k number of LOW states in this slice.
    :return: Result of slice according to k/N VOTING gate
    """
    if _is_number_in_slice(slice_):
        element = _get_number_in_slice(slice_)
    elif _is_at_least_k_down(slice_, k):
        element = LOW
    else:
        element = HIGH

    return element


def _evaluate_transitions(gate, coded_data_streams, length):
    """
    Evaluate edge_transition_coded_data_streams by taking each index at a time and creating a 'slice'
    from the streams that happen at the same instant.
    i is the index of the stream, j is the index of the slice
    :param gate: The logic gate used to evaluate the slice, or in case of voting gate, the k value.
    :param coded_data_streams: List of data streams with states and transition times
    :param length:
    :return:
    """
    result_list = _initialize_result_data_stream(length)
    for i in range(len(coded_data_streams[0])):
        slice_ = []
        for j in range(len(coded_data_streams)):
            slice_.append(coded_data_streams[j][i])
        if gate == 'AND':
            result_list[i] = _and_evaluate_slice(slice_)
        if gate == 'OR':
            result_list[i] = _or_evaluate_slice(slice_)
        if isinstance(gate, Number):
            result_list[i] = _k_voting_evaluate_slice(slice_, gate)

    return result_list


def _filter_stream(stream):
    """
    Filter out the UP and DOWN elements from data stream just to get transition times
    :param stream: Stream of data
    :return: List of the time transitions only
    """
    result_list = []
    for i in range(len(stream)):
        if isinstance(stream[i], Number):
            if stream[i - 1] != stream[i + 1]:
                result_list.append(stream[i])
    return result_list


def evaluate_time_series(gate, streams):
    """
    Evaluate the streams according to the gate, gate can be 'AND' or 'OR'
    Returns a data stream containing the edge transition times after the
    logic gate.
    :param gate: The logic gate used for evaluation, or in case of voting gate, the k value.
    :param streams: List of data streams
    :return: Resulting list of transition times according the the logic gate
    """
    sorted_stream = _create_data_stream_dict(streams)

    edge_transition_coded_data_streams = _initialize_coded_data_streams(len(sorted_stream), len(streams))

    '''
    Fill out edge_transition_coded_data_streams according to the transition times in
    increasing order. While keeping space between entries if another stream has entries
    in those spots. Counter is not increased after edge[2] since it would cause duplicate
    entries.
    '''
    counter = 0
    for time, stream_index in sorted_stream.items():
        edge = _get_subsequent_edge_type(streams[stream_index], time)
        coded_data_stream_i = edge_transition_coded_data_streams[stream_index]
        coded_data_stream_i[counter] = edge[0]
        counter += 1
        coded_data_stream_i[counter] = edge[1]
        counter += 1
        coded_data_stream_i[counter] = edge[2]

    _fill_out_missing_entries(edge_transition_coded_data_streams)

    result = _evaluate_transitions(gate, edge_transition_coded_data_streams, len(sorted_stream))

    result = _filter_stream(result)

    return result


def evaluate_boolean_logic(gate, boolean_values):
    result = None

    if gate == 'AND':
        result = all(boolean_values)
    if gate == 'OR':
        result = any(boolean_values)
    if isinstance(gate, Number):
        if gate <= boolean_values.count(False):
            result = False
        else:
            result = True
    return result
