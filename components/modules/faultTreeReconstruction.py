from numbers import Number
import itertools


EMPTY_LIST = list()


def get_basic_events(minimal_cut_sets):
    """
    Get list of basic events from the list of minimal cut sets.
    :param minimal_cut_sets: List of minimal cut sets
    :return: List of basic events
    """
    basic_events = set()
    for cut_set in minimal_cut_sets:
        for event in cut_set:
            basic_events.add(event)

    return basic_events


def create_event_cut_set_dict(basic_events, minimal_cut_sets):
    """
    Create a dictionary where the keys are index of basic events (index starts from 1) and the value is the
    indices of the minimal cut sets (index starts from 0) which include that basic event.
    :param basic_events: Set which includes the index of basic events (index starts from 1)
    :param minimal_cut_sets: List of minimal cut sets (index starts from 0)
    :return: Returns event dictionary
    """
    event_dictionary = {}

    for basic_event in basic_events:
        minimal_cut_set_ids = set()
        for i in range(len(minimal_cut_sets)):
            if basic_event in minimal_cut_sets[i]:
                minimal_cut_set_ids.add(i)
        event_dictionary[tuple([basic_event])] = minimal_cut_set_ids

    return event_dictionary


def is_sets_identical(set1, set2):
    """
    Checks if the two sets are identical or not.
    :param set1: First set
    :param set2: Second set
    :return: True if sets are identical, False if not
    """
    if set1 == set2:
        return True
    else:
        return False


def is_sets_mutually_exclusive(set1, set2):
    """
    Checks if the two sets are mutually exclusive or not.
    :param set1: First set
    :param set2: Second set
    :return: True if sets are mutually exclusive, False if not
    """
    if set1.isdisjoint(set2):
        return True
    else:
        return False


def print_event_dictionary(event_dictionary):
    """
    Print the event dictionary in easily readable format.
    :param event_dictionary: Event dictionary
    :return:
    """
    for event, sets in event_dictionary.items():
        print('Event: ' + str(event) + ' - Sets: ' + str(sets))


def add_to_event_set(event_set, event):
    """
    Add event to a set of events
    :param event_set: Set of events
    :param event: An event
    :return:
    """
    for e in event:
        event_set.add(e)


def convert_list_of_sets_to_list(sets):
    """
    Convert a list of sets into just a list.
    Ex: [{0, 1}, {0, 2}, {1, 2}] -> [0, 1, 0, 2, 1, 2]
    :param sets: List of sets
    :return: List
    """
    list_ = []
    for set_ in sets:
        for component in set_:
            list_.append(component)

    return list_


def is_n_choose_k_satisfied(parent, children):
    """
    Checks if the parent and children follow a n choose k pattern. The parent set represents
    the n, and the children are the possible solutions of n choose k.
    :param parent: Set
    :param children: List of set
    :return: True if it is n choose k pattern, False if not
    """
    # {0, 1, 2} : [{0, 1}, {0, 2}, {1, 2}]
    # Shortcomings: {0, 1, 2} : [{0, 1}, {0, 2}, {1, 2, 3}] Gives back True as well, that's why
    # the method that calls this method, checks other requirements
    decision = True
    children_list = convert_list_of_sets_to_list(children)
    k = calculate_k_in_voting_gate(parent, children)
    for i in parent:
        if k != children_list.count(i):
            decision = False

    return decision


def compress_identical_sets(event_dictionary):
    """
    Compress sets that are identical in the event dictionary.
    :param event_dictionary: Event dictionary
    :return: Compressed event dictionary
    """
    return compress_sets_that_are('identical', event_dictionary)


def compress_mutually_exclusive_sets(event_dictionary):
    """
    Compress sets that are mutually exclusive in the event dictionary.
    :param event_dictionary: Event dictionary
    :return: Compressed event dictionary
    """
    return compress_sets_that_are('mutually exclusive', event_dictionary)


def compress_interconnected_sets(event_dictionary):
    """
    Compress sets that are 'interconnected' in the event dictionary, this means sets that show
    a n choose k pattern, meaning they are joined by a k/N voting gate.
    :param event_dictionary: Event dictionary
    :return: Compressed event dictionary
    """
    return compress_sets_that_are('interconnected', event_dictionary)


def compress_sets_that_are(of_type, event_dictionary):
    """
    This method is called by the above three functions depending on the compression needed. The first code block
    loops through the events in such a way that the same events aren't compared only to other events. Then
    depending on the type of compression, a new event dictionary is created and returned.
    :param of_type: Type of sets to compress
    :param event_dictionary: Event dictionary
    :return: Compressed event dictionary
    """
    type_of_sets = {}
    for event, sets in event_dictionary.items():
        event_set = set()
        sets_to_add = sets
        add_to_event_set(event_set, event)
        for event_op, sets_op in event_dictionary.items():
            if event is not event_op:

                if of_type == 'identical':
                    if is_sets_identical(sets, sets_op):
                        add_to_event_set(event_set, event_op)

                if of_type == 'mutually exclusive':
                    if is_sets_mutually_exclusive(sets, sets_op):
                        add_to_event_set(event_set, event_op)
                        sets_to_add = sets_to_add.union(sets_op)

                # STRICTER CONDITIONS ARE NEEDED HERE
                if of_type == 'interconnected':
                    if not is_sets_identical(sets, sets_op) and not is_sets_mutually_exclusive(sets, sets_op):
                        if len(sets) == len(sets_op):
                            add_to_event_set(event_set, event_op)
                            sets_to_add = sets_to_add.union(sets_op)

        type_of_sets[tuple(event_set)] = sets_to_add
    return type_of_sets


def expand_event_dictionary(event_dictionary):
    """
    Expand the event dictionary by running the compression methods on them numerous times until the event dictionary
    only holds one element, the top event. Throughout the compression the entire event dictionary is saved to see
    how the algorithm propagated through the events.
    :param event_dictionary: Starting event dictionary holding only the basic events.
    :return: Entire event dictionary of all the events.
    """
    entire_event_dictionary = {}
    entire_event_dictionary.update(event_dictionary)

    print('----------------Start----------------------')
    print_event_dictionary(event_dictionary)
    # ALGORITHM STILL NEEDS TO BE OPTIMIZED!!!!!!!!
    while len(event_dictionary) != 1:
        length = len(event_dictionary)
        event_dictionary = compress_identical_sets(event_dictionary)
        print('----------------And----------------------')
        print_event_dictionary(event_dictionary)
        entire_event_dictionary.update(event_dictionary)

        # K/N VOTING SHOULD PROBABLY GO RIGHT HERE!!!!!!!!!
        event_dictionary = compress_interconnected_sets(event_dictionary)
        print('-----------------k/N---------------------')
        print_event_dictionary(event_dictionary)
        entire_event_dictionary.update(event_dictionary)

        event_dictionary = compress_mutually_exclusive_sets(event_dictionary)
        print('-----------------OR---------------------')
        print_event_dictionary(event_dictionary)
        entire_event_dictionary.update(event_dictionary)

        # If length of event dictionary didn't change, which means it can't be reduced, then break.
        if length == len(event_dictionary):
            entire_event_dictionary = None
            print('---------FAULT TREE CANNOT BE RECONSTRUCTED------')
            break

    print('--------------Entire------------------------')
    print_event_dictionary(entire_event_dictionary)

    return entire_event_dictionary


def give_names_to_events(events):
    """
    Give names to the events so they are labeled with a descriptive name instead of a number or a tuple of numbers.
    The events represented by one number are the basic events.
    The top event is the event represented by the largest tuple of numbers. Which is also the first event in the list
    of events. The rest are intermediate events where the event are represented by a tuple of numbers.
    :param events: List of events
    :return: List of event names
    """
    length = len(events)
    names = ['NULL'] * length

    # Top Event
    names[0] = 'Top Event'
    basic_event_index = 1

    for i in range(1, length):
        if len(events[i]) > 1:
            intermediate_event_name = 'Intermediate Event ' + str(i)
            names[i] = intermediate_event_name
        if len(events[i]) == 1:
            basic_event_name = 'Basic Event ' + str(basic_event_index)
            names[i] = basic_event_name
            basic_event_index += 1

    return names


def convert_list_of_tuples_to_list_of_sets(list_of_tuples):
    """
    Converts a list of tuples to a list of sets. Since sets only have unique elements (no duplicate elements).
    :param list_of_tuples: List of tuples
    :return: List of sets
    """
    list_of_sets = []
    for tuple_ in list_of_tuples:
        set_ = set(tuple_)
        list_of_sets.append(set_)
    return list_of_sets


def reverse_events_and_sets(events, sets):
    """
    Reverse the order of both events and sets according to events. They way it is reversed is the events with more
    element are listed first and then the events with less elements are listed later, however the order of those
    events with certain number of elements aren't changed. A visual example explains the concept better.
    Original events: [{1}, {2}, {3}, {4}, {1, 2}, {3, 4}, {1, 2, 3, 4}]
    Returned events: [{1, 2, 3, 4}, {1, 2}, {3, 4}, {1}, {2}, {3}, {4}]
    :param events: List of sets representing events
    :param sets: List of sets representing cut sets
    :return: Reversed list of events and list of sets
    """
    top_event_length = len(events[-1])
    grouped_events = []
    grouped_sets = []

    # Group events and sets according to the length of events
    for i in range(1, top_event_length + 1):
        i_event_group = []
        i_set_group = []
        for j in range(len(events)):
            if len(events[j]) == i:
                i_event_group.append(events[j])
                i_set_group.append(sets[j])

        grouped_events.append(i_event_group)
        grouped_sets.append(i_set_group)

    # Reverse the grouped events and grouped sets
    grouped_events.reverse()
    grouped_sets.reverse()

    # Resets events and sets
    events = []
    sets = []

    # Unpack the grouped events and grouped sets and append it to the events and sets, respectively
    for i in range(len(grouped_events)):
        if grouped_events[i] != EMPTY_LIST:
            for event_ in grouped_events[i]:
                events.append(event_)
        if grouped_events[i] != EMPTY_LIST:
            for set_ in grouped_sets[i]:
                sets.append(set_)

    return events, sets


def find_children_indices(parent_index, events):
    """
    Find the indices representing the children by looking through the events list of sets
    :param parent_index: Index of parent
    :param events: List of sets representing the events
    :return: The indices of the children of the parent
    """
    parent = events[parent_index]
    children_indices = []
    children = set()

    for i in range(len(events)):
        potential_child = events[i]
        if parent is not potential_child:
            if potential_child.issubset(parent):
                # If potential child already a subset of children, don't add them again
                if not potential_child.issubset(children):
                    # children holds the children so far
                    children.update(potential_child)
                    children_indices.append(i)
            # if children set is identical parent set, all children are found, break from loop and return indices
            if is_sets_identical(children, parent):
                break
    return children_indices


def get_children_from_indices(indices, sets):
    """
    Get children from list of sets that match the indices
    :param indices: List of indexes
    :param sets: List of sets
    :return: List of sets that matched the indices
    """
    list_of_sets = []
    for index in indices:
        list_of_sets.append(sets[index])
    return list_of_sets


def is_children_identical_to_parent(parent, children):
    """
    Checks if the all the children are identical to the parent.
    :param parent: Set representing the parent
    :param children: List of sets representing the children
    :return: True if all the children are identical to the parent, false if not
    """
    decision = False
    counter = 0
    for child in children:
        if is_sets_identical(child, parent):
            counter += 1

    if counter == len(children):
        decision = True

    return decision


def is_children_mutual_exclusive_union_of_parent(parent, children):
    """
    Checks if the children are mutually exclusive to each other and their union is identical to the parent.
    :param parent: Set representing the parent
    :param children: List of sets representing the children
    :return: True if the children are mutually exclusive to each other and the union of the children
    is identical to the parent, false if not.
    """
    sets = set()
    decision = False
    toggle = True

    # Checks if all children are mutually exclusive to each other.
    # Combination is enough, permutation is not needed since it doesnt have to check
    # both, for example, {0, 1}, {2, 3} and {2, 3}, {0, 1}
    for child, child_ in itertools.combinations(children, 2):
        if not is_sets_mutually_exclusive(child, child_):
            toggle = False

    # If children were mutually exclusive to each other,
    # check if the union of the children is identical to the parent.
    if toggle:
        for child in children:
            sets.update(child)
        if is_sets_identical(sets, parent):
            decision = True

    return decision


# Look over function and define it better, still results in incorrect returns
def is_children_n_choose_k_of_parent(parent, children):
    """
    Checks if the children are in a n choose k pattern to the parent.
    :param parent: Set representing the parent
    :param children: List of sets representing the children
    :return:
    """
    decision = True

    # Combination is enough, permutation is not needed since it doesnt have to check
    # both, for example, {0, 1}, {1, 2} and {1, 2}, {0, 1}
    for child, child_ in itertools.combinations(children, 2):
        if is_sets_identical(child, child_):
            decision = False
        if is_sets_mutually_exclusive(child, child_):
            decision = False
        if not child.intersection(child_):
            decision = False
        if not child.issubset(parent):
            decision = False
        if not child_.issubset(parent):
            decision = False
        if len(child) != len(child_):
            decision = False

    if not is_n_choose_k_satisfied(parent, children):
        decision = False

    return decision


def calculate_k_in_voting_gate(parent, children):
    """
    Calculate the k in the n/K voting gate.
    :param parent: Set representing the parent
    :param children: List of sets representing the children
    :return:
    """
    n = len(parent)
    n_simplified = len(children)
    k = len(children[0])

    k_simplified = int(k / (n / n_simplified))

    return k_simplified


def find_relationship(parent_index, children_indices, sets):
    """
    Find the relationship between the children, the gate that connects them.
    :param parent_index: Index of parent
    :param children_indices: Indexes of the children
    :param sets:
    :return: A string stating if the gate is AND or OR, or a number representing
    the k in the k/N VOTING gate.
    """
    parent = sets[parent_index]
    children = get_children_from_indices(children_indices, sets)
    relationship = 'NULL'

    if is_children_identical_to_parent(parent, children):
        relationship = 'AND'
    if is_children_mutual_exclusive_union_of_parent(parent, children):
        relationship = 'OR'
    if is_children_n_choose_k_of_parent(parent, children):
        relationship = calculate_k_in_voting_gate(parent, children)

    return relationship


def get_object_name(name):
    """
    Converts the name string by replacing ' ' (space) with '_' (underscore) and changing the letters to lowercase.
    :param name: A string
    :return: Converted string
    """
    return name.lower().replace(' ', '_')


def get_object_names(names):
    """
    Calls the get_object_name method in a loop for each name in the list of names.
    :param names: List of strings
    :return: List of converted strings
    """
    object_names = []
    for name in names:
        object_names.append(get_object_name(name))
    return object_names


def print_out_fault_tree(event_dictionary):
    """
    Print out the code of the reconstructed fault tree to the console. Copy the code snippet
    and run it to display the fault tree.
    :param event_dictionary: Event dictionary
    :return:
    """

    events, sets = zip(*event_dictionary.items())

    events = convert_list_of_tuples_to_list_of_sets(events)

    events, sets = reverse_events_and_sets(events, sets)

    print('--------------------------------------')
    print('Events: ' + str(events))

    name_of_events = give_names_to_events(events)
    object_event_names = get_object_names(name_of_events)

    print('--------------------------------------')

    print(object_event_names[0] + ' = Event("' + name_of_events[0] + '")')
    for i in range(len(events)):
        if len(events[i]) > 1:
            children = find_children_indices(i, events)
            gate = find_relationship(i, children, sets)
            k = 0
            # print('Gate: ' + gate)
            if isinstance(gate, Number):
                k = gate
                gate = 'VOTING'
            object_gate = get_object_name(gate) + str(i + 1)
            print(object_gate + ' = Gate("' + gate + '", parent=' + object_event_names[i],
                  end="", flush=True)
            if k > 0:
                print(', k=' + str(k) + ')')
            else:
                print(')')
            for j in range(len(children)):
                child = children[j]
                print(object_event_names[child] + ' = Event("' + name_of_events[child] +
                      '", parent=' + object_gate + ')')

    print('--------------------------------------')


def TAB(file):
    file.write('    ')


def NEWLINE(file):
    file.write('\n')


def write_fault_tree_to_file(event_dictionary, file_name):
    """
    Write code of regenerated fault tree into separate file named file_name.
    :param event_dictionary: Event dictionary
    :param file_name
    :return:
    """

    events, sets = zip(*event_dictionary.items())

    events = convert_list_of_tuples_to_list_of_sets(events)

    events, sets = reverse_events_and_sets(events, sets)
    name_of_events = give_names_to_events(events)
    object_event_names = get_object_names(name_of_events)

    file = open(file_name, 'w')

    file.write('from faultTreeContinuous import Event, Gate, FaultTree')
    NEWLINE(file)
    NEWLINE(file)
    NEWLINE(file)

    file.write(str(object_event_names[0]) + ' = Event("' + name_of_events[0] + '")')
    NEWLINE(file)
    for i in range(len(events)):
        if len(events[i]) > 1:
            children = find_children_indices(i, events)
            gate = find_relationship(i, children, sets)
            k = 0
            if isinstance(gate, Number):
                k = gate
                gate = 'VOTING'
            object_gate = get_object_name(gate) + str(i + 1)
            file.write(str(object_gate) + ' = Gate("' + str(gate) + '", parent=' + str(object_event_names[i]))
            if k > 0:
                file.write(', k=' + str(k) + ')')
                NEWLINE(file)
            else:
                file.write(')')
                NEWLINE(file)
            for j in range(len(children)):
                child = children[j]
                file.write(str(object_event_names[child]) + ' = Event("' + str(name_of_events[child]) + '", parent=' +
                           str(object_gate) + ')')
                NEWLINE(file)

    NEWLINE(file)
    file.write('fault_tree = FaultTree(' + object_event_names[0] + ')')
    NEWLINE(file)
    file.write('fault_tree.print_tree()')
    NEWLINE(file)

    file.close()


def export_fault_tree_to_method(event_dictionary, file_name):
    """
    Write a method into a file which reconstructs the fault tree and can be called later to reload into the
    Fault Tree class.
    :param event_dictionary: Event dictionary
    :param file_name
    :return:
    """

    events, sets = zip(*event_dictionary.items())

    events = convert_list_of_tuples_to_list_of_sets(events)

    events, sets = reverse_events_and_sets(events, sets)
    name_of_events = give_names_to_events(events)
    object_event_names = get_object_names(name_of_events)

    file = open(file_name, 'w')

    file.write('from modules.gate import Gate')
    NEWLINE(file)
    file.write('from modules.event import Event')
    NEWLINE(file)
    NEWLINE(file)
    NEWLINE(file)

    file.write('def build_fault_tree():')
    NEWLINE(file)

    TAB(file)
    file.write(object_event_names[0] + ' = Event("' + name_of_events[0] + '")')
    NEWLINE(file)
    for i in range(len(events)):
        if len(events[i]) > 1:
            children = find_children_indices(i, events)
            gate = find_relationship(i, children, sets)
            k = 0
            if isinstance(gate, Number):
                k = gate
                gate = 'VOTING'
            object_gate = get_object_name(gate) + str(i + 1)
            TAB(file)
            file.write(object_gate + ' = Gate("' + gate + '", parent=' + object_event_names[i])
            if k > 0:
                file.write(', k=' + str(k) + ')')
                NEWLINE(file)
            else:
                file.write(')')
                NEWLINE(file)
            for j in range(len(children)):
                child = children[j]
                TAB(file)
                file.write(object_event_names[child] + ' = Event("' + name_of_events[child] + '", parent=' +
                           object_gate + ')')
                NEWLINE(file)

    NEWLINE(file)
    TAB(file)
    file.write('return ' + object_event_names[0])
    NEWLINE(file)
    file.close()


def reconstruct_fault_tree(minimal_cut_sets, file_name):
    """
    Reconstruct the fault tree from the minimal cut sets.
    :param minimal_cut_sets: List of minimal cut sets
    :param file_name
    :return:
    """
    basic_events = get_basic_events(minimal_cut_sets)

    starting_event_dict = create_event_cut_set_dict(basic_events, minimal_cut_sets)

    entire_event_dict = expand_event_dictionary(starting_event_dict)

    print_out_fault_tree(entire_event_dict)

    # write_fault_tree_to_file(entire_event_dict, file_name)
    export_fault_tree_to_method(entire_event_dict, file_name)
