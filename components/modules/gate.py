import random
from functools import reduce
from anytree import NodeMixin
from modules import logicgate


class Gate(NodeMixin):
    def __init__(self, name, parent=None, k=None):
        """
        ID is only for visualization purposes so each gate has a unique identifier when using DOT language.
        :param name:
        :param parent:
        :param k:
        """
        self.name = name
        self.id = random.randint(0, 1000000)
        self.parent = parent
        self.k = k

    def k_N_voting(self, k, N, input_probabilities):
        """
        Pseudocode on page 38 of Fault tree analysis: A survey of the state-of-the-art
        in modeling, analysis and tools
        Using recursion to calculate reliability when the gate is k/N Voting

        :param k:
        :param N:
        :param input_probabilities:
        :return:
        """
        if k == 0:
            return 1
        if k == N:
            return reduce(lambda x, y: x * y, input_probabilities)

        result = input_probabilities[0] * self.k_N_voting(k - 1, N - 1, input_probabilities[1:]) + \
                 (1 - input_probabilities[0]) * self.k_N_voting(k, N - 1, input_probabilities[1:])
        return result

    def get_number_of_children(self):
        return len(self.children)

    def determine_fault_logic(self):
        fault_logic = None
        if self.name == 'AND':
            fault_logic = 'OR'
        if self.name == 'OR':
            fault_logic = 'AND'
        if self.name == 'VOTING':
            fault_logic = self.k

        return fault_logic

    def evaluate_time_series(self):
        """
        Evaluate the gate inputs and modify the gate output. Based on the gate type and gate input time series,
        calculate time series for gate output.
        Note: Fault tree gate logic is opposite, since it checks for failures not successes.
        :return:
        """
        data_streams = []
        for child in self.children:
            data_streams.append(child.time_series)

        fault_logic = self.determine_fault_logic()

        self.parent.time_series = logicgate.evaluate_time_series(fault_logic, data_streams)

    def evaluate_states(self):
        states = []

        for child in self.children:
            states.append(child.state)

        fault_logic = self.determine_fault_logic()

        self.parent.state = logicgate.evaluate_boolean_logic(fault_logic, states)

    def evaluate_reliability_maintainability(self):

        reliabilities = 1
        maintainabilities = 1

        if self.name == 'AND':
            for child in self.children:
                reliabilities *= (1 - child.reliability_function)
                maintainabilities *= (1 - child.maintainability_function)

            self.parent.reliability_function = 1 - reliabilities
            self.parent.maintainability_function = 1 - maintainabilities

        if self.name == 'OR':
            for child in self.children:
                reliabilities *= child.reliability_function
                maintainabilities *= child.maintainability_function

            self.parent.reliability_function = reliabilities
            self.parent.maintainability_function = maintainabilities

        if self.name == 'VOTING':
            child_reliability_functions = []
            child_maintainability_functions = []
            N = len(self.children)
            for child in self.children:
                child_reliability_functions.append(child.reliability_function)
                child_maintainability_functions.append(child.maintainability_function)

            self.parent.reliability_function = self.k_N_voting(self.k, N, child_reliability_functions)
            self.parent.maintainability_function = self.k_N_voting(self.k, N, child_maintainability_functions)

    def evaluate_proxel_probabilities(self):
        probabilities = 1
        if self.name == 'AND':
            for child in self.children:
                self.parent.proxel_time_series = child.proxel_time_series
                probabilities *= child.proxel_probability_of_failure

            self.parent.proxel_probability_of_failure = probabilities

        if self.name == 'OR':
            for child in self.children:
                self.parent.proxel_time_series = child.proxel_time_series
                probabilities *= (1 - child.proxel_probability_of_failure)

            self.parent.proxel_probability_of_failure = 1 - probabilities

        if self.name == 'VOTING':
            children_proxel_probabilities = []
            N = len(self.children)
            for child in self.children:
                self.parent.proxel_time_series = child.proxel_time_series
                children_proxel_probabilities.append(child.proxel_probability_of_failure)

            self.parent.proxel_probability_of_failure = self.k_N_voting(self.k, N, children_proxel_probabilities)

    def __repr__(self):
        if self.k is None:
            return self.name
        else:
            return str(self.k) + '/' + str(self.get_number_of_children()) + ' ' + self.name
