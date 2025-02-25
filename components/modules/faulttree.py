import collections
import itertools
import importlib
from scipy import integrate
from anytree import RenderTree, LevelOrderIter
from modules import timeseries, distributionfitting as DF
from modules import cutsets, faultTreeReconstruction as ftr
from modules import FTVisualizer, distributionplotting as DP
from modules.gate import Gate


class FaultTree:
    def __init__(self, top_event=None):
        self.top_event = top_event
        self.time_series = {}
        self.top_event_index = 0
        self.basic_event_start_index = 1
        self.cut_sets = []
        self.minimal_cut_sets = []

        if top_event is not None:
            self.number_of_basic_events = len(self.get_basic_events())
        else:
            self.number_of_basic_events = 0

    @staticmethod
    def _get_id_of_basic_event(basic_event):
        # basic_event.name[12:] is to only show numbers, ex: (Basic Event 1) => 1
        # 12 is the first is the number of characters (Basic Event ) takes up
        return int(basic_event.name[12:])

    def _get_gates_reversed(self):
        """
        Get the reverse order of the gates so it starts from the lower level gates and goes to the higher level gates.
        :return: List of gates ordered from the lower level to the higher level.
        """
        gates = []
        for node in LevelOrderIter(self.top_event):
            if type(node) is Gate:
                gates.append(node)
        return gates[::-1]

    def get_basic_events(self):
        """
        Get basic events from Fault Tree Structure, sorted numerically by name.
        :return: Basic events
        """
        basic_events = []
        for node in self.top_event.descendants:
            if node.is_leaf:
                basic_events.append(node)
        basic_events = sorted(basic_events, key=lambda event: event.name)

        return basic_events

    def get_basic_event_(self, basic_event_id):
        for basic_event in self.get_basic_events():
            if basic_event_id == self._get_id_of_basic_event(basic_event):
                return basic_event

        # If can't find it return None.
        return None

    def plot_reliability_distribution_of_basic_event_(self, basic_event_id, theoretical_distribution=None):
        basic_event = self.get_basic_event_(basic_event_id)
        name = basic_event.name
        reliability = 'Reliability'
        rel_dist = basic_event.reliability_distribution
        times = timeseries.calculate_time_to_failures(basic_event.time_series)

        DP.plot_distributions(name, reliability, rel_dist, times, theoretical_distribution)
        # IF CAN'T FIND DISTRIBUTION

    def plot_maintainability_distribution_of_basic_event_(self, basic_event_id, theoretical_distribution=None):
        basic_event = self.get_basic_event_(basic_event_id)
        name = basic_event.name
        maintainability = 'Maintainability'
        main_dist = basic_event.maintainability_distribution
        times = timeseries.calculate_time_to_repairs(basic_event.time_series)

        DP.plot_distributions(name, maintainability, main_dist, times, theoretical_distribution)

    def plot_reliability_distribution_of_top_event(self, linspace=None, theoretical=None):
        times = timeseries.calculate_time_to_failures(self.top_event.time_series)
        rel_dist = DF.determine_distribution(times)
        print('SUGGESTED RELIABILITY DISTRIBUTION FOR TOP EVENT: ' + str(rel_dist))
        DP.plot_arbitrary_distribution('Top Event', 'Reliability', times, linspace, theoretical)
        # DP.plot_weibull('Top Event', 'Reliability', rel_dist, times, theoretical)

    def plot_maintainability_distribution_of_top_event(self, linspace=None, theoretical=None):
        times = timeseries.calculate_time_to_repairs(self.top_event.time_series)
        main_dist = DF.determine_distribution(times)
        print('SUGGESTED MAINTAINABILITY DISTRIBUTION FOR TOP EVENT: ' + str(main_dist))
        DP.plot_arbitrary_distribution('Top Event', 'Maintainability', times, linspace, theoretical)
        # DP.plot_weibull('Top Event', 'Reliability', rel_dist, times, theoretical)

    def plot_probability_of_failure_of_basic_event_(self, basic_event_id):
        basic_event = self.get_basic_event_(basic_event_id)
        print(basic_event)
        DP.plot_probability_of_failure(basic_event.proxel_time_series, basic_event.proxel_probability_of_failure)

    def plot_probability_of_failure_of_top_event(self):
        DP.plot_probability_of_failure(self.top_event.proxel_time_series, self.top_event.proxel_probability_of_failure)

    def plot_probability_of_ok_of_basic_event_(self, basic_event_id):
        basic_event = self.get_basic_event_(basic_event_id)
        print(basic_event)
        DP.plot_probability_of_ok(basic_event.proxel_time_series, basic_event.proxel_probability_of_ok)

    def plot_probability_of_ok_of_top_event(self):
        self.top_event.proxel_probability_of_ok = 1 - self.top_event.proxel_probability_of_failure
        DP.plot_probability_of_ok(self.top_event.proxel_time_series, self.top_event.proxel_probability_of_ok)

    def get_top_event_state(self):
        return self.top_event.state

    def generate_basic_event_time_series(self, size):
        """
        Generate time series for basic events.
        :param size: Size of generated time series.
        :return:
        """
        for basic_event in self.get_basic_events():
            basic_event.generate(size)

    def calculate_time_series(self):
        """
        Calculate time series of all events except basic events.
        :return:
        """
        gates = self._get_gates_reversed()
        for gate in gates:
            gate.evaluate_time_series()

    def calculate_states(self):
        """
        Calculate states of all events except basic events.
        :return:
        """
        gates = self._get_gates_reversed()
        for gate in gates:
            gate.evaluate_states()

    def calculate_reliability_maintainability(self, linspace):
        for basic_event in self.get_basic_events():
            basic_event.calculate_reliability_function(linspace)
            basic_event.calculate_maintainability_function(linspace)

        gates = self._get_gates_reversed()
        for gate in gates:
            gate.evaluate_reliability_maintainability()

    def calculate_proxel_probalities(self, delta_time, simulation_time):
        for basic_event in self.get_basic_events():
            basic_event.calculate_probability_of_failure_using_proxel_based_method(delta_time, simulation_time)

        gates = self._get_gates_reversed()
        for gate in gates:
            gate.evaluate_proxel_probabilities()

    def print_tree(self):
        """
        Render the tree in the console.
        :return:
        """
        print(RenderTree(self.top_event))

    def export_to_dot(self, file_name):
        FTVisualizer.export_to_dot(self, file_name)
        
    def export_to_png(self, file_name):
        FTVisualizer.export_to_png(self, file_name)
        
    def export_time_series(self, file_name):
        """
        Export time series into a file called file_name.
        :param file_name: Name of the file with extension
        :return:
        """
        file = open(file_name, 'w')
        top_event = self.top_event
        for times in top_event.time_series:
            file.write('%s ' % times)
        file.write('\n')
        for basic_event in self.get_basic_events():
            for times in basic_event.time_series:
                file.write('%s ' % times)
            file.write('\n')

        file.close()

    def import_time_series(self, file_name):
        """
        Reads time series from the file given as file_name
        Places the times series of the top event as the first element in
        the dictionary and then all the basic events follow. The function
        also sets the number of basic events found in the file.
        :param file_name: Name of file
        :return:
        """
        file = open(file_name, 'r')
        time_series_dictionary = {}
        index = 0

        lines = file.readlines()
        for line in lines:
            event_time_series = []
            for time in line.split():
                event_time_series.append(float(time))
            time_series_dictionary[index] = event_time_series
            index += 1

        self.time_series = collections.OrderedDict(sorted(time_series_dictionary.items()))

        file.close()

        self.number_of_basic_events = index - 1

    def load_time_series_into_basic_events(self):
        basic_events = self.get_basic_events()
        for basic_event in basic_events:
            basic_event_id = self._get_id_of_basic_event(basic_event)
            basic_event.time_series = self.time_series[basic_event_id]

    def load_states_into_basic_events(self, states):
        basic_events = self.get_basic_events()
        i = 0
        for basic_event in basic_events:
            basic_event.state = states[i]
            i += 1

    def generate_truth_table(self):
        return list(itertools.product([True, False], repeat=self.number_of_basic_events))

    def export_truth_table(self, file_name=None):

        def convert_boolean_to_binary(boolean):
            if boolean is True:
                return 1
            else:
                return 0

        truth_table = self.generate_truth_table()

        file = open(file_name, 'w')

        for basic_event_id in self.basic_events_indexing():
            file.write('%s ' % basic_event_id)
        file.write('TE')
        file.write('\n')

        for row in truth_table:
            binaries = list(map(convert_boolean_to_binary, row))
            for binary in binaries:
                file.write('%s ' % binary)
            self.load_states_into_basic_events(row)
            self.calculate_states()
            binary = convert_boolean_to_binary(self.get_top_event_state())
            file.write('%s ' % binary)
            file.write('\n')

        file.close()

    def determine_distributions_of_basic_events(self):
        basic_events = self.get_basic_events()
        for basic_event in basic_events:
            basic_event.determine_reliability_distribution()
            basic_event.determine_maintainability_distribution()

    def basic_events_indexing(self):
        """
        self.number_of_basic_events + 1, the 1 is needed because go until that number.
        :return: The indexing of the basic events, useful for for loops in other methods to make code cleaner.
        """
        return range(self.basic_event_start_index, self.number_of_basic_events + 1)

    def get_basic_events_time_series(self):
        """
        Get time series of basic events.
        :return: List of time series of the basic events.
        """
        basic_events = []
        for i in self.basic_events_indexing():
            basic_events.append(self.time_series[i])
        return basic_events

    def display_event_time_series(self, display_up_to=-1):
        """
        Display the time series for the top event and the basic events.
        :param display_up_to: Displays time series up to display_up_to, default is -1 to display all.
        :return:
        """
        print('Top Event : ' + str(self.time_series[self.top_event_index][:display_up_to]))
        for i in self.basic_events_indexing():
            print('Basic Event ' + str(i) + ' : ' + str(self.time_series[i][:display_up_to]))

    def calculate_MTTF_of_top_event_from_time_series(self):
        self.top_event.calculate_MTTF_from_time_series()

    def calculate_MTTR_of_top_event_from_time_series(self):
        self.top_event.calculate_MTTR_from_time_series()

    def calculate_MTTF_of_basic_events_from_time_series(self):
        for basic_event in self.get_basic_events():
            basic_event.calculate_MTTF_from_time_series()

    def calculate_MTTR_of_basic_events_from_time_series(self):
        for basic_event in self.get_basic_events():
            basic_event.calculate_MTTR_from_time_series()

    def calculate_inherent_availability_of_basic_events(self):
        for basic_event in self.get_basic_events():
            basic_event.calculate_inherent_availability()

    def calculate_inherent_availability_of_top_event(self):
        self.top_event.calculate_inherent_availability()

    def calculate_operational_availability_of_top_event(self, operating_cycle):
        self.top_event.calculate_operational_availability(operating_cycle)

    def calculate_MTTF_of_top_event_from_reliability_function(self, linspace):
        y_int = integrate.cumtrapz(self.top_event.reliability_function, linspace, initial=0)
        self.top_event.MTTF = y_int[-1]

    def calculate_MTTR_of_top_event_from_maintainability_function(self, linspace):
        # Its 1 - maintainability, since it has to be in the shape of reliability,
        # meaning it will go down to zero
        y_int = integrate.cumtrapz(1 - self.top_event.maintainability_function, linspace, initial=0)
        self.top_event.MTTR = y_int[-1]

    def calculate_MTTF_of_basic_events_from_distributions(self):
        for basic_event in self.get_basic_events():
            basic_event.calculate_MTTF_from_distribution()

    def calculate_MTTR_of_basic_events_from_distributions(self):
        for basic_event in self.get_basic_events():
            basic_event.calculate_MTTR_from_distribution()

    def print_MTTF_MTTR_of_basic_events(self):
        for basic_event in self.get_basic_events():
            print(basic_event.name)
            print('MTTF: ' + str(basic_event.MTTF))
            print('MTTR: ' + str(basic_event.MTTR))

    def print_distributions_of_basic_events(self):
        for basic_event in self.get_basic_events():
            print(basic_event.name)
            print('Reliability: ' + str(basic_event.reliability_distribution))
            print('Maintainability: ' + str(basic_event.maintainability_distribution))

    def calculate_cut_sets(self):
        """
        Calculate the cut sets of the fault tree.
        :return:
        """
        top_event = self.time_series[self.top_event_index]
        basic_events = self.get_basic_events_time_series()
        self.cut_sets = cutsets.calculate_cut_sets(top_event, basic_events)
        print(self.cut_sets)

    def calculate_minimal_cut_sets(self):
        """
        Calculate the minimal cut sets from the cut sets.
        :return:
        """
        self.minimal_cut_sets = cutsets.calculate_minimal_cut_sets(self.cut_sets)
        print(self.minimal_cut_sets)

    def get_minimal_cut_sets(self):
        return self.minimal_cut_sets

    def reconstruct_fault_tree(self, file_name):
        ftr.reconstruct_fault_tree(self.minimal_cut_sets, file_name)

    def load_in_fault_tree(self, module_name):
        fault_tree_creator = importlib.import_module(module_name)
        self.top_event = fault_tree_creator.build_fault_tree()

    def get_length_of_top_event_time_series(self):
        return len(self.time_series[self.top_event_index])

    def check_if_top_event_same(self):
        return self.top_event.time_series == self.time_series[self.top_event_index]
        # Check if recalculated top event time series are the same as the top event times series in the exported file.
