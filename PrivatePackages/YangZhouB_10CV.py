import pandas as pd
import numpy as np
import statistics as s
import copy
import time
import pickle
import random

from scipy.spatial.distance import cdist
from scipy.stats import t

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)


class YangZhouB_10CV:

    def __init__(self):
        """Initialise class"""
        self._initialise_objects()

        print("YangZhouB Initialised")

    def _initialise_objects(self):
        """Helper to initialise objects"""
        self.train_x_list = None
        self.train_y_list = None
        self.val_x_list = None
        self.val_y_list = None
        self.tuning_result = None
        self.model = None
        self.parameter_choices = None
        self.hyperparameters = None
        self.feature_n_ningxiang_score_dict = None
        self.non_tuneable_parameter_choices = list()
        self.checked = None
        self.result = None
        self.checked_core = None
        self.been_best = None
        self.been_cruised = None
        self.tuning_result_saving_address = None
        self._parameter_value_map_index = None
        self._seed = 19260817
        self.best_score = -np.inf
        self.best_combo = None
        self.best_clf = None
        self.clf_type = None
        self.n_items = None
        self._core = None
        self._cruise_indices = None
        self._cruise_indices_values = None
        self._cruise_combinations = None
        self._restarts = 0
        self._cruising = True
        self._surrounding_vectors = None
        self._total_combos = None
        self._tune_features = False
        self._up_to = 0
        self._n_cruise_coord = None
        self._cruising_up_to = 0
        self._feature_combo_n_index_map = None
        self.best_model_saving_address = None
        self.pytorch_model = False
        self.optimised_metric = False

    def read_in_data(self, train_x_list, train_y_list, val_x_list, val_y_list):
        """Reads in train validate test data for tuning"""

        assert (
            len(train_x_list) == 10
        ), "Error: length of train_x_list must be 10 for 10-fold CV"
        assert (
            len(train_y_list) == 10
        ), "Error: length of train_y_list must be 10 for 10-fold CV"
        assert (
            len(val_x_list) == 10
        ), "Error: length of val_x_list must be 10 for 10-fold CV"
        assert (
            len(val_y_list) == 10
        ), "Error: length of val_y_list must be 10 for 10-fold CV"

        self.train_x_list = train_x_list
        print("Read in Train X data list")

        self.train_y_list = train_y_list
        print("Read in Train y data list")

        self.val_x_list = val_x_list
        print("Read in Val X data list")

        self.val_y_list = val_y_list
        print("Read in Val y data list")

    def read_in_model(self, model, type, optimised_metric=None, pytorch_model=False):
        """Reads in underlying model object for tuning, and also read in what type of model it is"""

        assert type == "Classification" or type == "Regression"  # check

        self.clf_type = type

        if self.clf_type == "Classification":
            assert optimised_metric in [
                None,
                "accuracy",
                "f1",
                "precision",
                "recall",
                "balanced_accuracy",
                "AP",
                "AUC",
            ], "evaluation_metric for classification must be one of ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy', 'AP', 'AUC']"
        if self.clf_type == "Regression":
            assert optimised_metric in [
                None,
                "r2",
                "rmse",
                "mape",
            ], "evaluation_metric for regression must be one of ['r2', 'rmse', 'mape']"

        if self.clf_type == "Classification":
            self.optimised_metric = (
                "accuracy" if optimised_metric is None else optimised_metric
            )
        elif self.clf_type == "Regression":
            self.optimised_metric = (
                "r2" if optimised_metric is None else optimised_metric
            )

        # record
        self.model = model

        self.pytorch_model = pytorch_model

        print(
            f"Successfully read in model {self.model}, which is a {self.clf_type} model optimising for {self.optimised_metric}"
        )

    def set_hyperparameters(self, parameter_choices):
        """Input hyperparameter choices"""

        self.parameter_choices = parameter_choices
        self._sort_hyperparameter_choices()

        self.hyperparameters = list(parameter_choices.keys())

        # automatically calculate how many different values in each hyperparameter
        self.n_items = [len(parameter_choices[key]) for key in self.hyperparameters]
        self.num_hyperparameters = {
            hyperparameter: len(parameter_choices[hyperparameter])
            for hyperparameter in self.hyperparameters
        }
        self._total_combos = np.prod(self.n_items)

        # automatically setup checked and result arrays and tuning result dataframe
        self._get_checked_and_result_array()
        self._setup_tuning_result_df()

        print("Successfully recorded hyperparameter choices")

    def _sort_hyperparameter_choices(self):
        """Helper to ensure all hyperparameter choice values are in order from lowest to highest"""

        for key in self.parameter_choices:
            tmp = copy.deepcopy(list(self.parameter_choices[key]))
            tmp = self._sort_with_none(tmp)
            self.parameter_choices[key] = tuple(tmp)

    def _sort_with_none(self, lst):
        """Helper to sort hyperparameters with None values"""
        if None in lst:
            no_none_list = [i for i in lst if i is not None]
            no_none_list.sort()
            no_none_list = [None] + no_none_list
            return no_none_list
        lst.sort()
        return lst

    def _get_checked_and_result_array(self):
        """Helper to set up checked and result array"""

        self.checked = np.zeros(shape=self.n_items)
        self.result = np.zeros(shape=self.n_items)
        self.checked_core = np.zeros(shape=self.n_items)
        # strictly for last part of Guidance Algorithm
        self.been_best = np.zeros(shape=self.n_items)
        self.been_cruised = np.zeros(shape=self.n_items)

    def _setup_tuning_result_df(self):
        """Helper to set up tuning result dataframe"""

        tune_result_columns = copy.deepcopy(self.hyperparameters)

        self.tuning_result = pd.DataFrame({col: list() for col in tune_result_columns})

    def set_non_tuneable_hyperparameters(self, non_tuneable_hyperparameter_choice):
        """Input Non tuneable hyperparameter choice"""

        if type(non_tuneable_hyperparameter_choice) is not dict:
            raise TypeError(
                "non_tuneable_hyeprparameters_choice must be dict, please try again"
            )

        # for nthp in non_tuneable_hyperparameter_choice:
        #     if type(non_tuneable_hyperparameter_choice[nthp]) in (set, list, tuple, dict):
        #         raise TypeError('non_tuneable_hyperparameters_choice must not be of array-like type')

        self.non_tuneable_parameter_choices = non_tuneable_hyperparameter_choice

        print("Successfully recorded non_tuneable_hyperparameter choices")

    def set_features(self, ningxiang_output):
        """Input features"""

        if type(ningxiang_output) is not dict:
            raise TypeError("Please ensure NingXiang output is a dict")

        if not self.hyperparameters:
            raise AttributeError(
                "Missing hyperparameter choices, please run .set_hyperparameters() first"
            )

        for feature in list(ningxiang_output.keys())[-1]:
            for i in range(len(self.train_x_list)):
                if feature not in list(self.train_x_list[i].columns):
                    raise ValueError(
                        f"feature {feature} in ningxiang output is not in train_x. Please try again"
                    )

                if feature not in list(self.val_x_list[i].columns):
                    raise ValueError(
                        f"feature {feature} in ningxiang output is not in val_x. Please try again"
                    )

        # sort ningxiang just for safety, and store up
        ningxiang_output_sorted = self._sort_features(ningxiang_output)
        self.feature_n_ningxiang_score_dict = ningxiang_output_sorted

        # activate this switch
        self._tune_features = True

        # update previous internal structures based on first set of hyperparameter choices
        # here used numbers instead of tuples as the values in parameter_choices; thus need another mapping to get map back to the features
        self.parameter_choices["features"] = tuple(
            [i for i in range(len(ningxiang_output_sorted))]
        )
        self._feature_combo_n_index_map = {
            i: list(ningxiang_output_sorted.keys())[i]
            for i in range(len(ningxiang_output_sorted))
        }

        self.hyperparameters = list(self.parameter_choices.keys())

        # automatically calculate how many different values in each hyperparameter
        self.n_items = [
            len(self.parameter_choices[key]) for key in self.hyperparameters
        ]
        self.num_hyperparameters = {
            hyperparameter: len(self.parameter_choices[hyperparameter])
            for hyperparameter in self.hyperparameters
        }
        self._total_combos = np.prod(self.n_items)

        # automatically calculate all combinations and setup checked and result arrays and tuning result dataframe
        self._get_checked_and_result_array()
        self._setup_tuning_result_df()

        print(
            "Successfully recorded tuneable feature combination choices and updated relevant internal structures"
        )

    def _sort_features(self, ningxiang_output):
        """Helper for sorting features based on NingXiang values (input dict output dict)"""

        ningxiang_output_list = [
            (key, ningxiang_output[key]) for key in ningxiang_output
        ]

        ningxiang_output_list.sort(key=lambda x: x[1])

        ningxiang_output_sorted = {x[0]: x[1] for x in ningxiang_output_list}

        return ningxiang_output_sorted

    def _get_core(self):
        """Helper to calculate core"""
        self._core = [int(i / 2) for i in self.n_items]

    def _get_cruise_combinations(self):
        """Helper to cruise combinations"""

        self._get_cruise_indices_values()
        # first get cruise indicies, then use indicies to get combinations
        self._generate_cruise_combinations()

        self._n_cruise_coord = len(self._cruise_combinations)

    def _get_cruise_indices_values(self):
        """Helper to get cruise indices values of each dimension which serves as building blocks for cruise combinations"""

        self._cruise_indices = dict()
        for hyperparameter in self.hyperparameters:
            self._cruise_indices[hyperparameter] = self._get_cruise_indices_1d(
                d_val=self.num_hyperparameters[hyperparameter], max_jump=5
            )

        self._cruise_indices_values = list(self._cruise_indices.values())

    def _get_cruise_indices_1d(self, d_val, max_jump=5):
        """Helper that returns the appropriate cruise indices based on the number of values in dimension. Second argument controls maximum split size, defaulted to 5"""

        assert (
            type(d_val) is int and type(max_jump) is int
        ), "Error: type of input(s) is not int"
        assert max_jump >= 1, "Error: max_jump must be >= 1"

        gap = d_val - 1
        split = (gap - 1) // max_jump

        jump = self._find_gaps(split, gap)

        cruise_indices_1d = self._find_cruise_indices_1d(jump)

        return cruise_indices_1d

    def _find_gaps(self, split, gap):
        """Helper that finds the size of jumps between each element of the final cruise indices, as evenly split as possible with jump size <= 5"""

        if split > 0:
            jump = [gap // (split + 1) for i in range(split + 1)]
            diff = gap - sum(jump)
            if diff:
                for i in range(diff):
                    jump[i] += 1
        else:
            jump = [gap]

        return jump

    def _find_cruise_indices_1d(self, jump):
        """Helper that finds the actual cruise_indices based on gaps"""

        cruise_indices_1d = [0]
        for i in range(len(jump)):
            cruise_indices_1d.append(sum(jump[: i + 1]))

        if cruise_indices_1d == [0, 0]:
            cruise_indices_1d = [0]

        return cruise_indices_1d

    def _generate_cruise_combinations(self):
        """Helper that generates the actual cruise combinations based on cruise indicies"""
        # ALGORITHM: how to generate all combinations of any dimensions given each dimension has different values
        self._cruise_combinations = [[]]
        for i in range(len(self._cruise_indices_values)):

            tmp = copy.deepcopy(self._cruise_combinations)
            self._cruise_combinations = list()

            for x in tmp:

                for k in self._cruise_indices_values[i]:
                    y = copy.deepcopy(x)

                    y.append(k)

                    self._cruise_combinations.append(y)

    def _sort_cruise_combos(self, max_combo):
        """sort the cruise combos based on Euclidean distance from current max"""

        edist = list(cdist([max_combo], self._cruise_combinations).flatten())
        ordered_cruise_combos = [
            (self._cruise_combinations[i], edist[i])
            for i in range(len(self._cruise_combinations))
        ]

        ordered_cruise_combos.sort(reverse=True, key=lambda x: x[1])

        sorted_cruise_combos = [
            ordered_cruise_combos[i][0] for i in range(len(ordered_cruise_combos))
        ]

        return sorted_cruise_combos

    def _get_max_surrounding_mean_sd(self):
        """Helper to get the surrounding mean and sd given the current maximum"""

        best_combo_surrounding_combos = self._get_surrounding_step_combos(
            self.best_combo
        )
        best_combo_surrounding_scores = [self.best_score]
        for combo in best_combo_surrounding_combos:
            best_combo_surrounding_scores.append(self.result[tuple(combo)])

        max_surrounding_mean = s.mean(best_combo_surrounding_scores)
        max_surrounding_sd = s.stdev(best_combo_surrounding_scores)

        return max_surrounding_mean, max_surrounding_sd

    def _cruise_warning_threshold(
        self, max_surrounding_mean, max_surrounding_sd, max_surrounding_n
    ):
        """Helper that gets the warning threshold by (mean of best_combo surrounds - halfwidth)"""

        # use 0.95 (one sided test)
        qt = t.ppf(0.95, max_surrounding_n - 1)
        halfwidth = max_surrounding_sd * qt * 1 / np.sqrt(max_surrounding_n)

        return max_surrounding_mean - halfwidth

    def _CruiseSystem(self):
        """Helper that performs cruising"""

        # get cruise combos in sorted order (furthest away from current max)
        sorted_cruise_combos = self._sort_cruise_combos(self.best_combo)

        # calculate warning threshold by getting max_surrounding_sd first
        max_surrounding_mean, max_surrounding_sd = self._get_max_surrounding_mean_sd()

        warning_threshold = self._cruise_warning_threshold(
            max_surrounding_mean, max_surrounding_sd, len(self._surrounding_vectors)
        )

        # check each cruise combo
        for cruise_combo in sorted_cruise_combos:

            # only search if it hasn't been cruised before (if has then is not an artifect of significance)
            if not self.been_cruised[tuple(cruise_combo)]:

                self._cruising_up_to += 1
                print(
                    f"Cruising Coordrdinate {self._cruising_up_to} of {self._n_cruise_coord}\n"
                )

                # actually been cruised
                self.been_cruised[tuple(cruise_combo)] = 2

                # if above warning threshold, then stop cruise and restart guide
                if self.result[tuple(cruise_combo)] >= warning_threshold:

                    print(f"Cruise suspended due to suspicious case")

                    return cruise_combo

        # if reach here then all cruise indicies checked. can safely say end cruise
        self._cruising = False

        return

    def _get_core(self):
        """Helper to calculate core"""

        self._core = [int(i / 2) for i in self.n_items]

    def _new_combos(self, core, vector):
        """Helper that gets particular COORDINATE using a move in direction of VECTOR from particular CORE"""

        assert len(core) == len(vector)

        new_combo = list()
        for i in range(len(vector)):
            val = core[i] + vector[i]
            # check whether combo is still in the field
            if val >= self.n_items[i] or val < 0:
                return False
            new_combo.append(val)

        return new_combo

    def _xlnx(self, x):
        """Helper that returns x*ln(x), rounding up to next int"""
        y = int(x * np.log(x)) + 1

        return y

    def _find_new_core(self, surrounding_combos, core):
        """Helper that finds new cores - only those candidates with difference between core and treatment < 0.005"""

        new_cores = []
        tmp_new_cores = list()

        for i in range(len(surrounding_combos)):

            diff = self.result[tuple(core)] - self.result[tuple(surrounding_combos[i])]

            if diff <= 0.005:
                if self.checked_core[tuple(surrounding_combos[i])] == 0:
                    tmp_new_cores.append([tuple(surrounding_combos[i]), diff])

        # # sort the tmp new cores list according to p values
        # tmp_new_cores.sort(key = lambda x:x[1])

        # # calculate how many new cores to accept according to the dimension of the grid (x = dim; accept x*ln(x))
        # n_accept = self._xlnx(len(core))

        for i in range(len(tmp_new_cores)):
            # if i >= n_accept:
            #     break

            new_cores.append(tmp_new_cores[i][0])
            self.checked_core[tmp_new_cores[i][0]] = 1

        return new_cores

    def _get_surrounding_step_vectors(self, core):
        """find all horizontal steps"""

        all_steps = list()

        for i in range(len(core)):
            for val in [-1, 1]:
                tmp = [0 for i in range(len(core))]
                tmp[i] = val

                all_steps.append(tmp)

        return all_steps

    def _get_surrounding_step_combos(self, core):
        """find all vectors that are one horizontal steps away from core"""

        surrounding_combos = list()
        for step in self._surrounding_vectors:

            add = 1
            candidate_surrounding_combo = list()

            for i in range(len(core)):
                new_index = core[i] + step[i]
                if new_index < 0 or new_index >= self.n_items[i]:
                    add = 0
                    break
                else:
                    candidate_surrounding_combo.append(new_index)

            if add:
                surrounding_combos.append(candidate_surrounding_combo)

        return surrounding_combos

    def _get_new_cores(self, core):
        """Helper that gets new cores given old cores"""

        # if (should be rare) case where core has been a core before, then skip. For prevention of infinite loops
        # 2 means actual checked core, 1 means appended to checked core list but not checked
        if self.checked_core[tuple(core)] == 2:
            return
        else:
            # actually been checked as a core
            self.checked_core[tuple(core)] = 2

        surrounding_combos = self._get_surrounding_step_combos(core)

        # actually tune the surrounding combos
        for combo in surrounding_combos:

            if self.checked[tuple(combo)] == 0:
                self._train_and_test_combo(combo)
            else:
                self._check_already_trained_best_score(combo)

        # perform welch test and return surrounding combos that should be used as new core
        new_cores = self._find_new_core(surrounding_combos, core)

        return new_cores

    def _GuidanceSystem(self, core):
        """Helper that performs guidance search"""

        if self._restarts == 0:
            print("Guidance: initial round \n")
        else:
            print("Guidance: round", self._restarts, "\n")

        print("\tround", self._restarts, "iteration: ", 0, "\n")

        # first get a surrounding 3^d tuned
        new_cores = self._get_new_cores(core)
        # represent don't need to be added as a to cruised - but not cruised, rather a core
        self.been_cruised[tuple(core)] = 1

        round = 1
        while new_cores:  # while new cores are being added
            print("\tround", self._restarts, "iteration: ", round, "\n")
            round += 1

            old_new_cores = copy.deepcopy(new_cores)
            new_cores = list()

            # for each of the new cores, 'recursively' tune and grab new cores; but each Iteration doesn't end until all cores of current round has been checked
            for new_core in old_new_cores:

                new_new_cores = self._get_new_cores(new_core)

                if not new_new_cores:
                    new_new_cores = []

                self.been_cruised[tuple(core)] == 1

                for new_new_core in new_new_cores:
                    if self.checked_core[tuple(new_new_core)] == 0:
                        new_cores.append(new_new_core)
                        # represent added to checked core - to prevent repeated added to core
                        self.checked_core[tuple(new_new_core)] = 1

        # for current max, get 3^d block. if new max happens to be found, continue to do 3^d block until no new max is found
        # just a cheap way to flesh out the max (the goal of YangZhouB)
        while self.been_best[tuple(self.best_combo)] == 0:

            self.been_best[tuple(self.best_combo)] = 1

            surrounding_combos = self._get_surrounding_step_combos(self.best_combo)
            for combo in surrounding_combos:

                if self.checked[tuple(combo)] == 0:
                    self._train_and_test_combo(combo)
                else:
                    self._check_already_trained_best_score(combo)

        # print information of this round

        print(
            "% Combos Checked Thus Far:",
            int(sum(self.checked.reshape((np.prod(self.n_items))))),
            "out of",
            np.prod(self.n_items),
            "which is",
            f"{np.mean(self.checked).round(8)*100}%",
        )

    def tune(self, key_stats_only=False):
        """Begin tuning"""

        if (
            self.train_x_list is None
            or self.train_y_list is None
            or self.val_x_list is None
            or self.val_y_list is None
        ):
            raise AttributeError(
                " Missing one of the datasets, please run .read_in_data() "
            )

        if self.model is None:
            raise AttributeError(" Missing model, please run .read_in_model() ")

        if self.tuning_result_saving_address is None:
            raise AttributeError(
                "Missing tuning result csv saving address, please run .set_tuning_result_saving_address() first"
            )

        self.key_stats_only = key_stats_only

        print("BEGIN TUNING\n\n")

        # FIRST: get all cruise combinations as well as core, and tune all these
        self._get_core()
        self._get_cruise_combinations()

        first_round_combinations = copy.deepcopy(self._cruise_combinations)
        first_round_combinations.append(self._core)

        random.seed(self._seed)
        random.shuffle(first_round_combinations)

        print("STAGE ZERO: Tune all Cruise combinations\n\n")
        for combo in first_round_combinations:

            if not self.checked[tuple(combo)]:

                self._train_and_test_combo(combo)

            else:
                self._check_already_trained_best_score(combo)

        # SECOND: from the core combo, begin guidance system
        self._surrounding_vectors = self._get_surrounding_step_vectors(self._core)

        print("\n")
        print("STAGE ONE: Begin initial Guidance system\n\n")

        self._restarts = 0
        self._GuidanceSystem(self._core)  # Initial Round of Guidance
        self._restarts += 1

        # THIRD: Recursively Cruise and restart Guide if find a combo that is within halfwidth of mean of best combo surrounds
        print("STAGE TWO: Begin Cruise system\n\n")
        self._cruising_up_to = 0
        self._cruising = True
        while self._cruising:
            suspicious_case_combo = self._CruiseSystem()

            if self._cruising:
                self._GuidanceSystem(tuple(suspicious_case_combo))
                self._restarts += 1

        # FINALLY: Final extensive guidance search around maxes.
        print("FINAL STAGE: Begin final Guidance system\n\n")
        old_best_score = copy.deepcopy(self.best_score)
        self._restarts = "FINAL"

        self._GuidanceSystem(self.best_combo)

        while self.best_score - old_best_score > 0:
            old_best_score = copy.deepcopy(self.best_score)
            self._GuidanceSystem(self.best_combo)

        # Display final information
        self.view_best_combo_and_score()

    def _eval_combo(self, df_building_dict, train_pred, val_pred, i):

        if self.clf_type == "Regression":

            train_score = val_score = train_rmse = val_rmse = train_mape = val_mape = 0

            try:
                train_score = r2_score(self.train_y_list[i], train_pred)
            except:
                pass
            try:
                val_score = r2_score(self.val_y_list[i], val_pred)
            except:
                pass

            try:
                train_rmse = np.sqrt(
                    mean_squared_error(self.train_y_list[i], train_pred)
                )
            except:
                pass
            try:
                val_rmse = np.sqrt(mean_squared_error(self.val_y_list[i], val_pred))
            except:
                pass

            if self.key_stats_only == False:
                try:
                    train_mape = mean_absolute_percentage_error(
                        self.train_y_list[i], train_pred
                    )
                except:
                    pass
                try:
                    val_mape = mean_absolute_percentage_error(
                        self.val_y_list[i], val_pred
                    )
                except:
                    pass

            df_building_dict["Train r2" + f" {i}"] = [np.round(train_score, 6)]
            df_building_dict["Val r2" + f" {i}"] = [np.round(val_score, 6)]
            df_building_dict["Train rmse" + f" {i}"] = [np.round(train_rmse, 6)]
            df_building_dict["Val rmse" + f" {i}"] = [np.round(val_rmse, 6)]

            if self.key_stats_only == False:
                df_building_dict["Train mape" + f" {i}"] = [np.round(train_mape, 6)]
                df_building_dict["Val mape" + f" {i}"] = [np.round(val_mape, 6)]

        elif self.clf_type == "Classification":

            train_score = val_score = train_bal_accu = val_bal_accu = train_f1 = (
                val_f1
            ) = train_precision = val_precision = train_recall = val_recall = (
                train_auc
            ) = val_auc = train_ap = val_ap = 0

            try:
                train_score = accuracy_score(self.train_y_list[i], train_pred)
            except:
                pass
            try:
                val_score = accuracy_score(self.val_y_list[i], val_pred)
            except:
                pass

            try:
                train_f1 = f1_score(self.train_y_list[i], train_pred, average="binary")
            except:
                pass
            try:
                val_f1 = f1_score(self.val_y_list[i], val_pred, average="binary")
            except:
                pass

            try:
                train_precision = precision_score(
                    self.train_y_list[i], train_pred, average="binary"
                )
            except:
                pass
            try:
                val_precision = precision_score(
                    self.val_y_list[i], val_pred, average="binary"
                )
            except:
                pass

            try:
                train_recall = recall_score(
                    self.train_y_list[i], train_pred, average="binary"
                )
            except:
                pass
            try:
                val_recall = recall_score(
                    self.val_y_list[i], val_pred, average="binary"
                )
            except:
                pass

            if self.key_stats_only == False:
                try:
                    train_bal_accu = balanced_accuracy_score(
                        self.train_y_list[i], train_pred
                    )
                except:
                    pass
                try:
                    val_bal_accu = balanced_accuracy_score(self.val_y_list[i], val_pred)
                except:
                    pass
                try:
                    train_auc = roc_auc_score(self.train_y_list[i], train_pred)
                except:
                    pass
                try:
                    val_auc = roc_auc_score(self.val_y_list[i], val_pred)
                except:
                    pass
                try:
                    train_ap = average_precision_score(self.train_y_list[i], train_pred)
                except:
                    pass
                try:
                    val_ap = average_precision_score(self.val_y_list[i], val_pred)
                except:
                    pass

            df_building_dict["Train accuracy" + f" {i}"] = [np.round(train_score, 6)]
            df_building_dict["Val accuracy" + f" {i}"] = [np.round(val_score, 6)]
            df_building_dict["Train f1" + f" {i}"] = [np.round(train_f1, 6)]
            df_building_dict["Val f1" + f" {i}"] = [np.round(val_f1, 6)]
            df_building_dict["Train precision" + f" {i}"] = [
                np.round(train_precision, 6)
            ]
            df_building_dict["Val precision" + f" {i}"] = [np.round(val_precision, 6)]
            df_building_dict["Train recall" + f" {i}"] = [np.round(train_recall, 6)]
            df_building_dict["Val recall" + f" {i}"] = [np.round(val_recall, 6)]

            if self.key_stats_only == False:
                df_building_dict["Train balanced_accuracy" + f" {i}"] = [
                    np.round(train_bal_accu, 6)
                ]
                df_building_dict["Val balanced_accuracy" + f" {i}"] = [
                    np.round(val_bal_accu, 6)
                ]
                df_building_dict["Train AUC" + f" {i}"] = [np.round(train_auc, 6)]
                df_building_dict["Val AUC" + f" {i}"] = [np.round(val_auc, 6)]
                df_building_dict["Train AP" + f" {i}"] = [np.round(train_ap, 6)]
                df_building_dict["Val AP" + f" {i}"] = [np.round(val_ap, 6)]

        return df_building_dict

    def _train_and_test_combo(self, combo):
        """Helper to train and test each combination as part of tune()"""

        combo = tuple(combo)

        clf_list = list()

        for i in range(len(self.train_x_list)):

            params = {
                self.hyperparameters[i]: self.parameter_choices[
                    self.hyperparameters[i]
                ][combo[i]]
                for i in range(len(self.hyperparameters))
            }

            if self._tune_features == True:
                del params["features"]
                tmp_train_x = self.train_x_list[i][
                    list(self._feature_combo_n_index_map[combo[-1]])
                ]
                tmp_val_x = self.val_x_list[i][
                    list(self._feature_combo_n_index_map[combo[-1]])
                ]

                if self.pytorch_model:
                    params["input_dim"] = len(
                        list(self._feature_combo_n_index_map[combo[-1]])
                    )

                # add non tuneable parameters
                for nthp in self.non_tuneable_parameter_choices:
                    params[nthp] = self.non_tuneable_parameter_choices[nthp]
                # initialise object
                clf = self.model(**params)

                params["features"] = [list(self._feature_combo_n_index_map[combo[-1]])]
                params["n_columns"] = len(
                    list(self._feature_combo_n_index_map[combo[-1]])
                )
                params["n_features"] = combo[-1]

            else:
                tmp_train_x = self.train_x_list[i]
                tmp_val_x = self.val_x_list[i]

                if self.pytorch_model:
                    params["input_dim"] = len(list(self.train_x_list[i].columns))

                # add non tuneable parameters
                for nthp in self.non_tuneable_parameter_choices:
                    params[nthp] = self.non_tuneable_parameter_choices[nthp]

                # initialise object
                clf = self.model(**params)

            # get time and fit
            start = time.time()
            clf.fit(tmp_train_x, self.train_y_list[i])
            end = time.time()

            clf_list.append(clf)

            # get predicted labels/values for three datasets
            train_pred = clf.predict(tmp_train_x)
            val_pred = clf.predict(tmp_val_x)

            # get scores and time used
            time_used = end - start

            # build output dictionary and save result

            if i == 0:  # first cv round create saving dict
                df_building_dict = copy.deepcopy(params)

            # get evaluation statistics
            df_building_dict = self._eval_combo(
                df_building_dict, train_pred, val_pred, i
            )

            df_building_dict["Time" + f" {i}"] = [np.round(time_used, 2)]

        df_building_dict[f"Mean Val {self.optimised_metric}"] = [
            np.round(
                np.mean(
                    [
                        df_building_dict[f"Val {self.optimised_metric}" + f" {i}"][0]
                        for i in range(len(self.train_x_list))
                    ]
                ),
                6,
            )
        ]
        df_building_dict[f"Mean Val {self.optimised_metric} Std"] = [
            np.round(
                np.std(
                    [
                        df_building_dict[f"Val {self.optimised_metric}" + f" {i}"][0]
                        for i in range(len(self.train_x_list))
                    ]
                ),
                6,
            )
        ]

        df_building_dict[f"Mean Train {self.optimised_metric}"] = [
            np.round(
                np.mean(
                    [
                        df_building_dict[f"Train {self.optimised_metric}" + f" {i}"][0]
                        for i in range(len(self.train_x_list))
                    ]
                ),
                6,
            )
        ]
        df_building_dict[f"Mean Train {self.optimised_metric} Std"] = [
            np.round(
                np.std(
                    [
                        df_building_dict[f"Train {self.optimised_metric}" + f" {i}"][0]
                        for i in range(len(self.train_x_list))
                    ]
                ),
                6,
            )
        ]

        val_score = df_building_dict[f"Mean Val {self.optimised_metric}"][0]

        for key in df_building_dict:
            if key == "estimators_list":
                df_building_dict[key] = [df_building_dict[key]]

        df_building_dict["Precedence"] = [self._up_to]

        tmp = pd.DataFrame(df_building_dict)

        self.tuning_result = pd.concat([self.tuning_result, tmp])
        self.tuning_result.index = range(len(self.tuning_result))
        self._save_tuning_result()

        # update best score stats
        if val_score > self.best_score:
            self.best_score = val_score
            self.best_clf = clf_list
            self.best_combo = combo

            if self.best_model_saving_address:
                self._save_best_model()

        # update internal governing DataFrames
        self.checked[combo] = 1
        self.result[combo] = val_score

        self._up_to += 1

        print(
            f"""Trained and Tested combination {self._up_to} of {self._total_combos}: {combo}, taking {np.round(time_used,2)} seconds to get val score of {np.round(val_score,4)}
        Current best combo: {self.best_combo} with val score {np.round(self.best_score, 4)}"""
        )

    def _check_already_trained_best_score(self, combo):
        """Helper for checking whether an already trained combo is best score"""

        combo = tuple(combo)

        # update best score stats
        if self.result[combo] > self.best_score:
            self.best_score = self.result[combo]
            self.best_clf = None
            print(f"As new Best Combo {combo} was read in, best_clf is set to None")
            self.best_combo = combo

        print(
            f"""Already Trained and Tested combination {combo}, which had val score of {np.round(self.result[combo],4)}
        Current best combo: {self.best_combo} with val score {np.round(self.best_score, 4)}. 
        Has trained {self._up_to} of {self._total_combos} combinations so far"""
        )

    def _save_tuning_result(self):
        """Helper to export tuning result csv"""

        tuning_result_saving_address_split = self.tuning_result_saving_address.split(
            ".csv"
        )[0]

        self.tuning_result.to_csv(
            f"{tuning_result_saving_address_split}.csv", index=False
        )

    def view_best_combo_and_score(self):
        """View best combination and its validation score"""

        max_val_id = self.tuning_result[f"Mean Val {self.optimised_metric}"].idxmax()

        print(f"Max Val Score: \n", self.best_score)
        print(
            f"Max Val Score Std: \n",
            self.tuning_result.iloc[max_val_id][
                f"Mean Val {self.optimised_metric} Std"
            ],
        )

        print(
            "Best Combo Train Score: \n",
            self.tuning_result.iloc[max_val_id][f"Mean Train {self.optimised_metric}"],
        )
        print(
            f"Best Combo Train Score Std: \n",
            self.tuning_result.iloc[max_val_id][
                f"Mean Train {self.optimised_metric} Std"
            ],
        )

        print(
            "Max Combo Index: \n",
            self.best_combo,
            "out of",
            self.n_items,
            "(note best combo is 0-indexed)",
        )

        final_combo = {
            self.hyperparameters[i]: self.parameter_choices[self.hyperparameters[i]][
                self.best_combo[i]
            ]
            for i in range(len(self.hyperparameters))
        }
        print("Max Combo Hyperparamer Combination: \n", final_combo)

        if self._tune_features:
            print(
                "Max Combo Features: \n",
                self._feature_combo_n_index_map[self.best_combo[-1]],
            )

        print(
            "% Combos Checked:",
            int(sum(self.checked.reshape((np.prod(self.n_items))))),
            "out of",
            np.prod(self.n_items),
            "which is",
            f"{np.mean(self.checked).round(8)*100}%",
        )

    def read_in_tuning_result_df(self, address):
        """Read in tuning result csv and read data into checked and result arrays"""

        BOOL_MAP = {
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
            True: True,
            False: False,
            "True": True,
            "False": False,
            1: True,
            0: False,
            1.0: True,
            0.0: False,
        }

        if self.parameter_choices is None:
            raise AttributeError(
                "Missing parameter_choices to build _parameter_value_map_index, please run set_hyperparameters() first"
            )

        if self.clf_type is None:
            raise AttributeError("Missing clf_type. Please run .read_in_model() first.")

        self.tuning_result = pd.read_csv(address)

        self._up_to = 0

        self._create_parameter_value_map_index()

        # read DataFrame data into internal governing DataFrames of JiaoCheng
        for row in self.tuning_result.iterrows():

            try:

                combo = list()
                for hyperparam in self.hyperparameters:
                    if hyperparam == "features":

                        # reverse two dicts
                        index_n_feature_combo_map = {
                            self._feature_combo_n_index_map[key]: key
                            for key in self._feature_combo_n_index_map
                        }
                        # special input
                        combo.append(
                            index_n_feature_combo_map[
                                tuple(self._str_to_list(row[1]["features"]))
                            ]
                        )

                    else:
                        if type(self.parameter_choices[hyperparam][0]) is bool:
                            combo.append(
                                self._parameter_value_map_index[hyperparam][
                                    BOOL_MAP[row[1][hyperparam]]
                                ]
                            )
                        else:
                            combo.append(
                                self._parameter_value_map_index[hyperparam][
                                    row[1][hyperparam]
                                ]
                            )

                combo = tuple(combo)

                self.result[combo] = row[1][f"Mean Val {self.optimised_metric}"]

                self._up_to += 1

                self.checked[combo] = 1

            except Exception as e:
                print(f"Error message: {str(e)}")
                print("Error Importing this Row:", row)

        print(
            f"Successfully read in tuning result of {len(self.tuning_result)} rows, for {sum(self.checked.reshape((np.prod(self.n_items))))} combos"
        )

    def _str_to_list(self, string):
        """Helper to convert string to list"""

        out = list()
        for feature in string.split(", "):
            out.append(feature.strip("[").strip("]").strip("'"))

        return out

    def _create_parameter_value_map_index(self):
        """Helper to create parameter-value index map"""

        self._parameter_value_map_index = dict()
        for key in self.parameter_choices.keys():
            tmp = dict()
            for i in range(len(self.parameter_choices[key])):
                tmp[self.parameter_choices[key][i]] = i
            self._parameter_value_map_index[key] = tmp

    def set_tuning_result_saving_address(self, address):
        """Read in where to save tuning object"""

        self.tuning_result_saving_address = address
        print("Successfully set tuning output address")

    def set_best_model_saving_address(self, address):
        """Read in where to save best model"""

        self.best_model_saving_address = address
        print("Successfully set best model output address")

    def _save_best_model(self):
        """Helper to save best model as a pickle"""

        best_model_saving_address_split = self.best_model_saving_address.split(
            ".pickle"
        )[0]

        with open(f"{best_model_saving_address_split}.pickle", "wb") as f:
            pickle.dump(self.best_clf, f)
