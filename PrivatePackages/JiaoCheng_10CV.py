import pandas as pd
import copy
import time
import numpy as np
import pickle

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


class JiaoCheng_10CV:

    def __init__(self):
        """Initialise class"""
        self._initialise_objects()

        print("JiaoCheng Initialised")

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
        self._feature_combo_n_index_map = None
        self.checked = None
        self.result = None
        self.tuning_result_saving_address = None
        self._up_to = 0
        self._tune_features = False
        self._seed = 2024
        self.best_score = -np.inf
        self.best_combo = None
        self.best_clf = None
        self.clf_type = None
        self.combos = None
        self.n_items = None
        self.hyperparameter_tuning_order = None
        self._tuning_order_map_hp = None
        self._parameter_value_map_index = None
        self._total_combos = None
        self._tune_features = False
        self.hyperparameter_default_values = None
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

        self.param_value_reverse_map = {
            param: {
                self.parameter_choices[param][j]: j
                for j in range(len(self.parameter_choices[param]))
            }
            for param in self.parameter_choices
        }

        self.hyperparameters = list(parameter_choices.keys())

        # automatically calculate how many different values in each hyperparameter
        self.n_items = [len(parameter_choices[key]) for key in self.hyperparameters]
        self._total_combos = np.prod(self.n_items)

        # automatically calculate all combinations and setup checked and result arrays and tuning result dataframe
        self._get_combinations()
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

    def _get_combinations(self):
        """Helper to calculate all combinations"""

        # ALGORITHM

        # recursively append values to get every combination in ordinal/numerical form
        self.combos = [[]]
        for i in range(len(self.n_items)):

            tmp = copy.deepcopy(self.combos)
            self.combos = list()

            for x in tmp:

                for k in range(self.n_items[i]):
                    y = copy.deepcopy(x)

                    y.append(k)

                    self.combos.append(y)

    def _get_checked_and_result_array(self):
        """Helper to set up checked and result array"""

        self.checked = np.zeros(shape=self.n_items)
        self.result = np.zeros(shape=self.n_items)

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

        self.param_value_reverse_map = {
            param: {
                self.parameter_choices[param][j]: j
                for j in range(len(self.parameter_choices[param]))
            }
            for param in self.parameter_choices
        }

        self.hyperparameters = list(self.parameter_choices.keys())

        # automatically calculate how many different values in each hyperparameter
        self.n_items = [
            len(self.parameter_choices[key]) for key in self.hyperparameters
        ]
        self._total_combos = np.prod(self.n_items)

        # automatically calculate all combinations and setup checked and result arrays and tuning result dataframe
        self._get_combinations()
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

    def set_tuning_order(self, order):
        """Input sorting order"""

        if type(order) is not list:
            raise TypeError("order must be a list, please try agian")

        if self.hyperparameters == False:
            raise AttributeError("Please run set_hyperparameters() first")

        if "features" in self.hyperparameters:
            if self._tune_features == False:
                raise AttributeError("Please run set_features() first")

        for hp in order:
            if hp not in self.hyperparameters:
                raise ValueError(
                    f"Feature {hp} is not in self.hyperparameters which was set by set_hyperparameters(); consider reinitiating JiaoCheng or double checking input"
                )

        self.hyperparameter_tuning_order = order
        self._tuning_order_map_hp = {
            self.hyperparameters[i]: i for i in range(len(self.hyperparameters))
        }

    def set_hyperparameter_default_values(self, default_values):
        """Input default values for hyperparameters"""

        if type(default_values) is not dict:
            raise TypeError("default_values must be a dict, please try agian")

        if self.hyperparameters == False:
            raise AttributeError("Please run set_hyperparameters() first")

        if "features" in self.hyperparameters:
            if self._tune_features == False:
                raise AttributeError("Please run set_features() first")

        for hp in default_values:
            if hp not in self.hyperparameters:
                raise ValueError(
                    f"Feature {hp} is not in self.hyperparameter which was set by set_hyperparameters(); consider reinitiating JiaoCheng or double checking input"
                )

            if default_values[hp] not in self.parameter_choices[hp]:
                raise ValueError(
                    f"{default_values[hp]} is not a value to try out in self.hyperparameter which was set by set_hyperparameters(). consider reinitiating JiaoCheng or double checking input"
                )

        self.hyperparameter_default_values = default_values

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

        if self.combos is None:
            raise AttributeError(
                "Missing hyperparameter choices, please run .set_hyperparameters() first"
            )

        if self.tuning_result_saving_address is None:
            raise AttributeError(
                "Missing tuning result csv saving address, please run .set_tuning_result_saving_address() first"
            )

        self.key_stats_only = key_stats_only

        starting_hp_combo = [
            self.param_value_reverse_map[hp][self.hyperparameter_default_values[hp]]
            for hp in self.hyperparameters
        ]  # setup starting combination
        print("\nDefault combo:", starting_hp_combo, "\n")

        round = 1
        # continuously loop through features until converge (combo stays same after a full round)
        continue_tuning = 1
        while continue_tuning:
            print("\nROUND", round)

            # first store previous round's best combo/the starting combo before each round; for comparison at the end
            last_round_starting_hp_combo = copy.deepcopy(starting_hp_combo)

            for hp in self.hyperparameter_tuning_order:  # tune each hp in order
                print(
                    "\nRound",
                    round,
                    "\nHyperparameter:",
                    hp,
                    f"(index: {self._tuning_order_map_hp[hp]})",
                    "\n",
                )

                last_hyperparameter_best_hp_combo = copy.deepcopy(
                    starting_hp_combo
                )  # store last iteration's best combo

                # tune the root combo
                combo = list(copy.deepcopy(starting_hp_combo))
                combo[self._tuning_order_map_hp[hp]] = 0

                for i in range(self.n_items[self._tuning_order_map_hp[hp]]):

                    if not self.checked[tuple(combo)]:
                        self._train_and_test_combo(combo)
                    else:
                        self._check_already_trained_best_score(combo)

                    combo[self._tuning_order_map_hp[hp]] += 1

                # take the best combo after this hyperparameter has been tuned
                starting_hp_combo = copy.deepcopy(self.best_combo)

                if starting_hp_combo == last_hyperparameter_best_hp_combo:
                    print(
                        "\nBest combo after this hyperparameter:",
                        starting_hp_combo,
                        ", NOT UPDATED SINCE LAST HYPERPARAMETER\n",
                    )
                else:
                    print(
                        "\nBest combo after this hyperparameter:",
                        starting_hp_combo,
                        ", UPDATED SINCE LAST HYPERPARAMETER\n",
                    )

            round += 1

            # if after this full round best combo hasn't moved, then can terminate
            if starting_hp_combo == last_round_starting_hp_combo:
                continue_tuning = 0

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

        tuning_result_saving_address_strip = self.tuning_result_saving_address.split(
            ".csv"
        )[0]

        self.tuning_result.to_csv(
            f"{tuning_result_saving_address_strip}.csv", index=False
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
