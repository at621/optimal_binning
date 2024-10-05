from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree

# import utils as ut

# from utils import memory_time_decorator
import time
from functools import wraps
import tracemalloc


def memory_time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.time()

        results = func(*args, **kwargs)

        end_time = time.time()

        current, malloc_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = end_time - start_time

        print(f"Function {func.__name__} peaked at {malloc_peak / 1024**2:.2f} MB.")
        print(f"Function {func.__name__} executed in {execution_time:.2f} seconds.")

        return results

    return wrapper


class BinningOptimizer:
    def __init__(self):
        self.input_df = None

    def get_dataframe(self):
        return self.input_df

    def calculate_iv(self, events, non_events, total_events, total_non_events):
        # Calculate the percentage of events and non-events
        event_rate = events / total_events
        non_event_rate = non_events / total_non_events

        # Add a constant to the denominator to avoid division by zero
        event_rate = np.where(event_rate == 0, 0.000001, event_rate)

        WoE = np.log(non_event_rate / event_rate)
        IV = (non_event_rate - event_rate) * WoE

        return IV

    def test_proportions(self, event_1, non_event_1, event_2, non_event_2):
        total_1 = event_1 + non_event_1
        total_2 = event_2 + non_event_2
        proportion_1 = event_1 / total_1
        proportion_2 = event_2 / total_2
        pooled_proportion = (event_1 + event_2) / (total_1 + total_2)

        z = (proportion_1 - proportion_2) / np.sqrt(
            pooled_proportion * (1 - pooled_proportion) * (1 / total_1 + 1 / total_2)
        )
        return abs(z)

    def individual_bin_stats(self, events, non_events):
        """Create stats for all purposes"""
        n = len(events)
        stats = {}

        counts = events + non_events
        total_events = events.sum()
        total_non_events = (counts - events).sum()

        for i in range(n + 1):
            for j in range(i, n):
                # Calculate the combined events and non-events for each bin combination
                combined_events = events[i : j + 1].sum()
                combined_non_events = (counts[i : j + 1] - events[i : j + 1]).sum()

                # Calculate IV for this bin combination
                iv = self.calculate_iv(
                    combined_events, combined_non_events, total_events, total_non_events
                )
                default_rate = combined_events / (combined_events + combined_non_events)

                stats[(i + 1, j + 1)] = [
                    iv,
                    combined_events,
                    combined_non_events,
                    total_events,
                    total_non_events,
                    default_rate,
                ]

        return stats

    def bin_combo_stats(self, stats_b):
        result_stats = {}
        for a in stats_b:
            for b in stats_b:
                if a[1] == b[0] - 1:
                    e1 = stats_b[a][1]
                    ne1 = stats_b[a][2]
                    e2 = stats_b[b][1]
                    ne2 = stats_b[b][2]

                    # Calculate mononicity of consecutive default rates
                    default_rate_1 = stats_b[a][5]
                    default_rate_2 = stats_b[b][5]
                    trend = 1 if default_rate_1 > default_rate_2 else 0

                    # CaLculate z-score of adjacent bins
                    z_score = self.test_proportions(e1, ne1, e2, ne2)

                    # Store results
                    result_stats[a, b] = [z_score, trend]

        return result_stats

    @memory_time_decorator
    def create_pools(self, min_bins=5, max_bins=20, monotonical=True):
        """Finds optimal binning given potential cutoffs"""

        n = len(self.events)

        # Create the model.
        model = cp_model.CpModel()

        # Variables
        x = {}
        iv = {}
        bins = []
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                # x[i, j] is a boolean that is true if included in the combination
                x[i, j] = model.NewBoolVar(f"x[{i},{j}]")

        # Create supporting data
        bin_data = self.individual_bin_stats(self.events, self.non_events)
        iv = {key: val[0] for key, val in bin_data.items()}

        # Create bin options that should be excluded from consideration
        excluded_bin_combos = self.bin_combo_stats(bin_data)

        # Constraints
        for i in range(1, n + 1):
            # Each bin i should be included in exactly one tuple.
            model.Add(
                sum(x[j, k] for j in range(1, i + 1) for k in range(i, n + 1)) == 1
            )

        # Reject bins if the z-score is below the desired threshold (todo: check signs)
        for key, value in excluded_bin_combos.items():
            if value[0] < 1.0:
                ((i1, j1), (i2, j2)) = key
                model.Add(x[i1, j1] + x[i2, j2] <= 1)

        # Reject bins if the default rates are not monotonically increasing
        if monotonical:
            for key, value in excluded_bin_combos.items():
                if value[1] == 1:
                    ((i1, j1), (i2, j2)) = key
                    model.Add(x[i1, j1] + x[i2, j2] <= 1)

        # Maximum number of bins (max_bins)
        model.Add(
            sum(x[i, j] for i in range(1, n + 1) for j in range(i, n + 1)) <= max_bins
        )

        # Minimum number of bins (min_bins)
        model.Add(
            sum(x[i, j] for i in range(1, n + 1) for j in range(i, n + 1)) >= min_bins
        )

        # Objective function
        model.Maximize(
            sum(iv[i, j] * x[i, j] for i in range(1, n + 1) for j in range(i, n + 1))
        )

        # Create a solver and solve.
        solver = cp_model.CpSolver()
        solution_printer = VarArraySolutionPrinter(x)
        status = solver.SolveWithSolutionCallback(model, solution_printer)

        if status == cp_model.OPTIMAL:
            print(f"Total IV = {solver.ObjectiveValue():.2f}, optimal solution")
            for i in range(1, n + 1):
                for j in range(i, n + 1):
                    if solver.Value(x[i, j]) == 1:
                        bins.append([i - 1, j - 1])

        # Creating a dictionary where the key is the element number
        bin_dict = {index: pair for index, pair in enumerate(bins)}
        self.add_mapped_column("bins", bin_dict)

        return bins, status, solver, iv, bin_data, excluded_bin_combos

    def add_mapped_column(self, bin_col, mapping_dict):
        # Define a function to map the old bin number using the dictionary
        def map_bin_number(bin_number):
            for key, value in mapping_dict.items():
                if value[0] <= bin_number <= value[1]:
                    return key
            return None  # If no match is found, return None

        # Apply the mapping function to the specified column to create the new column
        self.input_df["new_bin"] = self.input_df[bin_col].apply(map_bin_number)

    def create_bins(self, df, n_bins=40, min_samples_leaf=1000):
        """
        Create bins using a decision tree regressor.

        Args:
            df (pd.DataFrame): DataFrame containing the score and target variables.
            n_bins (int, optional): Number of bins to create. Default is 40.
            min_samples_leaf (int, optional): Minimum number of samples per leaf. Default is 1000.

        Returns:
            pd.DataFrame: DataFrame with an additional 'bins' column.
        """
        # Set the max_depth, min_samples_leaf, and max_leaf_nodes parameters
        clf = DecisionTreeRegressor(
            max_depth=n_bins, min_samples_leaf=min_samples_leaf, max_leaf_nodes=n_bins
        )

        # Fit the model to create bins
        clf.fit(df[["score"]], df["target"])
        thresholds = clf.tree_.threshold[clf.tree_.threshold > _tree.TREE_UNDEFINED]

        # Add minimum and maximum edges
        bins = sorted([-np.inf] + list(thresholds) + [np.inf])
        df["bins"] = pd.cut(df["score"], bins, labels=False)

        print([round(num, 4) for num in bins])

        # Get input data per bin
        stats_combined = df.groupby("bins").agg({"target": ["sum", "count"]})
        stats_combined = stats_combined.reset_index()
        stats_combined.columns = ["bins", "events", "count"]
        stats_combined["non_events"] = (
            stats_combined["count"] - stats_combined["events"]
        )
        stats_combined["default_rate"] = (
            stats_combined["events"] / stats_combined["count"]
        )

        # Add long-run averages
        annual_averages = df.groupby(["bins", "year"])["score"].mean()
        average_scores_per_bin = annual_averages.groupby("bins").mean().reset_index()
        average_scores_per_bin.columns = ["bins", "lra_dr"]

        stats_combined = pd.merge(
            stats_combined, average_scores_per_bin, on="bins", how="left"
        )

        # Test solution
        self.events = stats_combined["events"]
        self.non_events = stats_combined["non_events"]
        self.longrun_average_dr = stats_combined["lra_dr"]

        return stats_combined, df

    def create_raw_bins(self, df, method="tree", n_bins=40, min_samples_leaf=1000):
        """
        Create bins using the specified method ('tree', 'qcut', 'cut').

        Args:
            df (pd.DataFrame): DataFrame containing the score and target variables.
            method (str, optional): Method to create bins. Options are 'tree', 'qcut', 'cut'. Default is 'tree'.
            n_bins (int, optional): Number of bins to create. Default is 40.
            min_samples_leaf (int, optional): Minimum number of samples per leaf (used for 'tree' method only). Default is 1000.

        Returns:
            pd.DataFrame: DataFrame with an additional 'bins' column.
        """
        self.input_df = df

        if method == "tree":
            df = self.create_bins_with_decision_tree(
                df, n_bins=n_bins, min_samples_leaf=min_samples_leaf
            )
        elif method == "qcut":
            df = self.create_bins_with_qcut(df, n_bins=n_bins)
        elif method == "cut":
            df = self.create_bins_with_cut(df, n_bins=n_bins)
        else:
            raise ValueError("Invalid method. Choose from 'tree', 'qcut', 'cut'.")

        # Get input data per bin
        stats_combined = df.groupby("bins").agg({"target": ["sum", "count"]})
        stats_combined = stats_combined.reset_index()
        stats_combined.columns = ["bins", "events", "count"]
        stats_combined["non_events"] = (
            stats_combined["count"] - stats_combined["events"]
        )
        stats_combined["default_rate"] = (
            stats_combined["events"] / stats_combined["count"]
        )

        # Add long-run averages
        annual_averages = df.groupby(["bins", "year"])["score"].mean()
        average_scores_per_bin = annual_averages.groupby("bins").mean().reset_index()
        average_scores_per_bin.columns = ["bins", "lra_dr"]

        stats_combined = pd.merge(
            stats_combined, average_scores_per_bin, on="bins", how="left"
        )

        # Test solution
        self.events = stats_combined["events"]
        self.non_events = stats_combined["non_events"]
        self.longrun_average_dr = stats_combined["lra_dr"]

        return stats_combined, df

    def create_bins_with_decision_tree(self, df, n_bins=40, min_samples_leaf=1000):
        """
        Create bins using a decision tree regressor.

        Args:
            df (pd.DataFrame): DataFrame containing the score and target variables.
            n_bins (int, optional): Number of bins to create. Default is 40.
            min_samples_leaf (int, optional): Minimum number of samples per leaf. Default is 1000.

        Returns:
            pd.DataFrame: DataFrame with an additional 'bins' column.
        """
        clf = DecisionTreeRegressor(
            max_depth=n_bins, min_samples_leaf=min_samples_leaf, max_leaf_nodes=n_bins
        )
        clf.fit(df[["score"]], df["target"])
        thresholds = clf.tree_.threshold[clf.tree_.threshold > _tree.TREE_UNDEFINED]

        # Add minimum and maximum edges
        bins = sorted([-np.inf] + list(thresholds) + [np.inf])
        df["bins"] = pd.cut(df["score"], bins, labels=False)
        return df

    def create_bins_with_qcut(self, df, n_bins=40):
        """
        Create bins using quantile-based binning (qcut).

        Args:
            df (pd.DataFrame): DataFrame containing the score variable.
            n_bins (int, optional): Number of bins to create. Default is 40.

        Returns:
            pd.DataFrame: DataFrame with an additional 'bins' column.
        """
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(df["score"], quantiles)
        df["bins"] = np.digitize(df["score"], bins, right=False) - 1
        return df

    def create_bins_with_cut(self, df, n_bins=40):
        """
        Create bins using uniform-width binning (cut).

        Args:
            df (pd.DataFrame): DataFrame containing the score variable.
            n_bins (int, optional): Number of bins to create. Default is 40.

        Returns:
            pd.DataFrame: DataFrame with an additional 'bins' column.
        """
        min_score, max_score = df["score"].min(), df["score"].max()
        bins = np.linspace(min_score, max_score, n_bins + 1)
        df["bins"] = np.digitize(df["score"], bins, right=False) - 1
        return df


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Intermediate solutions for debugging and multiple solutions"""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            pass
