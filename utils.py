import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import time
import tracemalloc
from functools import wraps

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class SyntheticDatasetGenerator:
    """
    A class to generate a synthetic dataset and fit a logistic regression model.

    Attributes:
    ----------
    n_samples : int
        Number of samples to be generated.
    seed : int
        Seed for reproducibility.
    df : pd.DataFrame
        DataFrame containing the generated dataset.
    gini : float
        Gini coefficient of the fitted model.
    """

    def __init__(self, n_samples, seed=0):
        """
        Initializes the SyntheticDatasetGenerator with the number of samples and a random seed.

        Parameters:
        ----------
        n_samples : int
            Number of samples to generate.
        seed : int, optional
            Seed for random number generator (default is 0).
        """
        self.n_samples = n_samples
        np.random.seed(seed)
        self.df = None
        self.gini = None

    def generate_dataset(self):
        """
        Generates a synthetic dataset with features (`x1`, `x2`, `x3`, `year`) and target (`target`).

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the generated features and target.
        """
        target = np.random.binomial(1, 0.05, self.n_samples)
        year = np.random.randint(2010, 2023, self.n_samples)

        x1 = np.random.rand(self.n_samples)
        x1_noise = np.random.normal(0, 0.5, self.n_samples)
        x1 = 0.5 * x1 + 0.5 * target + x1_noise

        x2 = np.sin(3 * np.pi * np.random.rand(self.n_samples))
        x2_noise = np.random.normal(0, 0.5, self.n_samples)
        x2 = 0.5 * x2 + 0.5 * target + x2_noise

        # Consolidate all the columns into a DataFrame
        self.df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "year": year,
                "target": target,
            }
        )
        return self.df

    def fit_logistic_regression(self):
        """
        Fits a logistic regression model using features `x1` and `x2` to predict the target.
        Calculates the Gini coefficient based on the model's predictions.

        Returns:
        -------
        pd.DataFrame
            The DataFrame containing the generated features, target, and prediction scores.

        Raises:
        ------
        ValueError
            If the dataset has not been generated before fitting the model.
        """
        if self.df is None:
            raise ValueError(
                "Dataset has not been generated yet. Call 'generate_dataset()' first."
            )

        model = LogisticRegression()
        X = self.df[["x1", "x2"]]
        y = self.df["target"]
        model.fit(X, y)

        # Make predictions (probabilities)
        y_pred_proba = model.predict_proba(X)[:, 1]
        self.df["score"] = y_pred_proba

        # Compute Gini coefficient
        self.gini = 2 * roc_auc_score(y, y_pred_proba) - 1
        return self.df


def memory_time_decorator(func):
    """
    Decorator to measure the memory usage and execution time of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: Wrapped function with memory and time logging.
    """

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


def plot_stats(df):
    fig, ax1 = plt.subplots(figsize=(10,4)) 

    ax1.set_ylabel('count', color='black')
    lns3 = ax1.bar(df['bins'], df['count'], color='navy', alpha=0.5, label='count')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')) # Formatting secondary y-axis

    ax2 = ax1.twinx()
    ax2.set_xlabel('bins')
    ax2.set_ylabel('default_rate and lra_dr', color='black')
    lns1 = ax2.plot(df['bins'], df['default_rate'], color='red', label='default_rate')
    lns2 = ax2.plot(df['bins'], df['lra_dr'], color='gold', label='lra_dr')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # Formatting primary y-axis

    lns = [lns3] + lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.1, 1.0), frameon=False)

    fig.tight_layout()
    plt.show()
