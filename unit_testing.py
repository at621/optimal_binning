import unittest
import optimal_binning as ob
import pandas as pd
from ortools.sat.python import cp_model


# Unit tests
class TestBinStatisticsOptimalBinning(unittest.TestCase):
    def setUp(self):
        self.events = pd.Series([10, 20, 30, 40, 50])
        self.non_events = pd.Series([90, 80, 70, 60, 50])
        self.binning = ob.BinStatisticsOptimalBinning(
            self.events, self.non_events, 2, 4
        )

    def test_calculate_iv(self):
        iv = self.binning.calculate_iv(30, 70)
        self.assertIsInstance(iv, float)

    def test_test_proportions(self):
        z_score = self.binning.test_proportions(10, 90, 20, 80)
        self.assertIsInstance(z_score, float)

    def test_individual_bin_stats(self):
        stats = self.binning.individual_bin_stats()
        self.assertIsInstance(stats, dict)
        self.assertTrue(len(stats) > 0)

    def test_bin_combo_stats(self):
        stats_b = self.binning.individual_bin_stats()
        combo_stats = self.binning.bin_combo_stats(stats_b)
        self.assertIsInstance(combo_stats, dict)

    def test_create_variables(self):
        self.binning.create_variables()
        self.assertTrue(len(self.binning.x) > 0)

    def test_set_constraints(self):
        bin_data = self.binning.individual_bin_stats()
        excluded_bin_combos = self.binning.bin_combo_stats(bin_data)
        self.binning.create_variables()
        self.binning.set_constraints(bin_data, excluded_bin_combos)
        # Test passes if no exceptions are raised

    def test_set_objective(self):
        bin_data = self.binning.individual_bin_stats()
        self.binning.create_variables()
        self.binning.set_objective({key: val[0] for key, val in bin_data.items()})
        # Test passes if no exceptions are raised

    def test_solve(self):
        bins, status, solver, iv, bin_data, excluded_bin_combos = self.binning.solve()
        self.assertIsInstance(bins, list)
        self.assertTrue(status in [cp_model.OPTIMAL, cp_model.FEASIBLE])


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
