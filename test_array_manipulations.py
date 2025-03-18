import unittest
from array_manipulations import create_array_manipulations

class TestArrayManipulations(unittest.TestCase):
    def setUp(self):
        self.manipulations = create_array_manipulations()
        # Test arrays with different characteristics
        self.test_arrays = {
            'normal': [1, 2, 3, 4, 5],
            'negative': [-1, -2, -3, -4, -5],
            'mixed': [-2, 3, -4, 5, -6],
            'zeros': [0, 0, 0, 0, 0],
            'max_range': [-9, 9, -9, 9, -9],
            'repeated': [1, 1, 2, 2, 3],
            'even_odd': [2, 1, 4, 3, 6],
            'single': [5]
        }

    def test_permutations(self):
        # Test swap_first_last
        self.assertEqual(
            self.manipulations['permutations']['swap_first_last']([1, 2, 3, 4, 5]),
            [5, 2, 3, 4, 1]
        )
        
        # Test swap_second_third
        self.assertEqual(
            self.manipulations['permutations']['swap_second_third']([1, 2, 3, 4, 5]),
            [1, 3, 2, 4, 5]
        )
        
        # Test reverse_first_three
        self.assertEqual(
            self.manipulations['permutations']['reverse_first_three']([1, 2, 3, 4, 5]),
            [3, 2, 1, 4, 5]
        )

    def test_relational(self):
        # Test sort_ascending
        self.assertEqual(
            self.manipulations['relational']['sort_ascending']([3, 1, 4, 1, 5]),
            [1, 1, 3, 4, 5]
        )
        
        # Test sort_descending
        self.assertEqual(
            self.manipulations['relational']['sort_descending']([3, 1, 4, 1, 5]),
            [5, 4, 3, 1, 1]
        )
        
        # Test even_odd_separate
        self.assertEqual(
            self.manipulations['relational']['even_odd_separate']([1, 2, 3, 4, 5]),
            [2, 4, 1, 3, 5]
        )

    def test_mappings(self):
        # Test double_each with wrapping
        self.assertEqual(
            self.manipulations['mappings']['double_each']([5, 6, 7, 8, 9]),
            [-9, -7, -5, -3, -1]
        )
        
        # Test add_one with wrapping
        self.assertEqual(
            self.manipulations['mappings']['add_one']([9, 8, 7, 6, 5]),
            [-9, 9, 8, 7, 6]
        )
        
        # Test square_each with wrapping
        self.assertEqual(
            self.manipulations['mappings']['square_each']([3, 4, 5]),
            [9, -3, 6]
        )

    def test_removal(self):
        # Test remove_zeros
        self.assertEqual(
            self.manipulations['removal']['remove_zeros']([0, 1, 0, 2, 0]),
            [1, 2]
        )
        
        # Test remove_evens
        self.assertEqual(
            self.manipulations['removal']['remove_evens']([1, 2, 3, 4, 5]),
            [1, 3, 5]
        )
        
        # Test remove_duplicates
        self.assertEqual(
            self.manipulations['removal']['remove_duplicates']([1, 1, 2, 2, 3]),
            [1, 2, 3]
        )

    def test_addition(self):
        # Test append_zero
        self.assertEqual(
            self.manipulations['addition']['append_zero']([1, 2, 3]),
            [1, 2, 3, 0]
        )
        
        # Test prepend_zero
        self.assertEqual(
            self.manipulations['addition']['prepend_zero']([1, 2, 3]),
            [0, 1, 2, 3]
        )
        
        # Test interleave_zeros
        self.assertEqual(
            self.manipulations['addition']['interleave_zeros']([1, 2, 3]),
            [1, 0, 2, 0, 3, 0]
        )

    def test_conditional(self):
        # Test zero_evens
        self.assertEqual(
            self.manipulations['conditional']['zero_evens']([1, 2, 3, 4, 5]),
            [1, 0, 3, 0, 5]
        )
        
        # Test negate_odds with wrapping
        self.assertEqual(
            self.manipulations['conditional']['negate_odds']([1, 2, 3, 4, 5]),
            [-1, 2, -3, 4, -5]
        )
        
        # Test double_evens with wrapping
        self.assertEqual(
            self.manipulations['conditional']['double_evens']([2, 3, 4, 5, 6]),
            [4, 3, 8, 5, -7]
        )

    def test_sliding(self):
        # Test shift_left
        self.assertEqual(
            self.manipulations['sliding']['shift_left']([1, 2, 3, 4, 5]),
            [2, 3, 4, 5, 1]
        )
        
        # Test shift_right
        self.assertEqual(
            self.manipulations['sliding']['shift_right']([1, 2, 3, 4, 5]),
            [5, 1, 2, 3, 4]
        )
        
        # Test reverse_each_pair
        self.assertEqual(
            self.manipulations['sliding']['reverse_each_pair']([1, 2, 3, 4, 5]),
            [2, 1, 4, 3, 5]
        )

    def test_mathematical(self):
        # Test cumulative_sum with wrapping
        self.assertEqual(
            self.manipulations['mathematical']['cumulative_sum']([1, 2, 3, 4, 5]),
            [1, 3, 6, -9, -4]
        )
        
        # Test running_max
        self.assertEqual(
            self.manipulations['mathematical']['running_max']([3, 1, 4, 1, 5]),
            [3, 3, 4, 4, 5]
        )
        
        # Test difference_array with wrapping
        self.assertEqual(
            self.manipulations['mathematical']['difference_array']([1, 2, 3, 4, 5]),
            [1, 1, 1, 1, 1]
        )

    def test_pattern(self):
        # Test alternate_signs with wrapping
        self.assertEqual(
            self.manipulations['pattern']['alternate_signs']([1, 2, 3, 4, 5]),
            [1, -2, 3, -4, 5]
        )
        
        # Test repeat_pattern
        self.assertEqual(
            self.manipulations['pattern']['repeat_pattern']([1, 2, 3]),
            [1, 2, 3, 1, 2, 3]
        )
        
        # Test staircase with wrapping
        self.assertEqual(
            self.manipulations['pattern']['staircase']([1, 2, 3]),
            [1, 4, 9]
        )

    def test_special(self):
        # Test unique_count with wrapping
        self.assertEqual(
            self.manipulations['special']['unique_count']([1, 1, 2, 2, 3]),
            [2, 2, 2, 2, 1]
        )
        
        # Test position_value with wrapping
        self.assertEqual(
            self.manipulations['special']['position_value']([1, 2, 3]),
            [1, 4, 9]
        )
        
        # Test relative_position
        self.assertEqual(
            self.manipulations['special']['relative_position']([3, 1, 4, 1, 5]),
            [2, 0, 3, 0, 4]
        )

    def test_edge_cases(self):
        
        # Test single element
        single_array = [5]
        for group in self.manipulations.values():
            for func in group.values():
                result = func(single_array)
                self.assertIsInstance(result, list)
        
        # Test with maximum range values
        max_array = [-9, 9, -9, 9, -9]
        for group in self.manipulations.values():
            for func in group.values():
                result = func(max_array)
                self.assertTrue(all(-9 <= x <= 9 for x in result))

if __name__ == '__main__':
    unittest.main() 