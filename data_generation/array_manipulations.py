def create_array_manipulations():
    def wrap_to_range(x):
        # Wraps a number to stay within -9 to 9
        if x > 9:
            return ((x - 9) % 19) - 10
        elif x < -9:
            return ((x + 9) % 19) - 9
        return x

    def swap_positions(arr, i, j):
        arr = arr.copy()
        i = min(len(arr)-1, i)
        j = min(len(arr)-1, j)
        arr[i], arr[j] = arr[j], arr[i]
        return arr

    def reverse_range(arr, start, end):
        return arr[:start] + arr[start:end+1][::-1] + arr[end+1:]

    def rotate_left(arr, k):
        return arr[k:] + arr[:k]

    def rotate_right(arr, k):
        return arr[-k:] + arr[:-k]

    def swap_adjacent(arr):
        return [arr[i+1] if i % 2 == 0 and i < len(arr)-1 else arr[i-1] if i % 2 == 1 else arr[i] for i in range(len(arr))]

    # Group 1: Permutations
    permutations = {
        'swap_first_last': lambda arr: swap_positions(arr, 0, -1),
        'swap_second_third': lambda arr: swap_positions(arr, 1, 2),
        'reverse_first_three': lambda arr: reverse_range(arr, 0, 2),
        'rotate_left_one': lambda arr: rotate_left(arr, 1),
        'rotate_right_one': lambda arr: rotate_right(arr, 1)
    }

    # Group 2: Relational Rearrangements
    relational = {
        'sort_ascending': lambda arr: sorted(arr),
        'sort_descending': lambda arr: sorted(arr, reverse=True),
        'even_odd_separate': lambda arr: [x for x in arr if x % 2 == 0] + [x for x in arr if x % 2 == 1],
        'greater_than_5_first': lambda arr: [x for x in arr if x > 5] + [x for x in arr if x <= 5],
        'smallest_first': lambda arr: [min(arr)] + [x for x in arr if x != min(arr)]
    }

    # Group 3: Dictionary Mappings
    mappings = {
        'double_each': lambda arr: [wrap_to_range(x * 2) for x in arr],
        'add_one': lambda arr: [wrap_to_range(x + 1) for x in arr],
        'square_each': lambda arr: [wrap_to_range(x * x) for x in arr],
        'negate_each': lambda arr: [wrap_to_range(-x) for x in arr],
        'mod_three': lambda arr: [wrap_to_range(x % 3) for x in arr]
    }

    # Group 4: Removal Operations
    removal = {
        'remove_zeros': lambda arr: [x for x in arr if x != 0],
        'remove_evens': lambda arr: [x for x in arr if x % 2 == 1],
        'remove_odds': lambda arr: [x for x in arr if x % 2 == 0],
        'remove_duplicates': lambda arr: list(dict.fromkeys(arr)),
        'remove_greater_than_5': lambda arr: [x for x in arr if x <= 5]
    }

    # Group 5: Addition Operations
    addition = {
        'append_zero': lambda arr: arr + [0],
        'prepend_zero': lambda arr: [0] + arr,
        'interleave_zeros': lambda arr: [x for pair in zip(arr, [0] * len(arr)) for x in pair],
        'repeat_each': lambda arr: [x for x in arr for _ in range(2)],
        'append_reverse': lambda arr: arr + arr[::-1]
    }

    # Group 6: Conditional Transformations
    conditional = {
        'zero_evens': lambda arr: [0 if x % 2 == 0 else x for x in arr],
        'negate_odds': lambda arr: [wrap_to_range(-x) if x % 2 == 1 else x for x in arr],
        'double_evens': lambda arr: [wrap_to_range(x * 2) if x % 2 == 0 else x for x in arr],
        'increment_odds': lambda arr: [wrap_to_range(x + 1) if x % 2 == 1 else x for x in arr],
        'zero_greater_than_5': lambda arr: [0 if x > 5 else x for x in arr]
    }

    # Group 7: Sliding Operations
    sliding = {
        'shift_left': lambda arr: arr[1:] + [arr[0]],
        'shift_right': lambda arr: [arr[-1]] + arr[:-1],
        'reverse_each_pair': lambda arr: [y for i in range(0, len(arr), 2) for y in reversed(arr[i:i+2])],
        'swap_adjacent_pairs': lambda arr: [arr[i+1] if i % 2 == 0 and i < len(arr)-1 else arr[i-1] if i % 2 == 1 else arr[i] for i in range(len(arr))],
        'rotate_by_two': lambda arr: arr[2:] + arr[:2]
    }

    # Group 8: Mathematical Operations
    mathematical = {
        'cumulative_sum': lambda arr: [wrap_to_range(sum(arr[:i+1])) for i in range(len(arr))],
        'running_max': lambda arr: [max(arr[:i+1]) for i in range(len(arr))],
        'running_min': lambda arr: [min(arr[:i+1]) for i in range(len(arr))],
        'difference_array': lambda arr: [wrap_to_range(arr[i] - arr[i-1]) if i > 0 else arr[i] for i in range(len(arr))],
        'prefix_sum': lambda arr: [wrap_to_range(sum(arr[:i+1])) for i in range(len(arr))]
    }

    # Group 9: Pattern Operations
    pattern = {
        'alternate_signs': lambda arr: [wrap_to_range(x * (-1)**i) for i, x in enumerate(arr)],
        'repeat_pattern': lambda arr: arr * 2,
        'mirror_array': lambda arr: arr + arr[::-1],
        'staircase': lambda arr: [wrap_to_range(arr[i] * (i + 1)) for i in range(len(arr))],
        'wave_pattern': lambda arr: [wrap_to_range(x * (1 + (i % 2))) for i, x in enumerate(arr)]
    }

    # Group 10: Special Operations
    special = {
        'unique_count': lambda arr: [wrap_to_range(arr.count(x)) for x in arr],
        'position_value': lambda arr: [wrap_to_range(x * (i + 1)) for i, x in enumerate(arr)],
        'relative_position': lambda arr: [wrap_to_range(sum(1 for y in arr if y < x)) for x in arr],
        'frequency_map': lambda arr: [wrap_to_range(arr.count(x)) for x in arr]
    }

    return {
        'permutations': permutations,
        'relational': relational,
        'mappings': mappings,
        'removal': removal,
        'addition': addition,
        'conditional': conditional,
        'sliding': sliding,
        'mathematical': mathematical,
        'pattern': pattern,
        'special': special
    }

# Example usage:
if __name__ == "__main__":
    manipulations = create_array_manipulations()
    test_array = [1, 2, 3, 4, 5]
    
    # Example of using functions from different groups
    print("Original array:", test_array)
    print("Swap first and last:", manipulations['permutations']['swap_first_last'](test_array))
    print("Sort ascending:", manipulations['relational']['sort_ascending'](test_array))
    print("Double each:", manipulations['mappings']['double_each'](test_array))
