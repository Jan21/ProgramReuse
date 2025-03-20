import json
import argparse
import time
import ast
from tqdm import tqdm

def load_json_data(filepath):
    """Load data from a JSON file"""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} examples")
    return data

def extract_input_arrays(examples):
    """Extract and parse input arrays from the formatted examples"""
    input_arrays = set()
    input_arrays_list = []  # For statistical analysis
    
    print("Extracting input arrays...")
    for example in tqdm(examples, desc="Processing examples"):
        text = example["text"]
        
        # Extract the input array part
        input_start = text.find("input : ") + len("input : ")
        input_end = text.find(" ; operation")
        input_str = text[input_start:input_end]
        
        # Try to parse the array
        try:
            # Convert to tuple to make it hashable for the set
            input_array = tuple(ast.literal_eval(input_str))
            input_arrays.add(input_array)
            input_arrays_list.append(input_array)
        except Exception as e:
            print(f"Error parsing input array: {input_str}")
            print(f"Error details: {e}")
    
    return input_arrays, input_arrays_list

def analyze_arrays(arrays_list):
    """Analyze the arrays to provide statistics"""
    # Count unique arrays
    unique_arrays = set(arrays_list)
    
    # Count by length
    length_counts = {}
    for arr in arrays_list:
        length = len(arr)
        length_counts[length] = length_counts.get(length, 0) + 1
    
    # Calculate statistics
    stats = {
        "total_examples": len(arrays_list),
        "unique_arrays": len(unique_arrays),
        "duplication_rate": 1 - len(unique_arrays) / len(arrays_list) if arrays_list else 0,
        "length_distribution": {k: v / len(arrays_list) * 100 for k, v in length_counts.items()}
    }
    
    return stats

def check_data_leakage(train_filepath, test_filepath):
    """
    Check if there's data leakage between train and test sets
    
    Returns:
        dict: Results of the leakage check and dataset statistics
    """
    start_time = time.time()
    
    # Load datasets
    train_data = load_json_data(train_filepath)
    test_data = load_json_data(test_filepath)
    
    # Extract input arrays
    train_inputs, train_inputs_list = extract_input_arrays(train_data)
    test_inputs, test_inputs_list = extract_input_arrays(test_data)
    
    # Check for overlap
    overlapping_inputs = train_inputs.intersection(test_inputs)
    
    # Calculate statistics
    train_stats = analyze_arrays(train_inputs_list)
    test_stats = analyze_arrays(test_inputs_list)
    
    # Prepare results
    results = {
        "has_leakage": len(overlapping_inputs) > 0,
        "overlapping_arrays": len(overlapping_inputs),
        "train_stats": train_stats,
        "test_stats": test_stats,
        "runtime_seconds": time.time() - start_time
    }
    
    # Print results
    print("\n=== DATA LEAKAGE CHECK RESULTS ===")
    
    if results["has_leakage"]:
        print(f"❌ LEAKAGE DETECTED: {results['overlapping_arrays']} arrays appear in both train and test sets")
        print(f"   Leakage rate: {results['overlapping_arrays'] / len(test_inputs) * 100:.2f}% of test set")
        
        # Show some examples of leakage if there are any
        if overlapping_inputs:
            print("\nExample overlapping arrays:")
            for i, array in enumerate(list(overlapping_inputs)[:5]):
                print(f"  {i+1}. {list(array)}")
            if len(overlapping_inputs) > 5:
                print(f"  ... and {len(overlapping_inputs) - 5} more")
    else:
        print("✅ NO LEAKAGE DETECTED: Train and test sets contain completely different arrays")
    
    # Print statistics
    print("\n=== DATASET STATISTICS ===")
    print(f"Training set: {train_stats['total_examples']:,} examples with {train_stats['unique_arrays']:,} unique arrays")
    print(f"Testing set:  {test_stats['total_examples']:,} examples with {test_stats['unique_arrays']:,} unique arrays")
    
    print("\nArray length distribution in training set:")
    for length, percentage in sorted(train_stats["length_distribution"].items()):
        print(f"  Length {length}: {percentage:.2f}%")
    
    print("\nExecution time: {:.2f} seconds".format(results["runtime_seconds"]))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for data leakage between train and test sets")
    parser.add_argument("train_file", help="Path to the training data JSON file")
    parser.add_argument("test_file", help="Path to the test data JSON file")
    args = parser.parse_args()
    
    check_data_leakage(args.train_file, args.test_file)