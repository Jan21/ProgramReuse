import random
import json
import time
import os
from tqdm import tqdm
import threading
import queue
import multiprocessing as mp
from array_manipulations import create_array_manipulations

def generate_random_arrays(count=100000, min_length=5, max_length=9, min_val=1, max_val=9, seed=None):
    """Generate arrays with varying lengths and unique elements"""
    if seed is not None:
        random.seed(seed)
    
    arrays = []
    possible_values = list(range(min_val, max_val + 1))
    
    for _ in tqdm(range(count), desc="Generating arrays", unit="array"):
        length = random.randint(min_length, max_length)
        actual_length = min(length, len(possible_values))
        new_array = random.sample(possible_values, actual_length)
        arrays.append(new_array)
    
    # Print some statistics
    lengths = [len(arr) for arr in arrays]
    length_counts = {length: lengths.count(length) for length in range(min_length, max_length + 1)}
    
    print("\nLength distribution:")
    for length, count in sorted(length_counts.items()):
        print(f"  Length {length}: {count:,} arrays ({count/len(arrays)*100:.2f}%)")
    
    return arrays

def split_arrays(arrays, test_size=0.01, seed=None):
    """Split arrays into training and test sets to prevent data leakage"""
    if seed is not None:
        random.seed(seed)
    
    # Make a copy and shuffle
    shuffled_arrays = arrays.copy()
    random.shuffle(shuffled_arrays)
    
    # Deduplicate using tuples (tuples are hashable for set)
    unique_arrays_tuples = list(set(tuple(arr) for arr in shuffled_arrays))
    print(f"Generated {len(arrays)} arrays, {len(unique_arrays_tuples)} are unique")
    
    # Convert tuples back to lists before returning
    unique_arrays = [list(arr_tuple) for arr_tuple in unique_arrays_tuples]
    
    # Calculate split point
    split_idx = int(len(unique_arrays) * (1 - test_size))
    
    # Split the arrays (now as lists, not tuples)
    train_arrays = unique_arrays[:split_idx]
    test_arrays = unique_arrays[split_idx:]
    
    print(f"Split {len(arrays)} arrays into {len(train_arrays)} training arrays and {len(test_arrays)} test arrays")
    
    return train_arrays, test_arrays

def process_arrays_threaded(arrays, batch_size=1000, num_threads=None):
    """Process arrays in batches using threading"""
    # Get all manipulation functions
    manipulations = create_array_manipulations()
    
    # Determine number of threads
    if num_threads is None:
        num_threads = min(32, mp.cpu_count() * 2)
    
    # Split arrays into batches
    batches = [arrays[i:i+batch_size] for i in range(0, len(arrays), batch_size)]
    
    # Create result queue and result dictionary
    result_queue = queue.Queue()
    all_results = {}
    lock = threading.Lock()
    
    # Track completed batches for progress bar
    completed_batches = 0
    
    # Define worker function for threading
    def process_batch(batch_idx, batch_arrays):
        batch_results = {}
        for arr in batch_arrays:
            arr_tuple = tuple(arr)
            batch_results[arr_tuple] = {}
            
            for category_name, functions in manipulations.items():
                batch_results[arr_tuple][category_name] = {}
                
                for func_name, func in functions.items():
                    try:
                        batch_results[arr_tuple][category_name][func_name] = func(arr.copy())
                    except Exception as e:
                        batch_results[arr_tuple][category_name][func_name] = f"Error: {str(e)}"
        
        result_queue.put((batch_idx, batch_results))
    
    # Create thread pool with limited active threads
    thread_semaphore = threading.Semaphore(num_threads)
    
    def thread_worker(batch_idx, batch):
        with thread_semaphore:
            process_batch(batch_idx, batch)
    
    # Create and start threads for all batches
    threads = []
    for i, batch in enumerate(batches):
        thread = threading.Thread(target=thread_worker, args=(i, batch))
        threads.append(thread)
        thread.start()
    
    # Monitor progress with tqdm
    with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
        while completed_batches < len(batches):
            try:
                batch_idx, batch_results = result_queue.get(timeout=0.1)
                with lock:
                    all_results.update(batch_results)
                completed_batches += 1
                pbar.update(1)
            except queue.Empty:
                # No results yet, continue waiting
                pass
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return all_results

def process_arrays_sequential(arrays, manipulations=None):
    """Process arrays sequentially for reliability"""
    if manipulations is None:
        manipulations = create_array_manipulations()
    
    all_results = {}
    
    with tqdm(total=len(arrays), desc="Processing arrays", unit="array") as pbar:
        for arr in arrays:
            arr_tuple = tuple(arr)
            all_results[arr_tuple] = {}
            
            for category_name, functions in manipulations.items():
                all_results[arr_tuple][category_name] = {}
                
                for func_name, func in functions.items():
                    try:
                        all_results[arr_tuple][category_name][func_name] = func(arr.copy())
                    except Exception as e:
                        all_results[arr_tuple][category_name][func_name] = f"Error: {str(e)}"
            
            pbar.update(1)
    
    return all_results

def apply_all_manipulations(arrays, method='threaded', batch_size=1000, num_threads=None):
    """Apply all manipulations to arrays"""
    manipulations = create_array_manipulations()
    total_categories = len(manipulations)
    total_functions = sum(len(functions) for functions in manipulations.values())
    total_operations = len(arrays) * total_functions
    
    print(f"Processing {len(arrays):,} arrays with {total_functions} functions across {total_categories} categories")
    print(f"Total operations to perform: {total_operations:,}")
    
    start_time = time.time()
    
    if method == 'sequential':
        results = process_arrays_sequential(arrays, manipulations)
    elif method == 'threaded':
        results = process_arrays_threaded(arrays, batch_size, num_threads)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sequential' or 'threaded'.")
    
    total_time = time.time() - start_time
    operations_per_second = total_operations / total_time if total_time > 0 else 0
    
    print(f"\n✅ Processing completed in {total_time:.2f} seconds")
    print(f"🚀 Speed: {operations_per_second:.2f} operations/second")
    
    return results

def format_to_text(results_dict):
    """
    Format results to text format for neural network training
    with arrays formatted as [ 1 2 3 4 ] instead of [1, 2, 3, 4]
    and group separated from operation
    """
    formatted_examples = []
    
    total_entries = sum(len(categories) * len(funcs) 
                       for arr, categories in results_dict.items() 
                       for category, funcs in categories.items())
    
    with tqdm(total=total_entries, desc="Formatting entries", unit="entry") as pbar:
        for arr_tuple, categories in results_dict.items():
            # Format input array with spaces instead of commas
            arr_list = list(arr_tuple)
            formatted_input = "[ " + " ".join(str(x) for x in arr_list) + " ]"
            
            for category, functions in categories.items():
                for function, result in functions.items():
                    # Format result array with spaces if it's a list
                    if isinstance(result, list):
                        formatted_result = "[ " + " ".join(str(x) for x in result) + " ]"
                    else:
                        # For error messages or other non-list results
                        formatted_result = str(result)
                    
                    # Separate group and operation as requested
                    group = f"{category}"
                    operation = f"{function}"
                    
                    # Create the formatted text entry
                    text_entry = f"input ; {formatted_input} ; operation ; {group}_{operation} ; result : {formatted_result}"
                    formatted_examples.append({"text": text_entry})
                    pbar.update(1)
    
    return formatted_examples

def balance_operations(formatted_examples, seed=None):
    """Balance examples to have equal representation of each operation"""
    if seed is not None:
        random.seed(seed)
    
    # Group examples by operation
    operation_groups = {}
    
    for example in formatted_examples:
        text = example["text"]
        operation_start = text.find("operation : ") + len("operation : ")
        operation_end = text.find(" ,")
        operation = text[operation_start:operation_end]
        
        if operation not in operation_groups:
            operation_groups[operation] = []
        
        operation_groups[operation].append(example)
    
    print(f"Found {len(operation_groups)} distinct operations")
    
    # Find the minimum count across all operations
    min_count = min(len(examples) for examples in operation_groups.values())
    print(f"Each operation has at least {min_count} examples")
    
    # Sample examples from each operation group
    balanced_examples = []
    
    for operation, examples in operation_groups.items():
        # Shuffle the examples for this operation
        random.shuffle(examples)
        
        # Select min_count examples (or all if fewer)
        selected = examples[:min_count]
        balanced_examples.extend(selected)
    
    print(f"Balanced dataset has {len(balanced_examples):,} examples")
    return balanced_examples

def shuffle_thoroughly(examples, passes=5, seed=None):
    """Perform multiple shuffle passes for thorough randomization"""
    if seed is not None:
        random.seed(seed)
    
    shuffled = examples.copy()
    for _ in range(passes):
        random.shuffle(shuffled)
    
    return shuffled

def prepare_dataset(results_dict, balance=True, shuffle=True, seed=None):
    """Format, balance, and shuffle a dataset"""
    # Format to text
    formatted = format_to_text(results_dict)
    print(f"Formatted {len(formatted):,} examples")
    
    # Balance operations if requested
    if balance:
        formatted = balance_operations(formatted, seed=seed)
    
    # Shuffle thoroughly if requested
    if shuffle:
        formatted = shuffle_thoroughly(formatted, passes=5, seed=seed)
        print(f"Shuffled {len(formatted):,} examples")
    
    return formatted

def save_datasets(train_data, test_data, output_dir="./data", prefix=""):
    """Save datasets to JSON files"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create filenames
    train_file = os.path.join(output_dir, f"{prefix}train.json")
    test_file = os.path.join(output_dir, f"{prefix}test.json")
    
    # Save training data
    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    
    # Save test data
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    print(f"✅ Saved {len(train_data):,} training examples to {train_file}")
    print(f"✅ Saved {len(test_data):,} testing examples to {test_file}")
    
    return {
        "train_file": train_file,
        "test_file": test_file
    }

def run(count=100000, min_length=5, max_length=9, min_val=-9, max_val=9,
                         test_size=0.2, method='threaded', batch_size=1000, num_threads=None,
                         balance=True, shuffle=True, output_dir="data", prefix="", seed=42):
    """
    Run the improved pipeline with proper train/test splitting:
    1. Generate arrays
    2. Split arrays into train/test sets (prevents leakage)
    3. Process each set separately
    4. Format, balance, and shuffle each set
    5. Save to files
    """
    start_time = time.time()
    
    # Step 1: Generate arrays
    print(f"Step 1: Generating {count:,} random arrays...")
    all_arrays = generate_random_arrays(
        count=count,
        min_length=min_length,
        max_length=max_length,
        min_val=min_val,
        max_val=max_val,
        seed=seed
    )
    
    # Step 2: Split arrays into train/test sets
    print(f"\nStep 2: Splitting arrays into train/test sets...")
    train_arrays, test_arrays = split_arrays(all_arrays, test_size=test_size, seed=seed)
    
    # Step 3: Process training arrays
    print(f"\nStep 3a: Processing {len(train_arrays):,} training arrays...")
    train_results = apply_all_manipulations(
        train_arrays,
        method=method,
        batch_size=batch_size,
        num_threads=num_threads
    )
    
    # Step 3b: Process test arrays
    print(f"\nStep 3b: Processing {len(test_arrays):,} test arrays...")
    test_results = apply_all_manipulations(
        test_arrays,
        method=method,
        batch_size=batch_size,
        num_threads=num_threads
    )
    
    # Step 4: Prepare training dataset
    print(f"\nStep 4a: Preparing training dataset...")
    train_data = prepare_dataset(train_results, balance=balance, shuffle=shuffle, seed=seed)
    
    # Step 4b: Prepare test dataset
    print(f"\nStep 4b: Preparing test dataset...")
    test_data = prepare_dataset(test_results, balance=balance, shuffle=shuffle, seed=seed+1)
    
    # Step 5: Save datasets
    print(f"\nStep 5: Saving datasets...")
    file_paths = save_datasets(train_data, test_data, output_dir=output_dir, prefix=prefix)
    
    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n🎉 Pipeline completed in {hours}h {minutes}m {seconds}s!")
    print(f"Generated {len(all_arrays):,} arrays")
    print(f"Training data: {len(train_data):,} examples from {len(train_arrays):,} arrays")
    print(f"Test data: {len(test_data):,} examples from {len(test_arrays):,} arrays")
    
    return {
        "arrays_count": len(all_arrays),
        "train_arrays": len(train_arrays),
        "test_arrays": len(test_arrays),
        "train_examples": len(train_data),
        "test_examples": len(test_data),
        "train_file": file_paths["train_file"],
        "test_file": file_paths["test_file"],
        "runtime_seconds": total_time
    }

# If script is run directly
if __name__ == "__main__":
    # Run with desired parameters
    run(
        count=80_000,            # Number of arrays to generate
        min_length=5,             # Minimum array length
        max_length=9,             # Maximum array length
        min_val=-9,                # Minimum value in arrays
        max_val=9,                # Maximum value in arrays
        test_size=0.004,            # Fraction for test set
        method='threaded',        # Use threading for better performance
        batch_size=1000,          # Arrays per batch for threaded processing
        num_threads=None,         # Number of threads (None = auto)
        balance=True,             # Balance operation counts
        shuffle=True,             # Thoroughly shuffle examples
        output_dir="data",        # Output directory
        prefix="",                # Filename prefix
        seed=42                   # Random seed for reproducibility
    )