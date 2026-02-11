import os
import pickle
import hashlib
from typing import Optional, Tuple
from results import SearchOutput

CACHE_DIR = "ground_truth_cache"

def _get_cache_filename(dataset_path: str, query_path: str, num_neighbors: int, range_radius: float, search_for_range: bool) -> str:

    abs_dataset = os.path.abspath(dataset_path)
    abs_query = os.path.abspath(query_path)
    
    try:
        ds_stat = os.stat(abs_dataset)
        qs_stat = os.stat(abs_query)
        file_info = f"{ds_stat.st_size}_{ds_stat.st_mtime}_{qs_stat.st_size}_{qs_stat.st_mtime}"
    except OSError:
        file_info = "unknown"

    config_str = f"{abs_dataset}_{abs_query}_{num_neighbors}_{range_radius}_{search_for_range}_{file_info}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    dataset_name = os.path.basename(dataset_path)
    query_name = os.path.basename(query_path)
    
    filename = f"gt_{dataset_name}_{query_name}_N{num_neighbors}_R{range_radius}_{config_hash}.pkl"
    return os.path.join(os.path.dirname(__file__), CACHE_DIR, filename)

def load_ground_truth(dataset_path: str, query_path: str, num_neighbors: int, range_radius: float, search_for_range: bool) -> Optional[Tuple[SearchOutput, float]]:
    """
    Loads the ground truth from cache if it exists.
    Returns a tuple (SearchOutput, brute_force_time) or None.
    """
    filename = _get_cache_filename(dataset_path, query_path, num_neighbors, range_radius, search_for_range)
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                # Expecting a dict {'output': SearchOutput, 'time': float}
                return data['output'], data['time']
        except Exception as e:
            print(f"Warning: Failed to load ground truth cache: {e}")
            return None
    return None

def save_ground_truth(output: SearchOutput, brute_time: float, dataset_path: str, query_path: str, num_neighbors: int, range_radius: float, search_for_range: bool):
    """
    Saves the ground truth to cache.
    """
    filename = _get_cache_filename(dataset_path, query_path, num_neighbors, range_radius, search_for_range)
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    data = {
        'output': output,
        'time': brute_time
    }
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Ground truth cached to {filename}")
    except Exception as e:
        print(f"Warning: Failed to save ground truth cache: {e}")
