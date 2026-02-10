"""Disk cache for brute-force ground-truth results.

Brute-force search over SIFT-1M or even MNIST-60K is expensive, on the order of
minutes.  Since the ground truth depends only on the dataset, the queries, and the
search parameters (N, R, range flag), we cache the SearchOutput to a pickle file
and skip recomputation on subsequent runs.

Cache Invalidation Strategy:
The cache key incorporates the absolute paths of the dataset and query files, the
num_neighbors, range_radius, and search_for_range search parameters, and the file
sizes and modification times from os.stat of both input files.  All of these fields
are concatenated into a single string and hashed with MD5 to produce a 32-character
hex digest.  If either input file is modified (size or mtime changes), a new cache
filename is generated automatically, causing a clean recomputation.

Storage:
Cache files live in a ground_truth_cache directory next to this module.  Each file
is a pickle containing a dictionary with an output key holding the SearchOutput and
a time key holding the original brute-force wall-clock duration.  The time field is
useful for CSV speedup calculations even when the cache is hit.

Security Note:
pickle.load is used only on locally-generated files.  Do not point the cache
directory at an untrusted location.
"""

import os
import pickle
import hashlib
import logging
from typing import Optional, Tuple
from results import SearchOutput

logger = logging.getLogger(__name__)

CACHE_DIR = "ground_truth_cache"


def _get_cache_filename(
    dataset_path: str,
    query_path: str,
    num_neighbors: int,
    range_radius: float,
    search_for_range: bool,
) -> str:
    """Derive a deterministic, content-aware cache filename.

    The filename encodes the basename of both input files for human readability,
    plus an MD5 hash of all parameters and file metadata for uniqueness.  The
    dataset_path and query_path point to the input binary files, num_neighbors
    is the top-N parameter, range_radius is the Euclidean radius R, and
    search_for_range indicates whether range search was enabled.  Returns the
    absolute path to the cache pickle file.
    """

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
    """Load cached ground truth if a matching cache file exists.

    Returns a tuple of (SearchOutput, brute_force_time) on cache hit, or None on
    miss.  Silently returns None if the pickle is corrupted, for example after a
    code change that altered the SearchOutput schema.
    """
    filename = _get_cache_filename(dataset_path, query_path, num_neighbors, range_radius, search_for_range)
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                # Expecting a dict {'output': SearchOutput, 'time': float}
                return data['output'], data['time']
        except Exception as e:
            logger.warning("Failed to load ground truth cache: %s", e)
            return None
    return None

def save_ground_truth(output: SearchOutput, brute_time: float, dataset_path: str, query_path: str, num_neighbors: int, range_radius: float, search_for_range: bool) -> None:
    """Persist ground-truth results so future runs can skip brute force.

    Creates the ground_truth_cache directory if it does not exist.  A warning is
    printed but no exception raised if the write fails.
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
        logger.info("Ground truth cached to %s", filename)
    except Exception as e:
        logger.warning("Failed to save ground truth cache: %s", e)
