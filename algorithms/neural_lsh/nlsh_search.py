import torch
import time
import numpy as np

import parse_sift
import parse_mnist
import nlsh_build_and_search
import results
import nlsh_search_args
import brute_force
import metrics
import output_writer
import ground_truth_cache


# To help with de-allocating memory, potentially
def numpy_to_tensor(n):
    a = torch.from_numpy(n)
    return torch.utils.data.TensorDataset(a)


def main():
    arguments = nlsh_search_args.parse_args()

    cpu_or_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'  # used twice
    device = torch.device(cpu_or_cuda)

    parse = parse_sift.parse if arguments.is_sift else parse_mnist.parse

    # Load data as numpy for brute force and tensor for neural search
    dataset_as_numpy = parse(arguments.input_dataset_filename).astype(np.float32)
    dataset_as_tensor = torch.from_numpy(dataset_as_numpy).to(device)

    queries_as_numpy = parse(arguments.queries_filename).astype(np.float32)
    queries_as_tensor = numpy_to_tensor(queries_as_numpy)
    queries_loader = torch.utils.data.DataLoader(
        queries_as_tensor,
        batch_size=arguments.batch_size,
        shuffle=False
    )

    # Load checkpoint first to get metadata
    state = torch.load(arguments.input_index_filename, map_location=cpu_or_cuda)
    meta = state['metadata']['mlp']

    # Reconstruct model with correct dimensions from checkpoint
    # Fixed: Use dimensions from metadata instead of undefined variables
    model = nlsh_build_and_search.MLPClassifier(
        input_dim=state['metadata']['feature_dim'],
        output_dim=len(state['inverted_index']),
        hidden_dim=meta['hidden_dim'],
        hidden_layers=meta['layers'],
    ).to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()  # switch to evaluation mode (because a model can also be in training mode)


    # Neural LSH

    starting_time = time.time()

    batches_of_predicted = []
    with torch.no_grad():  # disable gradient computation, for performance
        for batch in queries_loader:
            assert len(batch) == 1
            queries = batch[0].to(device)  # Move batch to device
            possible_labels_per_query = model(queries)
            # max score labels for each image of the batch
            _, predicted = torch.topk(
                possible_labels_per_query.data,
                arguments.multi_probe_bins,
                dim=1
            )
            batches_of_predicted.append(predicted)

    # Initialize search output
    output = results.SearchOutput()
    output.algorithm = "neural LSH"

    ids_per_label = state['inverted_index']
    predicted_labels_for_all_queries = torch.cat(batches_of_predicted)

    # Iterate over queries
    for query_image_id, predicted_labels in enumerate(predicted_labels_for_all_queries):
        
        # Fixed: Collect all candidate IDs from all probed buckets first
        # This ensures we find the true nearest neighbors across all probes
        candidate_indices = []
        for bucket in predicted_labels:
            bucket_id = int(bucket.item())
            if bucket_id in ids_per_label:
                candidate_indices.extend(ids_per_label[bucket_id])
        
        # Deduplicate candidates
        candidate_indices = list(set(candidate_indices))
        
        query_result = results.QueryResult()
        
        if not candidate_indices:
            output.queries.append(query_result)
            continue

        # Convert candidates to tensor for distance calculation
        # Using the GPU tensor we created earlier
        image_ids_in_bucket = torch.tensor(candidate_indices, device=device)
        pointers_to_images = dataset_as_tensor[image_ids_in_bucket]
        
        # Get the query vector (ensure it's on device)
        the_query = torch.from_numpy(queries_as_numpy[query_image_id]).to(device)

        # squared euclidean distances
        differences = the_query - pointers_to_images
        each_element_squared = differences ** 2
        squared_distances = each_element_squared.sum(dim=1)
        
        # best candidates
        k = min(arguments.num_nearest_neighbors, len(candidate_indices))
        smallest_squared_distances, nearest_as_indexes_within_bucket = torch.topk(
            squared_distances,
            k=k,
            largest=False,
        )
        
        # Map back to global IDs
        offsets_in_dataset = image_ids_in_bucket[nearest_as_indexes_within_bucket]
        smallest_distances = torch.sqrt(smallest_squared_distances)

        # Store results
        offsets_cpu = offsets_in_dataset.cpu().numpy()
        distances_cpu = smallest_distances.cpu().numpy()
        
        for global_id, dist in zip(offsets_cpu, distances_cpu):
            query_result.nearest_neighbors.append(
                results.NearNeighbor(id=int(global_id), distance_approximate=float(dist))
            )

        if arguments.is_range_search:
            # Re-calculate distances for all candidates for range check
            all_dists = torch.sqrt(squared_distances)
            is_r_near = torch.lt(all_dists, arguments.search_radius)
            
            ids = image_ids_in_bucket[is_r_near]
            query_result.r_near_neighbors = ids.cpu().numpy().tolist()

        output.queries.append(query_result)

    total_query_time = time.time() - starting_time


    # Brute force

    starting_time = time.time()

    cached_data = ground_truth_cache.load_ground_truth(
        arguments.input_dataset_filename,
        arguments.queries_filename,
        arguments.num_nearest_neighbors,
        arguments.search_radius,
        arguments.is_range_search
    )

    if cached_data:
        print("Using cached ground truth results.")
        cached_output, brute_time = cached_data
        
        for i, query_res in enumerate(output.queries):
            cached_query_res = cached_output.queries[i]
            
            dataset_size = dataset_as_numpy.shape[0]
            count = min(arguments.num_nearest_neighbors, dataset_size)
            if len(query_res.nearest_neighbors) < count:
                 query_res.nearest_neighbors.extend(
                    results.NearNeighbor() for _ in range(count - len(query_res.nearest_neighbors))
                )
            
            for j in range(count):
                if j < len(cached_query_res.nearest_neighbors):
                    query_res.nearest_neighbors[j].id_true = cached_query_res.nearest_neighbors[j].id_true
                    query_res.nearest_neighbors[j].distance_true = cached_query_res.nearest_neighbors[j].distance_true
            
            query_res.r_near_neighbors = cached_query_res.r_near_neighbors

    else:
        # Fixed: Pass existing_output to fill in true distances
        # Pass numpy arrays for CPU-based brute force
        brute_force.brute_force_search(
            dataset=dataset_as_numpy,
            queries=queries_as_numpy, 
            num_neighbors=arguments.num_nearest_neighbors,
            search_for_range=arguments.is_range_search,
            range_radius=arguments.search_radius,
            existing_output=output 
        )
        brute_time = time.time() - starting_time

        ground_truth_cache.save_ground_truth(
            output,
            brute_time,
            arguments.input_dataset_filename,
            arguments.queries_filename,
            arguments.num_nearest_neighbors,
            arguments.search_radius,
            arguments.is_range_search
        )


    # Fixed: Calculate Metrics using shared module
    metrics.calculate_metrics(
        output, 
        t_approximate_total=total_query_time, 
        t_true_total=brute_time
    )

    # CSV output
    if arguments.output_csv_filename is not None:
        # Calculate Speedup
        # Hardcoded brute force times from Part 1
        brute_time_per_query = 0.125492 if arguments.input_dataset_filename.lower().find('mnist') != -1 else 0.110198
        
        # Avoid division by zero
        t_approx = output.t_approximate_average if output.t_approximate_average > 0 else 1e-9
        speedup = brute_time_per_query / t_approx

        # Extract build parameters from metadata
        build_meta = state['metadata']
        k_knn = arguments.num_nearest_neighbors # This is search N, not build K. 
        
        m_blocks = build_meta['kahip']['blocks']
        epochs = build_meta['mlp']['epochs']
        # Try to get k from metadata, fallback to '?' if not present (for old indices)
        k_knn = build_meta['kahip'].get('knn_graph_k', '?')
        
        s = f"{arguments.input_dataset_filename},Neural LSH,"
        s += f"{k_knn},{m_blocks},{epochs},{arguments.multi_probe_bins},{arguments.num_nearest_neighbors},{arguments.search_radius},"
        s += f"{output.average_af:.6f},{output.recall_at_n:.6f},{output.queries_per_second:.2f},"
        s += f"{output.t_approximate_average:.6f},{output.t_true_average:.6f},{speedup:.6f}\n"
        
        with open(arguments.output_csv_filename, "a") as f:
            f.write(s)


    # The output format from the handout
    if not arguments.minimal_output and arguments.output_filename:
        output_writer.write_output(output, arguments.output_filename)


if __name__ == '__main__':
    main()
