#include <iostream>
#include <sstream>
#include <cstdint>
#include <algorithm>  // partial_sort
#include <limits>  // std::numeric_limits<double>::infinity()
#include <ivfpq.hpp>
#include <utils.hpp>


template <typename T, typename TWithNegatives, typename CodebookIndex>
Ivfpq<T, TWithNegatives, CodebookIndex>::Ivfpq(
    const Matrix<T>& dataset,
    int k_clusters,
    int dimensions_of_sub_vectors,
    int nbits,
    int seed
) :
dimensions_of_sub_vectors(dimensions_of_sub_vectors),
nbits(nbits),
seed(seed),
dataset(dataset),
coarse_clusters(dataset, k_clusters, seed) {
    // Excess dimensions aren't considered.
    // That is, if the groups are of 4 dimensions and 3 remain in the end, they won't be considered.
    pieces_of_a_vector = dataset.get_cols() / dimensions_of_sub_vectors;
}
template <typename T, typename TWithNegatives, typename CodebookIndex>
Ivfpq<T, TWithNegatives, CodebookIndex>::Ivfpq(
    const Matrix<T>& dataset,
    int k_clusters,
    int dimensions_of_sub_vectors,
    int nbits,
    int seed,
    bool should_print
) :
dimensions_of_sub_vectors(dimensions_of_sub_vectors),
nbits(nbits),
seed(seed),
should_print(should_print),
dataset(dataset),
coarse_clusters(dataset, k_clusters, seed) {

    // Excess dimensions aren't considered.
    // That is, if the groups are of 4 dimensions and 3 remain in the end, they won't be considered.
    pieces_of_a_vector = dataset.get_cols() / dimensions_of_sub_vectors;
    if (!should_print) {
        coarse_clusters.disable_printing();
    }
}

template <typename T, typename TWithNegatives, typename CodebookIndex>
void Ivfpq<T, TWithNegatives, CodebookIndex>::build() {

    if (should_print) {
        std::cout << "Building IVFPQ index for " << dataset.get_rows() << " points..." << std::endl;
    }
    coarse_clusters.set_max_build_iterations(20);
    coarse_clusters.build();

    const size_t num_coarse = coarse_clusters.get_centroids().get_rows();

    code_per_piece_per_image_per_cluster.reserve(num_coarse);

    residual_dimension_groups_per_offset.reserve(num_coarse * pieces_of_a_vector);
    // Elements are pointed to as they're pushed, with the current implementation. Reserving is necessary for that.

    for (size_t centroid_id = 0; centroid_id < num_coarse; centroid_id++) {

        const std::vector<int> & image_ids = coarse_clusters.get_image_ids_per_cluster()[centroid_id];
        auto & cpic = code_per_piece_per_image_per_cluster;
        cpic.emplace_back(image_ids.size(), pieces_of_a_vector);
        Matrix<CodebookIndex> & code_per_piece_per_image = cpic[cpic.size() - 1];
        // Used later. (If it's placed later, the image_ids will need a similar comment.)

        for (int i_piece = 0; i_piece < pieces_of_a_vector; i_piece++) {
            int first_pixel = i_piece * dimensions_of_sub_vectors;

            // Please read bottom to top.
            // Prepare for k-means clustering, which takes a Matrix<T>. Copy the equivalent pieces, AS RESIDUALS.
            std::vector<Matrix<TWithNegatives>> & rdg = residual_dimension_groups_per_offset;
            rdg.emplace_back(image_ids.size(), dimensions_of_sub_vectors);
            Matrix<TWithNegatives> & residual_i_th_pieces = rdg[rdg.size() - 1];
            for (size_t i_image = 0; i_image < image_ids.size(); i_image++) {
                TWithNegatives *residual_piece = residual_i_th_pieces.get_row(i_image);
                const T *of_image = & dataset.get_row(image_ids[i_image])[first_pixel];
                const T *of_centroid = & coarse_clusters.get_centroids().get_row(centroid_id)[first_pixel];
                for (int i = 0; i < dimensions_of_sub_vectors; i++) {
                    if constexpr(std::is_same_v<T, uint8_t>) {
                        residual_piece[i] = static_cast<TWithNegatives>(
                            static_cast<int32_t>(of_image[i]) - 
                            static_cast<int32_t>(of_centroid[i])
                        );
                    }
                    else {
                        residual_piece[i] = of_image[i] - of_centroid[i];
                    }
                }
            }

            // With the pieces compacted into a Matrix<T>, they can be used for k-means clustering.
            std::vector<Ivfflat<TWithNegatives>> & cpd = clusters_per_dimension_group_per_coarse_cluster;
            cpd.emplace_back(residual_i_th_pieces, 1u << nbits, seed);
            // Indeed the vector represents a 2x2 matrix.
            assert(cpd.size() - 1 == centroid_id * pieces_of_a_vector + i_piece);
            assert(cpd.size() == residual_dimension_groups_per_offset.size());
            // Perhaps this assertion should be debug-only.
            Ivfflat<TWithNegatives> & inner_clusters = cpd[cpd.size() - 1];
            if (should_print) {
                std::stringstream ss;
                ss << "Inner cluster " << cpd.size() << "/" << (num_coarse * pieces_of_a_vector);
                inner_clusters.set_progress_bar_label(ss.str());
                inner_clusters.stop_flickering();
            }
            else {
                inner_clusters.disable_printing();
            }
            inner_clusters.build();
            //std::cout << "Built inner clusters, printing:" << std::endl;
            //inner_clusters.print();

            // For each centroid index, place it in some codebook slots.
            // In other words, traverse the image IDs of the inner cluster and assign them this inner centroid.
            const size_t n = inner_clusters.get_centroids().get_rows();  // may not be 1 << nbits
            for (size_t inner_centroid_index = 0; inner_centroid_index < n; inner_centroid_index++) {
                
                // TODO The slides mention inverse indexes, but the current implementation doesn't reflect that need.
                
                // Put inner_centroid_index in some slots.
                const std::vector<int> & image_id = inner_clusters.get_image_ids_per_cluster()[inner_centroid_index];
                for (size_t i = 0; i < image_id.size(); i++) {
                    code_per_piece_per_image.get_row(image_id[i])[i_piece] = inner_centroid_index;
                }
            }
        }
    }
    if (should_print) {
        std::cout << "IVFPQ index build complete.                      " << std::endl;  // The spaces will go on top of the progress bar.
    }
}

static size_t sort_n(std::vector<std::tuple<double, int>> & distances, size_t num_nearest_centroids) {
    if (num_nearest_centroids <= distances.size()) {
        std::partial_sort(
            distances.begin(),
            distances.begin() + num_nearest_centroids,
            distances.end());
    
        return num_nearest_centroids;
    }
    else {
        std::sort(distances.begin(), distances.end());
        return distances.size();
    }
}

/*  Before:

  Built index: k=16 m=28 nbits=8 (build time: 181.15s)
    nprobe=1 | Recall=0.647 | Speedup=48.5x
    nprobe=4 | Recall=0.749 | Speedup=12.4x
Building IVFPQ index for 60000 points...
IVFPQ index build complete.                      ids, 0/409 centroid pixels modified                                                           
  Built index: k=16 m=49 nbits=8 (build time: 196.41s)
    nprobe=1 | Recall=0.650 | Speedup=39.8x
    nprobe=4 | Recall=0.754 | Speedup=10.0x
Building IVFPQ index for 60000 points...
IVFPQ index build complete.                      ids, 0/358 centroid pixels modified                                                           
  Built index: k=16 m=56 nbits=8 (build time: 215.54s)
    nprobe=1 | Recall=0.651 | Speedup=36.9x
    nprobe=4 | Recall=0.753 | Speedup=9.3x
Building IVFPQ index for 60000 points...
IVFPQ index build complete.                      roids, 0/179 centroid pixels modified                                                           
  Built index: k=16 m=112 nbits=8 (build time: 250.96s)
    nprobe=1 | Recall=0.650 | Speedup=24.5x
*/

template <typename T, typename TWithNegatives, typename CodebookIndex>
void Ivfpq<T, TWithNegatives, CodebookIndex>::add_local_candidates(
    std::vector<std::tuple<double, int>> & candidates, size_t insert_position,
    const T * pixels_of_query, size_t candidates_per_cluster, size_t i_centroid
) const {

    // Residual.

    const size_t pixels_per_image = dataset.get_cols();
    std::vector<TWithNegatives> residual(pixels_per_image);
    const T * coarse_centroid_pixels = coarse_clusters.get_centroids().get_row(i_centroid);
    for (size_t i = 0; i < pixels_per_image; i++) {
        if constexpr(std::is_same_v<T, uint8_t>) {
            residual[i] = static_cast<TWithNegatives>(
                static_cast<int32_t>(pixels_of_query[i]) - 
                static_cast<int32_t>(coarse_centroid_pixels[i])
            );
        }
        else {
            residual[i] = pixels_of_query[i] - coarse_centroid_pixels[i];
        }
    }

    // Pre-calculate distances to sub-centroids.

    Matrix<double> distance_to_centroids_of_pieces(pieces_of_a_vector, 1 << nbits);
    // "piece" as in group of dimensions
    for (int i_piece = 0; i_piece < pieces_of_a_vector; i_piece++) {
        const Ivfflat<TWithNegatives> & inner_clusters = clusters_per_dimension_group_per_coarse_cluster[i_centroid * pieces_of_a_vector + i_piece];
        const Matrix<TWithNegatives> & inner_centroids = inner_clusters.get_centroids();
        double * distance_to_centroids = distance_to_centroids_of_pieces.get_row(i_piece);
        TWithNegatives * residual_piece = & residual[i_piece * dimensions_of_sub_vectors];
        for (size_t i = 0; i < inner_centroids.get_rows() /* may not be 1 << nbits */; i++) {
            distance_to_centroids[i] = euclidean_distance_squared<TWithNegatives>(residual_piece, inner_centroids.get_row(i), dimensions_of_sub_vectors);
        }
    }

    // Store asymmetric distances to the images, using the sub-centroids.

    const std::vector<int>& image_ids_in_cluster = coarse_clusters.get_image_ids_per_cluster()[i_centroid];
    size_t images_in_the_cluster = image_ids_in_cluster.size();
    std::vector<std::tuple<double, int>> ds(images_in_the_cluster);

    for (size_t i_image = 0; i_image < images_in_the_cluster; i_image++) {

        double asymmetric_distance = 0;

        // "piece" as in group of dimensions

        const CodebookIndex * code_per_piece = code_per_piece_per_image_per_cluster[i_centroid].get_row(i_image);
        for (int i_piece = 0; i_piece < pieces_of_a_vector; i_piece++) {
            int code_of_x = code_per_piece[i_piece];
            asymmetric_distance += distance_to_centroids_of_pieces.unchecked(i_piece, code_of_x);
        }

        // Store local index in ds, will map to global later
        ds[i_image] = std::make_tuple(asymmetric_distance, i_image);
    }

    // Find the best candidates for this probe coarse cluster.
    // IMPORTANT: Take more candidates per cluster to compensate for PQ approximation errors
    // With high nprobe, we need to over-sample within each cluster
    size_t n = sort_n(ds, candidates_per_cluster);

    // Add these candidates to the global candidate list
    // We'll re-sort the entire list after adding candidates from this cluster

    for (size_t i = 0; i < n; i++) {
        // Map local cluster index to global dataset index
        int local_image_index = std::get<1>(ds[i]);
        int global_image_id = image_ids_in_cluster[local_image_index];
        
        // Calculate real distance using global image ID
        double real_distance_squared = euclidean_distance_squared(
            pixels_of_query, 
            dataset.get_row(global_image_id), 
            pixels_per_image
        );
        
        // Store real distance and global image ID
        if (insert_position + i < candidates.size()) {
            candidates[insert_position + i] = std::make_tuple(real_distance_squared, global_image_id);
        }
    }
}

template <typename T, typename TWithNegatives, typename CodebookIndex>
std::vector<std::tuple<double, int>> Ivfpq<T, TWithNegatives, CodebookIndex>::get_candidates(
    const T* pixels_of_query, int num_nearest_centroids, int num_results
) const {

    // Allocate enough space for candidates from all probed clusters
    // We take 5x num_results per cluster to compensate for PQ approximation errors
    size_t candidates_per_cluster = num_results * 20;
    size_t max_candidates = candidates_per_cluster * num_nearest_centroids + num_results;
    
    std::vector<std::tuple<double, int>> candidates(
        max_candidates,  // Enough space for all candidates from all probed clusters
        std::make_tuple(std::numeric_limits<double>::infinity(), -1)
    );
    const size_t pixels_per_image = dataset.get_cols();

    // Find the nearest centroids.

    const Matrix<T> & centroids = coarse_clusters.get_centroids();
    std::vector<std::tuple<double, int>> distances(centroids.get_rows());
    for (size_t i = 0; i < centroids.get_rows(); i++) {
        double d = euclidean_distance_squared(pixels_of_query, centroids.get_row(i), pixels_per_image);
        distances[i] = std::make_tuple(d, i);
    }
    const size_t min_nearest_centroids = sort_n(distances, num_nearest_centroids);

    // For each, find the nearest images.
    
    for (size_t c = 0; c < min_nearest_centroids; c++) {
        size_t i_centroid = std::get<1>(distances[c]);
        size_t insert_position = num_results + c * candidates_per_cluster;

        // This function was originally extracted to experiment by wrapping it in an if-statement.
        // It can be inlined back without drawbacks.
        add_local_candidates(candidates, insert_position, pixels_of_query, candidates_per_cluster, i_centroid);
        
        // After each cluster, re-sort to keep the best num_results at the front
        sort_n(candidates, num_results);
    }

    // Truncate and return. There may be gaps, with infinite distance and -1 index.

    candidates.resize(num_results);
    return candidates;
}

template <typename T, typename TWithNegatives, typename CodebookIndex>
void Ivfpq<T, TWithNegatives, CodebookIndex>::print() const {

    std::cout << "=== coarse clusters ===" << std::endl;
    coarse_clusters.print();
    std::cout << std::endl;
    size_t total_coarse = coarse_clusters.get_centroids().get_rows();
    for (size_t coarse_cluster = 0; coarse_cluster < total_coarse; coarse_cluster++) {
        for (int dimension_group = 0; dimension_group < pieces_of_a_vector; dimension_group++) {
            std::cout << "=== inner cluster, coarse = " << coarse_cluster
            << ", dimension group = " << dimension_group << " ===" << std::endl;
            size_t idx = coarse_cluster * pieces_of_a_vector + dimension_group;
            clusters_per_dimension_group_per_coarse_cluster[idx].print();
            std::cout << std::endl;
        }
    }
}

// Explicit template instantiations
template class Ivfpq<float, float, uint8_t>;
template class Ivfpq<float, float, uint32_t>;
template class Ivfpq<uint8_t, int8_t, uint8_t>;
template class Ivfpq<uint8_t, int8_t, uint32_t>;
