#include <cstdint>
#include <iostream>
#include <limits> // numeric_limits<double>::infinity()
#include <random>

// For querying only:
#include <algorithm> // partial_sort, min
#include <tuple>

#include <ivfflat.hpp>
#include <utils.hpp>

//     COMPILED, NOT RAN

template <typename T>
Ivfflat<T>::Ivfflat(const Matrix<T>& dataset, int k_clusters, int seed)
    : seed(seed), pixels_per_image(dataset.get_cols()),
      worth_clustering(dataset.get_rows() > static_cast<size_t>(k_clusters)), dataset(dataset),
      centroids(worth_clustering ? k_clusters : 1, dataset.get_cols()),
      image_ids_per_cluster(k_clusters) {
    changed_centroid_pixels_that_would_end_build = (k_clusters * pixels_per_image) * 10 / 100;
}

// Copy random images, from the dataset, to the centroids of k-means clustering.
// At first each centroid will be identical to some image,
// but as the cluster they represent changes, they wouldn't be replacable with pointers.

template <typename T> void Ivfflat<T>::initialize_centroids_as_random_images(int seed) {
    std::mt19937 random_generator{static_cast<unsigned int>(seed)};
    std::uniform_int_distribution<int> distribution(0, dataset.get_rows() - 1);

    for (size_t centroid_id = 0; centroid_id < centroids.get_rows(); centroid_id++) {
        int random_image_id = distribution(random_generator);

        T* centroid_pixels = centroids.get_row(centroid_id);
        const T* pixels_of_random_image = dataset.get_row(random_image_id);
        for (int i = 0; i < pixels_per_image; i++) {
            centroid_pixels[i] = pixels_of_random_image[i];
        }
    }
}

// Group the images by nearest centroid. The groups are lists of image IDs.

template <typename T> void Ivfflat<T>::group_by_nearest_centroid() {

    for (size_t i_image = 0; i_image < dataset.get_rows(); i_image++) {
        double smallest_square_yet = std::numeric_limits<double>::infinity();
        int closest_centroid = -1;
        for (size_t i_centroid = 0; i_centroid < centroids.get_rows(); i_centroid++) {
            double d_squared = eucl_d_sq_if_smaller_else_inf(centroids.get_row(i_centroid),
                                                             dataset.get_row(i_image),
                                                             pixels_per_image, smallest_square_yet);

            if (d_squared < smallest_square_yet) {
                smallest_square_yet = d_squared;
                closest_centroid = i_centroid;
            }
        }
        image_ids_per_cluster[closest_centroid].push_back(i_image);
    }
}

// PERFORMANCE SPEED-UP IDEAS
// 1. eucl_dist_squared_if_smaller(vec_1, vec_2, dim, other_distance) { ... if (sum >
// other_distance) return -1.0; }
// 2. auto arr = std::make_unique<int[]>(dataset.get_rows());

template <typename T> void Ivfflat<T>::build() {

    // Can the dataset be empty? The class is used by IVFPQ for the inner centroids, which are
    // chosen automatically. It does happen.
    if (dataset.get_rows() == 0) {
        image_ids_per_cluster.emplace_back(); // TODO This may be unnecessary.
        return;
    }
    // If the images are too few, there will only be one cluster.
    // The benefit comes from IVFPQ, therefore an alternative solution
    // is that IVFPQ has either instances of this class or instances of another as appropriate.
    else if (!worth_clustering) {
        T* centroid_pixels = centroids.get_row(0);
        const T* first_image = dataset.get_row(0);
        for (int i = 0; i < pixels_per_image; i++) {
            centroid_pixels[i] = first_image[i];
        }
    } else {
        initialize_centroids_as_random_images(seed);
    }

    group_by_nearest_centroid();

    // Make the centroids the average of their cluster and reconsider their cluster.
    // Repeat that until all the centroids remain similar to the last iteration.
    // NOTE: For benchmarking large datasets, we use early stopping at 10 iterations

    for (int revisions_of_centroids = 0; revisions_of_centroids < max_centroid_revisions;
         revisions_of_centroids++) {
        int number_of_changed_centroid_pixels = 0;

        // Averaging each cluster into its new centroid

        for (size_t i_centroid = 0; i_centroid < centroids.get_rows(); i_centroid++) {
            const auto& images = image_ids_per_cluster[i_centroid];
            if (images.empty())
                continue; // Skip empty clusters

            for (int i_pixel = 0; i_pixel < pixels_per_image; i_pixel++) {

                // TODO a template function with two types (ex. uint8_t, uint32_t) could refactor
                // this code. It's harder than it looks however.

                if constexpr (std::is_same_v<T, uint8_t>) {
                    uint32_t sum_of_pixel_values = 0;
                    for (size_t i_image = 0; i_image < images.size(); i_image++) {
                        sum_of_pixel_values += dataset.get_row(images[i_image])[i_pixel];
                    }
                    uint32_t average = sum_of_pixel_values / images.size();
                    T* centroid_pixel = &(centroids.get_row(i_centroid)[i_pixel]);
                    uint32_t last_value = static_cast<uint32_t>(*centroid_pixel);
                    if (average != last_value) {
                        number_of_changed_centroid_pixels++;
                    }
                    *centroid_pixel = static_cast<uint8_t>(average);
                } else if constexpr (std::is_same_v<T, int8_t>) {
                    double sum_of_pixel_values = 0;
                    for (size_t i_image = 0; i_image < images.size(); i_image++) {
                        sum_of_pixel_values += dataset.get_row(images[i_image])[i_pixel];
                    }
                    int32_t average = sum_of_pixel_values / images.size();
                    T* centroid_pixel = &(centroids.get_row(i_centroid)[i_pixel]);
                    int32_t last_value = static_cast<int32_t>(*centroid_pixel);
                    if (average != last_value) {
                        number_of_changed_centroid_pixels++;
                    }
                    *centroid_pixel = static_cast<int8_t>(average);
                } else if constexpr (std::is_same_v<T, float>) {
                    double sum_of_pixel_values = 0;
                    for (size_t i_image = 0; i_image < images.size(); i_image++) {
                        sum_of_pixel_values += dataset.get_row(images[i_image])[i_pixel];
                    }
                    double average = sum_of_pixel_values / images.size();
                    T* centroid_pixel = &(centroids.get_row(i_centroid)[i_pixel]);
                    double last_value = static_cast<double>(*centroid_pixel);
                    if (average != last_value) {
                        number_of_changed_centroid_pixels++;
                    }
                    *centroid_pixel = static_cast<float>(average);
                }
            }
        }

        if (should_print) {
            std::cout << label << " revision " << (revisions_of_centroids + 1) << "/"
                      << max_centroid_revisions << " of the centroids, "
                      << number_of_changed_centroid_pixels << "/"
                      << changed_centroid_pixels_that_would_end_build
                      << " centroid pixels modified                                                "
                         "        \r";
            std::cout.flush();
        }

        if (number_of_changed_centroid_pixels <= changed_centroid_pixels_that_would_end_build)
            break;

        // Reconsidering the clusters.

        for (size_t i_centroid = 0; i_centroid < centroids.get_rows(); i_centroid++) {
            image_ids_per_cluster[i_centroid]
                .clear(); // TODO Reallocations may cause a noticable performance hit
        }

        group_by_nearest_centroid();
    }
    if (should_print && should_clear_progress) {
        std::cout << "                                                                             "
                     "                \r";
        std::cout.flush();
    }
}

static void sort_n_tuples(std::vector<std::tuple<double, int>>& v, size_t n) {
    auto last_to_sort = (n > v.size() ? v.end() : v.begin() + n);
    std::partial_sort(v.begin(), last_to_sort, v.end());
}

template <typename T>
std::vector<std::tuple<double, int>> Ivfflat<T>::get_candidates(const T* pixels_of_query,
                                                                int num_nearest_clusters,
                                                                int num_results) const {

    // Find the nearest clusters.

    std::vector<std::tuple<double, int>> distances_and_centroid_ids;
    // TODO This memory doesn't need to be allocated and deallocated per query, it may be large and
    // there will be many queries.
    distances_and_centroid_ids.reserve(centroids.get_rows());
    for (size_t centroid_id = 0; centroid_id < centroids.get_rows(); centroid_id++) {
        double d = euclidean_distance_squared(pixels_of_query, centroids.get_row(centroid_id),
                                              pixels_per_image);
        distances_and_centroid_ids.push_back(std::make_tuple(d, centroid_id));
    }

    sort_n_tuples(distances_and_centroid_ids, num_nearest_clusters);
    // Defined above.

    std::vector<std::tuple<double, int>> distances_and_image_ids;
    // TODO This memory doesn't need to be allocated and deallocated per query, it may be large and
    // there will be many queries.
    int images_in_the_clusters = 0;
    for (int i = 0;
         i < num_nearest_clusters && static_cast<size_t>(i) < distances_and_centroid_ids.size();
         i++) {
        auto [unused_distance, centroid_id] = distances_and_centroid_ids[i];
        images_in_the_clusters += image_ids_per_cluster[centroid_id].size();
    }
    distances_and_image_ids.reserve(images_in_the_clusters);
    for (int i = 0;
         i < num_nearest_clusters && static_cast<size_t>(i) < distances_and_centroid_ids.size();
         i++) {
        auto [unused_distance, centroid_id] = distances_and_centroid_ids[i];
        for (int image_id : image_ids_per_cluster[centroid_id]) {
            double d = euclidean_distance_squared(pixels_of_query, dataset.get_row(image_id),
                                                  pixels_per_image);
            distances_and_image_ids.push_back(std::make_tuple(d, image_id));
        }
    }

    // Return top num_results. There may be empty values.

    sort_n_tuples(distances_and_image_ids, num_results);
    distances_and_image_ids.resize(num_results,
                                   std::make_tuple(std::numeric_limits<double>::infinity(),
                                                   -1) // TODO What should the default value be?
    );
    return distances_and_image_ids; // Modern compilers try to avoid copying memory in this case.
}

template <typename T>
double Ivfflat<T>::average_distance(const T* image, const std::vector<int>& image_ids) const {
    double sum_of_distances = 0.0;
    for (int id : image_ids) {
        sum_of_distances +=
            euclidean_distance_squared(image, dataset.get_row(id), pixels_per_image);
    }
    return sum_of_distances / image_ids.size();
}

template <typename T> double Ivfflat<T>::get_silhouette() const {

    double sum_of_silhouettes = 0.0;

    // Caching would need to be clever to be practical. It might not be possible to cache
    // the distance between any 2 images. The closest centroid to another centroid may not
    // be the closest for all the images in the cluster. The list of images in a cluster
    // can be used to avoid some checks, but there's no list of next-clusters.
    // Maybe build() can be modified to pre-calculate every image's next closest cluster.

    for (size_t i_image = 0; i_image < dataset.get_rows(); i_image++) {

        if (should_print && (i_image + 1) % 100 == 0) {
            std::cout << label << " get_silhouette: image " << i_image + 1 << "/"
                      << dataset.get_rows() << "\r";
            std::cout.flush();
        }

        size_t nearest_cluster = -1;
        size_t second_nearest = -1;
        double distance_to_nearest_cluster = std::numeric_limits<double>::infinity();
        double distance_to_second_nearest = std::numeric_limits<double>::infinity();
        for (size_t i_cluster = 0; i_cluster < centroids.get_rows(); i_cluster++) {
            double d = eucl_d_sq_if_smaller_else_inf(dataset.get_row(i_image),
                                                     centroids.get_row(i_cluster), pixels_per_image,
                                                     distance_to_second_nearest);
            if (d < distance_to_nearest_cluster) {
                distance_to_second_nearest = distance_to_nearest_cluster;
                distance_to_nearest_cluster = d;
                second_nearest = nearest_cluster;
                nearest_cluster = i_cluster;
            } else if (d < distance_to_second_nearest) {
                distance_to_second_nearest = d;
                second_nearest = i_cluster;
            }
        }
        const T* img = dataset.get_row(i_image);
        double a = average_distance(img, image_ids_per_cluster[nearest_cluster]);
        double b = average_distance(img, image_ids_per_cluster[second_nearest]);
        double s = 0.0;
        if (a < b)
            s = 1 - a / b;
        else if (a > b)
            s = b / a - 1;
        sum_of_silhouettes += s;
    }

    if (should_print) {
        std::cout << "                                                               \r";
        std::cout.flush();
    }
    return sum_of_silhouettes / static_cast<double>(centroids.get_rows());
}

template <typename T> void Ivfflat<T>::print() const {
    for (size_t c = 0; c < centroids.get_rows(); c++) {
        std::cout << "centroid " << c << ":\t";
        for (int i = 0; i < pixels_per_image; i++) {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                std::cout << static_cast<int>(centroids.at(c, i)) << "\t";
            } else {
                std::cout << centroids.at(c, i) << "\t";
            }
        }
        std::cout << std::endl;
        const std::vector<int>& ids = image_ids_per_cluster[c];
        for (size_t d = 0; d < ids.size(); d++) {
            std::cout << "  image id " << ids[d] << " (\t";
            for (int i = 0; i < pixels_per_image; i++) {
                if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                    std::cout << static_cast<int>(dataset.at(ids[d], i)) << "\t";
                } else {
                    std::cout << dataset.at(ids[d], i) << "\t";
                }
            }
            std::cout << "\t)" << std::endl;
        }
    }
}

// Explicit template instantiations
template class Ivfflat<float>;
template class Ivfflat<int8_t>; // Used by IVFPQ, residuals need negative values.
template class Ivfflat<uint8_t>;
