#pragma once

#include <string>
#include <data_types.hpp>

// COMPILED, NOT RAN

template <typename T>
class Ivfflat {
public:
    Ivfflat(const Matrix<T> & dataset, int k_clusters, int seed);
    void set_how_many_centroid_pixels_end_the_build(int n) { changed_centroid_pixels_that_would_end_build = n; }
    void set_max_build_iterations(int n) { max_centroid_revisions = n; }
    void disable_printing() { should_print = false; }
    void stop_flickering() { should_clear_progress = false; }
    void set_progress_bar_label(std::string s) { label = s; }
    void build();
    std::vector<std::tuple<double, int>> get_candidates(const T* pixels_of_query, int num_nearest_clusters, int num_results) const;
    double get_silhouette() const;
    
    void print() const;
    const Matrix<T> & get_centroids() const { return centroids; }
    const std::vector<std::vector<int>> & get_image_ids_per_cluster() const { return image_ids_per_cluster; }

private:
    void initialize_centroids_as_random_images(int seed);
    void group_by_nearest_centroid();
    double average_distance(const T * image, const std::vector<int> & image_ids) const;

    const int seed;
    const int pixels_per_image;
    const bool worth_clustering;
    int changed_centroid_pixels_that_would_end_build = 1;
    int max_centroid_revisions = 10;
    bool should_print = true;
    bool should_clear_progress = true;
    std::string label = "IVF-Flat";
    const Matrix<T> & dataset;
    Matrix<T> centroids;
    std::vector<std::vector<int>> image_ids_per_cluster;  // TODO maybe pre-allocating the max size is better for many iterations
};