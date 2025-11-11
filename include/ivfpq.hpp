#pragma once

#include <data_types.hpp>
#include <ivfflat.hpp>
#include <random>

template <typename T, typename TWithNegatives /* one bit shorter is okay */, typename CodebookIndex>
// TWithNegatives would be replaced with constexpr if it wasn't for the private members below.
class Ivfpq {
  public:
    Ivfpq(const Matrix<T>& dataset, int k_clusters, int dimensions_of_sub_vectors, int nbits,
          int seed);
    Ivfpq(const Matrix<T>& dataset, int k_clusters, int dimensions_of_sub_vectors, int nbits,
          int seed, bool should_print);
    void build();
    std::vector<std::tuple<double, int>>
    get_candidates(const T* pixels_of_query, int num_nearest_centroids, int num_results) const;
    double get_silhouette() const {
        return coarse_clusters.get_silhouette();
    }
    void print() const;

  private:
    const int dimensions_of_sub_vectors;
    int pieces_of_a_vector;
    const int nbits;
    const int seed;
    const bool should_print = true;
    const Matrix<T>& dataset;
    Ivfflat<T> coarse_clusters;

    std::vector<Matrix<TWithNegatives>> residual_dimension_groups_per_offset;

    // std::vector used as Matrix (2D), need std::vector::emplace_back
    std::vector<Ivfflat<TWithNegatives>> clusters_per_dimension_group_per_coarse_cluster;

    std::vector<Matrix<CodebookIndex>> code_per_piece_per_image_per_cluster;
    // a[coarse_cluster].get_row(image)[dimension_group] = centroid

    void add_local_candidates(std::vector<std::tuple<double, int>>& candidates,
                              size_t insert_position, const T* pixels_of_query,
                              size_t candidates_per_cluster, size_t i_centroid) const;
};
