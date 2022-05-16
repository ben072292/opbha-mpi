#ifndef LATTICE_MODEL_H
#define LATTICE_MODEL_H

#include "pool_stat.h"
#include <iostream>
#include <vector>
class lattice_model{
    private:
    int pool_size;
    double* posterior_probability_map;

    public:
    lattice_model(pool_stat *pool_stat, bool gen_prior);
    ~lattice_model();
    double* get_posterior_probability_map(){return posterior_probability_map;}
    void set_posterior_probability_map(double* val){posterior_probability_map = val;}
    void delete_posterior_probability_map(){delete[] posterior_probability_map;}
    void free_posterior_probability_map(){free(posterior_probability_map);}
    int* get_up_set(int state);
    int* generate_power_set_adder(int* add_index, int state);
    int* get_down_set(int state);
    int* generate_power_set_reducer(int* reduce_index, int state);
    double* update_posterior_probability(int state, int response, double** dilution_matrix);
    double** generate_dilution_matrix(double alpha, double h);
    double compute_dilution_response(double** dilution_matrix, int e, int state);
    int find_halving_state(double prob);
    int find_halving_state_openmp_static_scheduler(double prob);
    int find_halving_state_openmp_dynamic_scheduler(double prob);
    double get_up_set_mass(int state);
    int* sophisticated_selection();
    int binomial(int N, int K);

};

#endif