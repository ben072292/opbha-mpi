#include <iostream>
#include <stdlib.h>
#include <vector>
#include "lattice_model.h"

lattice_model::lattice_model(pool_stat *pool_stat, bool gen_prior){
    pool_size = pool_stat->get_pool_size();
    if(gen_prior) posterior_probability_map = pool_stat->generate_prior_probability_map();
}

lattice_model::~lattice_model(){
    delete[] posterior_probability_map;
}

int* lattice_model::get_up_set(int state){
    int* add_index = new int[pool_size-__builtin_popcount(state)];
    int counter=0, i, index;
    for(i = 0; i < pool_size; i++){
        index = (1 << i);
        if((state & index) == 0) add_index[counter++] = index;
    }
    return generate_power_set_adder(add_index, state);
}

int* lattice_model::generate_power_set_adder(int* add_index, int state){
    int n = pool_size-__builtin_popcount(state), pow_set_size = 1 << n;
    int* ret = new int[pow_set_size];
    int i, j, temp;
    for(i= 0; i < pow_set_size; i++){
        temp = state;
        for(j = 0; j < n; j++){
            if((i&(1<<j)) > 0) temp += add_index[j];
        }
        ret[i] = temp;
    }
    delete[] add_index;
    return ret;
}

int* lattice_model::get_down_set(int state){
    int* reduce_index = new int[__builtin_popcount(state)];
    int counter = 0, index;
    for(int i = 0; i < pool_size; i++){
        index = (1<<i);
        if((state&index) == index) reduce_index[counter++] = index;
    }
    return generate_power_set_reducer(reduce_index, state);
}

int* lattice_model::generate_power_set_reducer(int* reduce_index, int state){
    int n = __builtin_popcount(state), pow_set_size = (1 << n);
    int* ret = new int[pow_set_size];
    int i, j, temp;
    for(i= 0; i < pow_set_size; i++){
        temp = state;
        for(j = 0; j < n; j++){
            if((i&(1<<j)) > 0) temp -= reduce_index[j];
        }
        ret[i] = temp;
    }
    delete[] reduce_index;
    return ret;
}

double* lattice_model::update_posterior_probability(int state, int response, double** dilution_matrix){
    double* ret = new double[1<<pool_size];
    double denominator = 0.0;
    if(response == 1){
        for(int i = 0; i < (1 << pool_size); i++){
            ret[i] = compute_dilution_response(dilution_matrix, i, state) * posterior_probability_map[i];
            denominator += ret[i];
        }
        for(int i = 0; i < (1 << pool_size); i++){
            ret[i] /= denominator;
        }
    }
    else if(response == 0){
        for(int i = 0; i < (1 << pool_size); i++){
            ret[i] = (1.0-compute_dilution_response(dilution_matrix, i, state)) * posterior_probability_map[i];
            denominator += ret[i];
        }
        for(int i = 0; i < (1 << pool_size); i++){
            ret[i] /= denominator;
        }
    }
    delete[] posterior_probability_map;
    return ret;
}

double** lattice_model::generate_dilution_matrix(double alpha, double h){
    double** ret = new double*[pool_size];
    int k;
    for(int rk = 1; rk <= pool_size; rk++){
        ret[rk-1] = new double[rk+1];
        ret[rk-1][0] = alpha;
        for(int r = 1; r <= rk; r++){
            k = rk-r;
            ret[rk-1][r] = 1-alpha * r / (k * h + r);
        }
    }
    return ret;
}

double lattice_model::compute_dilution_response(double** dilution_matrix, int e, int state){
    int index = __builtin_popcount(state);
    return dilution_matrix[index-1][index-__builtin_popcount(e & state)];
}

int lattice_model::find_halving_state(double prob){
    int ret = 0, size = (1 << pool_size);
    double thres = 2.0;
    double val, temp;
    std::vector<bool> checked_set(size);
    int* sophisticated_selection_table = sophisticated_selection();
    for(int i = 0; i < size; i++){
        int state = sophisticated_selection_table[i];
        if(checked_set[state]) continue;
        val = get_up_set_mass(state);
        if(val < prob){
            int* up_set = get_up_set(state);
            for(int j = 0; j < (1 << (pool_size-__builtin_popcount(state))); j++) checked_set[up_set[j]] = true;
            delete[] up_set;
        }
        else if (val > prob){
            int* down_set = get_down_set(state);
            for(int j = 0; j < (1 << __builtin_popcount(state)); j++) checked_set[down_set[j]] = true;
            delete[] down_set;
        }
        temp = abs(val - prob);
        if(temp < thres){
            thres = temp;
            ret = state;
        }
    }
    return ret;
}

int lattice_model::find_halving_state_openmp_static_scheduler(double prob){
    int ret = 0, size = (1 << pool_size);
    double thres = 2.0;
    std::vector<bool> checked_set(size);
    int* sophisticated_selection_table = sophisticated_selection();

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        int state = sophisticated_selection_table[i];
        if(checked_set[state]) continue;
        double val = get_up_set_mass(state);
        if(val < prob){
            int* up_set = get_up_set(state);
            for(int j = 0; j < (1 << (pool_size-__builtin_popcount(state))); j++) checked_set[up_set[j]] = true;
            delete[] up_set;
        }
        else if (val > prob){
            int* down_set = get_down_set(state);
            for(int j = 0; j < (1 << __builtin_popcount(state)); j++) checked_set[down_set[j]] = true;
            delete[] down_set;
        }
        double temp = abs(val - prob);
        #pragma omp critical
        {
            if(temp < thres){
                thres = temp;
                ret = state;
            }
        }
    }
    return ret;
}

int lattice_model::find_halving_state_openmp_dynamic_scheduler(double prob){
    int ret = 0, size = (1 << pool_size);
    double thres = 2.0;
    std::vector<bool> checked_set(size);
    int* sophisticated_selection_table = sophisticated_selection();

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < size; i++){
        int state = sophisticated_selection_table[i];
        if(checked_set[state]) continue;
        double val = get_up_set_mass(state);
        if(val < prob){
            int* up_set = get_up_set(state);
            for(int j = 0; j < (1 << (pool_size-__builtin_popcount(state))); j++) checked_set[up_set[j]] = true;
            delete[] up_set;
        }
        else if (val > prob){
            int* down_set = get_down_set(state);
            for(int j = 0; j < (1 << __builtin_popcount(state)); j++) checked_set[down_set[j]] = true;
            delete[] down_set;
        }
        double temp = abs(val - prob);
        #pragma omp critical
        {
            if(temp < thres){
                thres = temp;
                ret = state;
            }
        }
    }
    return ret;
}

double lattice_model::get_up_set_mass(int state){
    if(state == 0) return 1.0;
    int i, j, index = 0, temp = 0, n = pool_size-__builtin_popcount(state), pow_set_size= (1 << n);
    double ret = 0.0;
    int* add_index = new int[n];
    for(i = 0; i < pool_size; i++){
        index = (1 << i);
        if((state & index) == 0) add_index[temp++] = index;
    }
    for(i = 0; i < pow_set_size; i++){
        temp = state;
        for(j = 0; j < n; j++){
            if((i & (1 << j)) > 0) temp += add_index[j];
        }
        ret += posterior_probability_map[temp];
    }
    delete[] add_index;
    return ret;
}

int* lattice_model::sophisticated_selection(){
    int i, temp = 0;
    int* ret = new int[(1 << pool_size)]{0};
    int* counter = new int[pool_size+1]{0};
    int* real_start_pos = new int[pool_size+1]{0};
    for(i = 0; i <= pool_size-i; i++){
        real_start_pos[i] = temp;
        temp += binomial(pool_size, i);
        if(pool_size-i != i){
            real_start_pos[pool_size-i] = temp;
            temp += binomial(pool_size, pool_size-i);
        }
    }
    for(i = 0; i < (1 << pool_size); i++){
        int count = __builtin_popcount(i);
        ret[(real_start_pos[count] + counter[count]++)] = i;
        
    }
    delete[] counter;
    delete[] real_start_pos;
    return ret;
}

int lattice_model::binomial(int n, int k){
    if (k == 0) return 1;
	return (n * binomial(n - 1, k - 1)) / k;
}

    

