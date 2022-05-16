#include "pool_stat.h"

using namespace std;
pool_stat::pool_stat(int pool_size, double* prior){
    this->pool_size = pool_size;
    this->prior = prior;
}

pool_stat::~pool_stat(){
    delete[] prior;
}

double* pool_stat::generate_prior_probability_map(){
    double* ret = new double[(1 << pool_size)];
    for(int i = 0; i < (1 << pool_size); i++){
        ret[i] = generate_prior_probability(i);
    }
    return ret;
}

double pool_stat::generate_prior_probability(int state){
    double ret = 1.0;
    for(int i = pool_size-1; i >=0; i--){
        if((state & 1) == 1) ret *= (1-prior[i]);
        else ret *= prior[i];
        state = state >> 1;
    }
    return ret;
}
