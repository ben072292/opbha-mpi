#ifndef POOL_STAT_H
#define POOL_STAT_H
class pool_stat{
    public:
    pool_stat(int pool_size, double* prior);

    ~pool_stat();

    int get_pool_size(){return pool_size;}

    double* get_prior(){return prior;}

    double* generate_prior_probability_map();

    double generate_prior_probability(int state);

    private: 
    int pool_size;
    double *prior;

};
#endif