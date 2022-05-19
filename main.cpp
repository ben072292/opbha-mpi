#include "lattice_model.h"
#include <chrono>
#include <iostream>
#include <cstdlib>
using namespace std;

int main(int argc, char* argv[]){
    int size = atoi(argv[1]);
    double prior_val = atof(argv[2]);
    double* prior = new double[size];
    for(int i = 0; i < size; i++) prior[i] = prior_val;
    pool_stat* pool = new pool_stat(size, prior);
    auto start = chrono::high_resolution_clock::now();
    lattice_model* model = new lattice_model(pool, true);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Build: " << duration.count() << endl;

    double** dilution_matrix = model->generate_dilution_matrix(0.99, 0.005);
    start = chrono::high_resolution_clock::now();
    model->update_posterior_probability((1<<size)-1, 1, dilution_matrix);
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Update: " << duration.count() << endl;

    start = chrono::high_resolution_clock::now();
    //model->update_posterior_probability(1, 1, dilution_matrix);
    cout << model->find_halving_state_openmp_static_scheduler(0.5) << endl;
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "bha static: " << duration.count() << endl;

    start = chrono::high_resolution_clock::now();
    //model->update_posterior_probability(1, 1, dilution_matrix);
    cout << model->find_halving_state_openmp_dynamic_scheduler(0.5) << endl;
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "bha dynamic: " << duration.count() << endl;

    start = chrono::high_resolution_clock::now();
    //model->update_posterior_probability(1, 1, dilution_matrix);
    cout << model->find_halving_state(0.5) << endl;
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "bha single thread: " << duration.count() << endl;


}