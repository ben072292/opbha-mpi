#include "lattice_model.h"
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <thread>
#include <omp.h>
#include <algorithm>
#include <random>
using namespace std;

bool is_done(bool* checked_set, int size, int world_rank){
    if(world_rank == 0){
        for(int i = 0; i < size; i++){
            if(!checked_set[i]) return false;
        }
        return true;
    }
    return false;
}

void select_n(bool* checked_set, int* table, int size, int select_size, vector<int>& ret){
    int counter = 0;
    for(int i = 0; i < size; i++){
        if(!checked_set[table[i]]) ret[counter++] = table[i];
        if(counter == select_size) break;
    }
}

int main(int argc, char* argv[]){
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    int pool_size = atoi(argv[1]);
    double prior_val = atof(argv[2]);
    omp_set_num_threads(atoi(argv[3]));
    double* prior = new double[pool_size];
    for(int i = 0; i < pool_size; i++) prior[i] = prior_val;
    pool_stat* pool = new pool_stat(pool_size, prior);
    lattice_model* model;

    double run_time = 0.0;

    if(world_rank == 0) {
        model = new lattice_model(pool, true);
        double** dilution_matrix = model->generate_dilution_matrix(0.99, 0.005);
    }
    else{
        model = new lattice_model(pool, false);
        model->set_posterior_probability_map((double*)malloc(sizeof(double) * (1 << pool_size)));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    run_time -= MPI_Wtime();
    MPI_Bcast(model->get_posterior_probability_map(), (1 << pool_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int select_size = (1 << (pool_size-6)); // 1/64
    int* sophsticated_table;
    bool* checked_set = (bool*)malloc(sizeof(bool) * (1 << pool_size));
    vector<int> select;
    auto rng = default_random_engine {};
    int sub_select_size = select_size / world_size;
    int* sub_select = (int*)malloc(sizeof(int) * sub_select_size);
    bool done = false;
    double local_min = 2.0;
    double global_min = 2.0;

    // initialize rank 0 variables
    if(world_rank == 0){
        select.resize(select_size, 0);
        sophsticated_table = model->sophisticated_selection();
    }
    while(!done){
    // for(int k = 0; k < 3; k++){
        if (world_rank == 0) {
            select_n(checked_set, sophsticated_table, 1<<pool_size, select_size, select);
            std::shuffle(begin(select), end(select), rng); // shuffle for better load balance
        }
        // Scatter states to all processes
        // MPI_Scatterv will be supported in the future for varied dispatch size
        MPI_Scatter(select.data(), sub_select_size, MPI_INT, sub_select, sub_select_size, MPI_INT, 0, MPI_COMM_WORLD);
        
        // auto start = chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < sub_select_size; i++){
            int state = sub_select[i];
            if(checked_set[state]) continue;
            double val = model->get_up_set_mass(state);
            if(val < 0.5){
                int* up_set = model->get_up_set(state);
                for(int j = 0; j < (1 << (pool_size-__builtin_popcount(state))); j++) checked_set[up_set[j]] = true;
                delete[] up_set;
            }
            else if (val > 0.5){
                int* down_set = model->get_down_set(state);
                for(int j = 0; j < (1 << __builtin_popcount(state)); j++) checked_set[down_set[j]] = true;
                delete[] down_set;
            }
            double temp = abs(val - 0.5);
            if(temp < local_min){
                local_min = temp;
            }
        }
        // auto stop = chrono::high_resolution_clock::now();
        // auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        // cout << "Time: " << duration.count() << endl;
        
        // Reduce all of the local checked sets into process 0
        // in place reduce
        if(world_rank == 0){
            MPI_Reduce(MPI_IN_PLACE, checked_set, 1<<pool_size, MPI_CXX_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
        }
        else{
            MPI_Reduce(checked_set, checked_set, 1<<pool_size, MPI_CXX_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
        }

        // Reduce all of the local min into the global min
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

        local_min = global_min;

        MPI_Bcast(&local_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // if(world_rank == 0) cout << "Temp min: " << global_min << endl;

        MPI_Bcast(checked_set, 1 << pool_size, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

        done = is_done(checked_set, 1 << pool_size, world_rank);
        // done = true;
        // this_thread::sleep_for(std::chrono::milliseconds(1000));
        MPI_Bcast(&done, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    run_time += MPI_Wtime();

    if(world_rank == 0){
        cout << "Min: " << global_min << endl << "Time: " << run_time << "s" << endl;
    }

    // final cleanup
    if(world_rank == 0){
        delete[] sophsticated_table;
        // free(select);
    }
    else model->free_posterior_probability_map();
    
    free(checked_set);

    // Finalize the MPI environment.
    MPI_Finalize();
}