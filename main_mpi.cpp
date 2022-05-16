#include "lattice_model.h"
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <thread>
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

int* select_n(bool* checked_set, int* table, int size, int select_size){
    int* ret = new int[select_size]{0};
    int counter = 0;
    for(int i = 0; i < size; i++){
        if(!checked_set[table[i]]) ret[counter++] = table[i];
        if(counter == select_size) break;
    }
    return ret;
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
    double* prior = new double[pool_size];
    for(int i = 0; i < pool_size; i++) prior[i] = prior_val;
    pool_stat* pool = new pool_stat(pool_size, prior);
    lattice_model* model;

    double total_mpi_bcast_time = 0.0;

    if(world_rank == 0) {
        model = new lattice_model(pool, true);
        double** dilution_matrix = model->generate_dilution_matrix(0.99, 0.005);
    }
    else{
        model = new lattice_model(pool, false);
        model->set_posterior_probability_map((double*)malloc(sizeof(double) * (1 << pool_size)));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time -= MPI_Wtime();
    MPI_Bcast(model->get_posterior_probability_map(), (1 << pool_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int select_size = (1 << (pool_size-5)); // 1/64
    int* sophsticated_table;
    bool* checked_set = (bool*)malloc(sizeof(bool) * (1 << pool_size));
    bool* global_checked_set;
    int* select;
    int sub_select_size = select_size / world_size;
    int* sub_select = (int*)malloc(sizeof(int) * sub_select_size);

    bool done = false;
    double local_min = 2.0;
    double global_min = 2.0;

    if (world_rank == 0) {
        sophsticated_table = model->sophisticated_selection();
    }
    while(!done){
    // for(int k = 0; k < 3; k++){
        if (world_rank == 0) {
            select = select_n(checked_set, sophsticated_table, 1<<pool_size, select_size);
        }
        // Scatter states to all processes
        MPI_Scatter(select, sub_select_size, MPI_INT, sub_select, sub_select_size, MPI_INT, 0, MPI_COMM_WORLD);
        
        // if (world_rank == 0){
        //     cout << "0: ";
        //     for(int i = 0; i < sub_select_size; i++){
        //         cout << sub_select[i] << " ";
        //     }
        //     cout << endl << endl;
        // }

        // if (world_rank == 1){
        //     cout << "1: ";
        //     for(int i = 0; i < sub_select_size; i++){
        //         cout << sub_select[i] << " ";
        //     }
        //     cout << endl << endl;
        // }
        
        double val = 2.0, temp = 2.0;
        for(int i = 0; i < sub_select_size; i++){
            int state = sub_select[i];
            if(checked_set[state]) continue;
            val = model->get_up_set_mass(state);
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
            temp = abs(val - 0.5);
            if(temp < local_min){
                local_min = temp;
            }
        }

        if(world_rank == 0){
            if(!global_checked_set) free(global_checked_set);
            global_checked_set = (bool*)malloc(sizeof(bool) * (1 << pool_size));
        }
        // // Reduce all of the local checked sets into the global checked set
        MPI_Reduce(checked_set, global_checked_set, 1<<pool_size, MPI_CXX_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
        
        
        if(world_rank == 0){
            delete[] select;
            free(checked_set);
            checked_set = global_checked_set;
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
    total_mpi_bcast_time += MPI_Wtime();

    if(world_rank == 0){
        cout << "Min: " << global_min << endl << "Time: " << total_mpi_bcast_time << "s" << endl;
    }

    // final cleanup
    if(world_rank == 0) delete[] sophsticated_table;
    else model->free_posterior_probability_map();
    free(checked_set);

    // Finalize the MPI environment.
    MPI_Finalize();
}