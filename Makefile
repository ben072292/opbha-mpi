OPENMP_NAME = opbha-openmp
MPI_NAME = opbha-mpi

all: $(OPENMP_NAME) $(MPI_NAME)

$(OPENMP_NAME): main.cpp lattice_model.cpp pool_stat.cpp
	g++-11 -O2 -fopenmp main.cpp lattice_model.cpp pool_stat.cpp -o $(OPENMP_NAME)

$(MPI_NAME): main_mpi.cpp lattice_model.cpp pool_stat.cpp
	g++-11 -I/opt/homebrew/Cellar/open-mpi/4.1.3/include -L/opt/homebrew/Cellar/open-mpi/4.1.3/lib -L/opt/homebrew/opt/libevent/lib -lmpi -fopenmp main_mpi.cpp lattice_model.cpp pool_stat.cpp -o $(MPI_NAME)

clean: 
	rm $(OPENMP_NAME) $(MPI_NAME)

runopenmp:
	@$(MAKE) && ./$(OPENMP_NAME) 16 0.2

runmpi:
	@$(MAKE) && mpirun -n 8 ./$(MPI_NAME) 16 0.2

# 16 0.2, first is pool_size, second is prior probability
