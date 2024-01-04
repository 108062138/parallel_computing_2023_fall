make
srun -N2 -n3 -c4 ./dct_omp_mpi ./src/Mona_Lisa_1024.jpg  ./out/ms.png ./out/ms_dct_omp.png # ./src/Mona_Lisa_1024.jpg  ./out/ms.png ./out/ms_dct_omp.png