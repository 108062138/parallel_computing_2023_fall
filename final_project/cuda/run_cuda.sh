make
# srun -n 1 --gres=gpu:1 ./dct_cuda ../src/Barbara.jpg ../out/Barbara.png ../out/Barbara_cuda.png
srun -N1 -n1 -c2 --gres=gpu:2 ./dct_cuda_many_gpu ../src/Barbara.jpg         ../out/Barbara.png ../out/Barbara_cuda.png
srun -N1 -n1 -c2 --gres=gpu:2 ./dct_cuda_many_gpu ../src/Mona_Lisa_1024.jpg  ../out/ms.png      ../out/ms_dct_omp.png
srun -N1 -n1 -c2 --gres=gpu:2 ./dct_cuda_many_gpu ../src/Mona_Lisa_2048.jpg  ../out/ms.png      ../out/ms_dct_cuda.png