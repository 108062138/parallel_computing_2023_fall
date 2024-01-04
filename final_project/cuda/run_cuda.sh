make
srun -n 1 --gres=gpu:1 ./dct_cuda ../src/Mona_Lisa_1024.jpg ../out/Barbara.png ../out/Barbara_cuda.png