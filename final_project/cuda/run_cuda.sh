make
srun -n 1 --gres=gpu:1 ./dct_cuda ../src/Barbara.jpg ../out/Barbara.png ../out/Barbara_cuda.png