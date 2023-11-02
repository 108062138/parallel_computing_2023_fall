make clean
make
srun -n1 -c4 ./hw2a out.png 22 -3 0.2 -3 0.2 4 23
srun -n1 -c4 ./hw2seq sample.png 22 -3 0.2 -3 0.2 4 23
diff out.png sample.png
# srun -n1 -c4 ./hw2a out.png 2602 -3 0.2 -3 0.2 979 2355
# diff out.png /home/pp23/share/hw2/testcases/fast01.png
# srun -n1 -c8 time ./hw2a out.png 7485 0.27483841838734274 0.4216774226409377 0.5755165572756626 0.5039244805312306 3840 2160
# diff out.png /home/pp23/share/hw2/testcases/slow07.png