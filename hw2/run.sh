make clean
make
srun -n1 -c8 ./hw2seq out.png 2602 -3 0.2 -3 0.2 979 2355
srun -n1 -c8 ./hw2a tmp.png 2602 -3 0.2 -3 0.2 979 2355
diff out.png /home/pp23/share/hw2/testcases/fast01.png
diff tmp.png /home/pp23/share/hw2/testcases/fast01.png