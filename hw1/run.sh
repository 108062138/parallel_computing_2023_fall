# remove the directory that starts with hw1_
rm -rf hw1_*
rm *.xml
rm *.tar
rm *.out


module load mpi/latest

# module load ipm/latest
# export IPM_REPORT=full
# export IPM_REPORT_MEM=yes
# export IPM_LOG=full
# export LD_PRELOAD=/opt/ipm/lib/libipm.so
# export IPM_HPM="PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_REF_CYC,PAPI_SP_OPS,PAPI_DP_OPS,PAPI_DP_OPS,PAPI_VEC_SP,PAPI_VEC_DP"

make

# from n= 1 to 12. use srun
for i in {4..7}
do
    j=`expr $i \* 2`
    srun -pjudge -N$i -n$j ./hw1 536831999 /home/pp23/share/hw1/testcases/38.in 38.out
done

# srun -pjudge -N3 -n12 time ./hw1 536831999 /home/pp23/share/hw1/testcases/38.in 38.out

# read in the xml file starts with pp23s80 and convert it to html

find . -type f -name 'pp23s80*' -exec ipm_parse -html {} \;

# zip the file
find . -type d -name 'hw1_*' -exec tar -cvf {}.tar {} \;