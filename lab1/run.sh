#!/bin/bash
#SBATCH -ptest
#SBATCH -N3
#SBATCH -n24
module load mpi
export IPM_REPORT=full
export IPM_REPORT_MEM=yes
export IPM_LOG=full
export LD_PRELOAD=/opt/ipm/lib/libipm.so
export IPM_HPM=“PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_REF_CYC,\
PAPI_SP_OPS,PAPI_DP_OPS,PAPI_VEC_SP,PAPI_VEC_DP”
mpirun ./lab1 5 100