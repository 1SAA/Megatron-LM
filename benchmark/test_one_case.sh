set -x
source /opt/lcsoftware/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.3.0/miniconda3-4.10.3-u6p3tgreee7aigtnvuhr44yqo7vcg6r6/etc/profile.d/conda.sh
conda activate megatron

T_MODEL=${T_MODEL:-"gpt2-4b"}

N_GPU=${N_GPU:-2}
N_BS=${N_BS:-4}

mkdir -p benchmark_logs

env CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=${N_GPU} --master_port=29911 ./run_gpt.py ${T_MODEL} ${N_BS} \
2>&1 | tee ./benchmark_logs/${T_MODEL}_bs_${N_BS}_gpu_${N_GPU}_megatron.log
