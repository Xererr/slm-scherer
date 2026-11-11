#!/bin/bash
# Fase 4: pre-treino final (~114M params, 1 epoca do FineWeb-Edu = ~2,5B tokens).
# Padrao: 2 GPUs (DDP, exigencia do enunciado). Plano B em 1 GPU:
#   sbatch --gres=gpu:1 -c 16 --mem=48G slurm/pretrain.sbatch
# Le e grava em $SLM_SHARED: qualquer membro do grupo retoma do ultimo checkpoint ao resubmeter,
# inclusive com outro numero de GPUs.
#SBATCH -J slm-pretrain
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 1-12:00:00
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
umask 002  # o grupo precisa escrever e retomar
source .venv/bin/activate
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/$USER/slm}"        # caches por usuario
export SLM_SHARED="${SLM_SHARED:-/scratch/ap211/slm-shared}"   # dados e checkpoints do grupo
export TRITON_CACHE_DIR="$SLM_SCRATCH/triton-cache"            # fora do /home (cota de 10 GB)
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_SHAPE_PADDING=0                           # sem isso o torch.compile estoura a memoria
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_dir="$SLM_SHARED/checkpoints/pretrain"                     # o mesmo run_dir de configs/final/pretrain.yaml
mkdir -p "$run_dir"

# Um job por vez neste run_dir: dois jobs gravando os mesmos checkpoints se corromperiam.
exec 9>"$run_dir/.lock"
if ! flock -n 9; then
    echo "outro job ja esta usando $run_dir: abortando"
    exit 1
fi

ngpu=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$ngpu" -lt 1 ]; then
    echo "nenhuma GPU visivel: abortando"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
echo "=== pre-treino final em $ngpu GPU(s)"

# Corte por tempo meia hora antes do limite do Slurm (36h): salva checkpoint e sai; resubmeter retoma.
torchrun --standalone --nproc_per_node="$ngpu" scripts/pretrain.py \
    --config configs/final/pretrain.yaml --max-hours 35.5
