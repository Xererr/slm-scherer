#!/bin/bash
# Fases 5 e 6 num job so: avaliacao do pre-treino -> mid-training M e C -> SFT (a partir de M)
# -> avaliacao de M, C e SFT -> McNemar. 1 GPU.
# Le e grava em $SLM_SHARED (checkpoints do grupo); resultados em results/final/ do clone.
# Cada etapa pula se ja terminou (final.pt ou JSON): resubmeter retoma de onde parou.
# Pre-requisito: checkpoints/pretrain/final.pt (Fase 4).
#SBATCH -J slm-mid-sft
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
umask 002  # o grupo precisa escrever e retomar
source .venv/bin/activate
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/$USER/slm}"        # caches por usuario
export SLM_SHARED="${SLM_SHARED:-/scratch/ap211/slm-shared}"   # dados e checkpoints do grupo
export HF_HOME="$SLM_SCRATCH/hf-cache"                         # benchmarks fora do /home (cota de 10 GB)
export TRITON_CACHE_DIR="$SLM_SCRATCH/triton-cache"
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_SHAPE_PADDING=0                           # sem isso o torch.compile estoura a memoria
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ck="$SLM_SHARED/checkpoints"
results=results/final
mkdir -p "$results"

if [ ! -f "$ck/pretrain/final.pt" ]; then
    echo "falta $ck/pretrain/final.pt: o pre-treino (Fase 4) ainda nao terminou. Abortando"
    exit 1
fi

# Um job por vez: dois jobs gravando os mesmos checkpoints se corromperiam.
exec 9>"$ck/.lock_mid_sft"
if ! flock -n 9; then
    echo "outro job das Fases 5 e 6 ja esta rodando: abortando"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

evaluate() {  # evaluate <etapa> <checkpoint>
    if [ -f "$results/$1.json" ]; then
        echo "=== avaliacao: $1 ja existe, pulando"
        return
    fi
    echo "=== avaliacao: $1"
    python scripts/evaluate.py --checkpoint "$2" --output "$results/$1.json" \
        --val-dir "$SLM_SHARED/data/pretrain/val" --benchmarks hellaswag arc_easy
}

# Fechamento da Fase 4: o pre-treino e medido aqui, sem gastar outra entrada na fila.
evaluate pretrain "$ck/pretrain/final.pt"

for run in mid_m mid_c; do
    echo "=== mid-training: $run"
    python scripts/midtrain.py --config "configs/final/$run.yaml"
done

echo "=== SFT (a partir de M)"
python scripts/sft.py --config configs/final/sft.yaml

evaluate mid_m "$ck/mid_m/final.pt"
evaluate mid_c "$ck/mid_c/final.pt"
evaluate sft "$ck/sft/final.pt"

echo "=== M contra o pre-treino"
python scripts/compare.py "$results/pretrain.json" "$results/mid_m.json"
echo "=== M contra C"
python scripts/compare.py "$results/mid_c.json" "$results/mid_m.json"
echo "=== SFT contra M"
python scripts/compare.py "$results/mid_m.json" "$results/sft.json"
