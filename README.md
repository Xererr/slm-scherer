# SLM: modelo de linguagem pequeno treinado do zero

Trabalho de Aprendizado Profundo II (PUCRS, 2026). Um modelo de 114M parâmetros passa por pré-treino, mid-training e SFT. O experimento central testa se um mid-training com matemática melhora a matemática do modelo, usando um grupo de controle sem matemática.

- **Modelos (Hugging Face, público):** https://huggingface.co/Xererr/slm-114m
- **Decisões, cronograma e histórico:** `PLANO.md`
- **Dados (origem, contagens, onde está cada arquivo):** `DADOS.md`
- **O que falta, tutorial de uso e roteiro da apresentação:** `ENTREGA.md`

## Resultados

Avaliados na mesma máquina (RTX 4060). Perplexidade em 5,12M tokens de validação do FineWeb-Edu (dump que o treino não usa); HellaSwag e ARC-Easy por log-likelihood; aritmética em 1.107 contas; GSM8K com 3 exemplos no prompt.

| Modelo | Perplexidade | HellaSwag | ARC-Easy | Aritmética | GSM8K |
|---|---|---|---|---|---|
| Pré-treino | 18,77 | 28,1% | 50,8% | 0,7% | 1,9% |
| M (chat + 25% de matemática) | 61,07 | 27,1% | 37,2% | **3,2%** | **4,1%** |
| SFT (a partir de M) | 62,65 | 27,4% | 37,3% | 3,1% | 2,8% |
| C (só chat, controle) | *treinado; avaliação final pendente* | | | | |

- M contra o pré-treino (teste de McNemar): aritmética p = 7,4e-6; GSM8K p = 0,0009. HellaSwag (p = 0,0007) e ARC-Easy (p = 2e-34) caem.
- C: durante o treino, a aritmética ficou entre 0,99% e 2,44%, sem tendência. No piloto (32M), M superou C na aritmética (p = 0,006). Detalhes em `ENTREGA.md`, seção 1.

## Instalação

Requisitos: `git` e [`uv`](https://docs.astral.sh/uv/).
```bash
git clone https://github.com/FHaggs/slm.git
cd slm
uv sync
```
O `uv sync` instala o Python 3.12 e o PyTorch 2.11 com CUDA 12.8. Testado no Linux (cluster) e no Windows 11.

## Usar o modelo

```bash
uv run hf download Xererr/slm-114m sft/final.pt --local-dir modelos
SLM_CHECKPOINT=modelos/sft/final.pt uv run streamlit run app/chat.py
```
No PowerShell, troque a segunda linha por `$env:SLM_CHECKPOINT = "modelos\sft\final.pt"` seguido de `uv run streamlit run app/chat.py`. Uso no Python, controles da demo e avaliação: `ENTREGA.md`, seção 5.

## Reprodução por etapa

Os jobs do cluster usam Slurm e leem e gravam no diretório compartilhado do grupo, definido pela variável `SLM_SHARED`. Rodar a partir da raiz do repositório.

| Etapa | Comando | Onde rodou | Duração |
|---|---|---|---|
| Ambiente | `sbatch slurm/setup.sbatch` | Cluster (CPU) | ~5 min |
| Testes | `sbatch slurm/tests.sbatch` | Cluster (CPU) | ~1 min |
| Dados (pré-treino, M, C, SFT) | `sbatch slurm/data.sbatch` | Cluster (CPU) | 43 min |
| Piloto (32M) | `sbatch slurm/pilot.sbatch` | 1× RTX A5000 | 1h30 |
| Pré-treino (114M, 2,5B tokens) | `sbatch slurm/pretrain.sbatch` | 2× RTX A5000 (DDP) | 5h44 |
| Mid-training C | `sbatch slurm/mid_c.sbatch` | 1× RTX A5000 | 1h30 |
| Mid-training M, SFT e avaliações | `uv run python scripts/run_local.py` | 1× RTX 4060 (8 GB) | ~9h |

- **`run_local.py`:** espera os dados e o `final.pt` do pré-treino numa pasta `shared` ao lado do repositório, com a mesma estrutura do diretório compartilhado (`DADOS.md`, seção "Cópia local"). Rode antes `uv run python scripts/run_local.py --check`, que confere dados, disco e GPU e escolhe o micro-lote. M e SFT usam acúmulo de gradiente para manter o lote de 24 do servidor (`configs/local/`; testes em `tests/test_grad_accum.py`).
- **Avaliar e comparar:** `scripts/evaluate.py` e `scripts/compare.py` (`ENTREGA.md`, seção 5.4).
- **Configurações de cada etapa:** `configs/` (`smoke`, `pilot`, `final`, `local`).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `slm/` | Modelo, tokenizer, dados, laço de treino, geração, avaliação |
| `scripts/` | Preparação de dados, treino de cada etapa, avaliação, comparação, benchmark, execução local |
| `configs/` | Hiperparâmetros de cada etapa |
| `slurm/` | Jobs do cluster |
| `app/` | Demo em Streamlit |
| `tests/` | Testes (pytest) |
| `hf/` | Página do modelo no Hugging Face |

## Créditos

Referências e trabalhos que guiaram as decisões: seção "Referências" do `PLANO.md`. `CREDITS.md`, com o crédito específico a código de terceiros adaptado, está pendente (`ENTREGA.md`, seção 3). O código foi escrito com auxílio de IA e revisado pelo grupo.
