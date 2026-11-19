# Entrega do projeto SLM: o que falta e como usar o modelo

Atualizado em 18/11/2026, às 23h30. **Prazo: 19/11/2026.** Este arquivo complementa o `PLANO.md` (decisões e histórico) e o `DADOS.md` (onde está cada arquivo).

---

## 1. Onde estamos

| Fase | Situação | Onde está |
|---|---|---|
| 0–3. Ambiente, código, dados e piloto | ✅ Concluídas | `PLANO.md` |
| 4. Pré-treino (114M parâmetros, 2,5B tokens, 2 GPUs) | ✅ 15–16/11, 5h44, perplexidade final 18,8 | Hugging Face (`pretrain/`), servidor e PC local |
| 5. Mid-training M (chat + 25% de matemática) | ✅ 16/11, RTX 4060, com acúmulo de gradiente | Hugging Face (`mid_m/`) e PC local |
| 5. Mid-training C (só chat, controle) | 🟡 **Treinado** (16/11, servidor, 1h30); **avaliação final pendente** | Servidor (`checkpoints/mid_c/`) |
| 6. SFT (a partir de M) | ✅ 16/11, RTX 4060 | Hugging Face (`sft/`) e PC local |
| 7. Avaliação | 🟡 Pré-treino, M e SFT avaliados; C pendente | Hugging Face (`results/`) |
| 8. Entrega | 🟡 Ver seção 3 | — |

**Hugging Face** (público): https://huggingface.co/Xererr/slm-114m. Em 18/11, ele tem `final.pt` e `metrics.jsonl` de `pretrain`, `mid_m` e `sft`, mais `results/{pretrain,mid_m,sft}.json`. **Faltam:** tudo do C e a página do modelo (`README.md`).

**Código:** https://github.com/FHaggs/slm (`main`). Os PRs #1, #2 e #3 foram aceitos (o #3 em 18/11). As mudanças de 18/11 (este arquivo, `README.md` e `hf/README.md`) estão na `main` de https://github.com/Xererr/slm-scherer e entram no PR #4 (seção 3).

### Resultados (mesma máquina, RTX 4060)

| Modelo | Perplexidade (FineWeb) | HellaSwag | ARC-Easy | Aritmética | GSM8K | Respondeu com número |
|---|---|---|---|---|---|---|
| Pré-treino | 18,77 | 28,1% | 50,8% | 0,7% | 1,9% | 86,4% |
| M | 61,07 | 27,1% | 37,2% | **3,2%** | **4,1%** | 99,4% |
| SFT | 62,65 | 27,4% | 37,3% | 3,1% | 2,8% | 99,5% |
| C | *treinado; avaliação final pendente (seção 4)* | | | | | |

- **M contra o pré-treino (McNemar):** aritmética p = 7,4e-6 (de 8 para 35 acertos); GSM8K p = 0,0009. Entre as perguntas respondidas com número, o acerto do GSM8K sobe de 2,2% para 4,1%: **não é só formato**, diferente do piloto.
- **Regressões:** HellaSwag −1,0 ponto (p = 0,0007) e ARC-Easy −13,6 pontos (p = 2e-34).
- **SFT contra M:** praticamente igual (GSM8K −1,3 ponto, p = 0,0498, no limite).

### O que já se sabe do C sem a avaliação final
- **Curva do treino (servidor):** perplexidade no chat de 22,8 para 4,86. Aritmética de 0,81% para 1,17%, oscilando entre 0,99% e 2,44%, **sem tendência**, como se espera de um controle.
- **Piloto (modelo de 32M):** M acima de C na aritmética (p = 0,006). A queda do ARC-Easy apareceu em M **e** em C, o que aponta para o chat, e não para a matemática.
- ⚠️ Os números da curva vêm da medição **durante o treino, no servidor**. Eles não entram na tabela acima, que é da avaliação final na RTX 4060. A comparação justa com M é curva contra curva (`mid_m/metrics.jsonl` contra `mid_c/metrics.jsonl`).

---

## 2. Decisões de 18/11

- **Entrega com o que está no Hugging Face:** M e SFT locais (RTX 4060) são os oficiais.
- **O C fica em seção separada**, como último item pendente: treinado, com a curva do treino e a evidência do piloto. Se a avaliação final (seção 4) rodar antes da apresentação, os números entram na tabela.

---

## 3. O que falta (prioridade para 19/11)

| # | Tarefa | Como |
|---|---|---|
| 1 | **Aceitar o PR #4** (mudanças de 18/11) | Abrir em https://github.com/FHaggs/slm/compare/main...Xererr:slm-scherer:main → *Create pull request* → aceitar |
| 2 | **Slides** (10–15 min) | Roteiro na seção 6 |
| 3 | **Vídeo de reserva da demo** | Seção 5.2, gravando a tela (no Windows 11, a Ferramenta de Captura grava vídeo) |
| 4 | **Exemplos de geração** (exigidos na entrega) | Demo com SFT e M, mesmas perguntas em inglês; salvar pergunta e respostas num `exemplos.md` |
| 5 | **Gráficos** das curvas | Seção 3.1 |
| 6 | **`CREDITS.md`** (o enunciado exige crédito a código de terceiros adaptado) | Listar o que inspirou código (Referências do `PLANO.md`, por exemplo nanochat e SmolLM3) e o uso de IA |
| 7 | **Página do modelo e C no Hugging Face** | Seção 4 |
| 8 | **Avaliar o C** (opcional, ~30 min) | Seção 4 |
| 9 | **Job 1175 no servidor** | Ver abaixo |
| 10 | **Ensaio cronometrado** | Todo o grupo precisa saber explicar o código (seção 7) |

### 3.1. Gráficos
As curvas estão no Hugging Face: `pretrain/metrics.jsonl`, `mid_m/metrics.jsonl` e `sft/metrics.jsonl`. A do C está no servidor (`checkpoints/mid_c/metrics.jsonl`) até ser enviada. Cada linha é um JSON com `step` e `tokens`. As linhas de treino têm `loss`, `lr`, `grad_norm` e `tok_per_s`. As de validação têm `val_loss` e `val_ppl` e, no mid-training, `arith_acc` e uma coluna por célula da aritmética.

Sugestões: (a) loss de treino e de validação do pré-treino, com o decaimento do WSD (últimos 10%) marcado; (b) `arith_acc` de M e de C ao longo do mid-training, no mesmo gráfico; (c) `val_loss` do SFT, para mostrar que não houve overfitting.

### 3.2. Job 1175 (servidor)
Ele era uma cópia de M e SFT no servidor, com início previsto para 18/11 às 11h58. Situação: **não verificada**.
```bash
sacct -j 1175 -X --format=JobID,State,Start,End,Elapsed
```
- **Se ainda estiver na fila:** `scancel 1175`. Isso libera a GPU para outras turmas e não afeta nada do que foi entregue.
- **Se já rodou:** ele gravou M e SFT de servidor em `checkpoints/mid_m` e `checkpoints/sft`, no diretório compartilhado. Esses **não** são os oficiais (que estão no Hugging Face). Não misturar.

---

## 4. Avaliar e publicar o C (no PC local, com a RTX 4060)

Usa a mesma máquina e a mesma estrutura de pastas de M e SFT (`DADOS.md`, seção "Cópia local"). O treino do C não é refeito: só a avaliação (~25 min).

**No servidor**, gerar a impressão digital do arquivo:
```bash
sha256sum <diretorio-compartilhado>/checkpoints/mid_c/final.pt
```

**No PC local** (PowerShell, na pasta que tem `shared` e `repo`):
```powershell
New-Item -ItemType Directory -Force shared\checkpoints\mid_c
scp -J <usuario-entrada>@<servidor-entrada> <usuario-cluster>@<servidor-cluster>:<diretorio-compartilhado>/checkpoints/mid_c/final.pt shared\checkpoints\mid_c\
scp -J <usuario-entrada>@<servidor-entrada> <usuario-cluster>@<servidor-cluster>:<diretorio-compartilhado>/checkpoints/mid_c/metrics.jsonl shared\checkpoints\mid_c\
Get-FileHash shared\checkpoints\mid_c\final.pt -Algorithm SHA256
```
O código tem de ser igual ao do `sha256sum`. O Windows mostra em maiúsculas; tirando isso, os dois são idênticos. Se forem diferentes, apague o arquivo e baixe de novo.

```powershell
cd repo
git checkout main
git pull
uv run python scripts/run_local.py
```
Ele pula o que já existe (M, SFT e as avaliações deles), avalia só o C e, no fim, roda a comparação **`mid_c x mid_m`**.

**Publicar no Hugging Face** (ainda na pasta `repo`):
```powershell
uv run hf upload Xererr/slm-114m ..\shared\checkpoints\mid_c\final.pt mid_c/final.pt
uv run hf upload Xererr/slm-114m ..\shared\checkpoints\mid_c\metrics.jsonl mid_c/metrics.jsonl
uv run hf upload Xererr/slm-114m results\final results
uv run hf upload Xererr/slm-114m hf\README.md README.md
```
O último comando cria a página do modelo. Ele pode ser rodado mesmo sem avaliar o C, depois do `git pull`.

**Com o resultado:** pôr a linha do C na tabela da seção 1 (e no `README.md` e no `hf/README.md`) e aplicar o critério da Seção 6 do `PLANO.md`: M acima de C em aritmética.

---

## 5. Tutorial: como usar o modelo

### 5.1. Instalar (uma vez)
Requisitos: `git` e o [`uv`](https://docs.astral.sh/uv/). Testado no Linux do servidor e no Windows 11; **não testado no macOS**.
```bash
git clone https://github.com/FHaggs/slm.git
cd slm
uv sync
uv run hf download Xererr/slm-114m sft/final.pt mid_m/final.pt --local-dir modelos
```
O `uv sync` instala o Python 3.12 e o PyTorch 2.11 com CUDA 12.8 (~3 GB). Sem GPU NVIDIA, roda em CPU, mais devagar.

### 5.2. Conversar com o modelo (demo)
PowerShell:
```powershell
$env:SLM_CHECKPOINT = "modelos\sft\final.pt"
uv run streamlit run app/chat.py
```
Linux ou macOS:
```bash
SLM_CHECKPOINT=modelos/sft/final.pt uv run streamlit run app/chat.py
```
O navegador abre em `http://localhost:8501`. Na barra lateral:

| Controle | O que faz | Sugestão |
|---|---|---|
| Checkpoint | Caminho do modelo. Troque por `modelos/mid_m/final.pt` para comparar | SFT |
| Temperatura | 0 = sempre a palavra mais provável (repetitivo); perto de 1,5 = muito aleatório | 0,7 |
| Top-k | Só sorteia entre as k palavras mais prováveis | 40 |
| Máximo de tokens | Tamanho máximo da resposta (o modelo também para sozinho no `<\|end\|>`) | 200 |
| Limpar conversa | Começa uma conversa nova | — |

**O que esperar:** responde em formato de conversa e para sozinho, mas é um modelo pequeno, que erra fatos e contas com frequência. Escreva em **inglês**. Conversas longas passam do contexto de 1.024 tokens; nesse caso, use "Limpar conversa".

### 5.3. Usar no Python
Da raiz do repositório, com `uv run python seu_script.py`. O trecho foi testado em CPU com um checkpoint mínimo no mesmo formato:
```python
import torch
from slm.checkpoint import load_model
from slm.chat import render_prompt
from slm.generate import stream
from slm.tokenizer import END, ENDOFTEXT

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model("modelos/sft/final.pt", device)
print(f"{model.num_params() / 1e6:.1f}M parametros em {device}")

messages = [{"role": "user", "content": "What is 7 + 5?"}]
prompt = render_prompt(messages, tokenizer)  # termina em <|assistant|>
tokens = list(stream(model, prompt, max_new_tokens=200, temperature=0.7, top_k=40,
                     stop_tokens=frozenset({END, ENDOFTEXT})))
print(tokenizer.decode(tokens))
```
Para continuar a conversa, acrescente `{"role": "assistant", "content": <resposta>}` e a próxima pergunta em `messages`.

### 5.4. Avaliar e comparar
```bash
uv run python scripts/evaluate.py --checkpoint modelos/sft/final.pt --output results/sft.json --benchmarks hellaswag arc_easy
uv run python scripts/compare.py results/mid_m.json results/sft.json
```
- A perplexidade só é calculada com `--val-dir <dados>/pretrain/val`. Esses dados não estão no Hugging Face.
- Em GPU de 8 GB, acrescente `--ppl-batch-size 4 --gsm-batch-size 8`. Para um teste rápido: `--mc-limit 200 --gsm-limit 100`.
- O `compare.py` faz o teste de McNemar item a item: `so A` e `so B` contam os itens que só um dos dois acertou. p < 0,05 indica diferença real.

---

## 6. Roteiro da apresentação (10–15 min)

| # | Slide | Conteúdo | Tempo |
|---|---|---|---|
| 1 | Objetivo e pipeline | Pré-treino → mid-training → SFT. Hipótese: treinar com matemática melhora a matemática | 1 min |
| 2 | Arquitetura | 114M parâmetros; GQA, RoPE, SwiGLU, RMSNorm; contexto de 1.024 | 1,5 min |
| 3 | Dados e Chinchilla | 2,5B tokens (~22 × N); FineWeb-Edu, SmolTalk, matemática | 1,5 min |
| 4 | Infraestrutura | DDP em 2 GPUs (122 mil tok/s, 89% do ideal); fila compartilhada; acúmulo de gradiente para M e SFT numa RTX 4060, validado por testes | 2 min |
| 5 | Curvas | Pré-treino (perplexidade 18,8; decaimento do WSD); aritmética de M e de C ao longo do mid | 1,5 min |
| 6 | Resultados | Tabela de pré-treino, M e SFT, com McNemar | 2 min |
| 7 | Discussão | Ganho real em aritmética e GSM8K; queda no HellaSwag e no ARC-Easy | 1,5 min |
| 8 | **Grupo de controle (C): trabalho pendente** | Por que existe; treinado, avaliação final pendente; curva estável (~1–2%); evidência do piloto (M > C, p = 0,006; ARC caiu em M e C) | 1 min |
| 9 | Demo ao vivo | SFT e M nas mesmas perguntas (vídeo de reserva) | 2 min |
| 10 | Conclusões e limitações | Modelo pequeno; uma execução por configuração; GPUs diferentes no mid; C sem avaliação final | 0,5 min |

Um resultado negativo, bem explicado, também conta (`PLANO.md`, Seção 6).

---

## 7. Perguntas prováveis da banca

| Pergunta | Resposta curta |
|---|---|
| Por que 114M parâmetros e 2,5B tokens? | Chinchilla: ~20 tokens por parâmetro. 2,5B / 114M ≈ 22 |
| O que é o WSD? | A LR sobe no aquecimento (*warmup*), fica estável e cai até zero nos 10% finais. O efeito aparece na curva de loss do pré-treino |
| Para que serve o C? | Separar o efeito da matemática do efeito do formato de chat |
| Por que o C não está na tabela? | Foi treinado, mas a avaliação final na mesma máquina dos outros ficou pendente. A curva do treino e o piloto dão a evidência disponível |
| O que é o teste de McNemar? | Compara dois modelos nos **mesmos** itens, contando só os que um acertou e o outro errou. Adequado para dados pareados |
| Como M e SFT rodaram numa GPU de 8 GB? | Acúmulo de gradiente: 8 micro-lotes de 3 somados num passo de 24, com a loss ponderada pelos tokens válidos. Testes mostram que é equivalente a um lote único de 24 |
| Por que o ARC-Easy caiu? | O mid-training em chat muda o formato que o modelo espera. No piloto, a queda apareceu em M e em C |
| O SFT melhorou alguma coisa? | Nos benchmarks, quase nada. O papel dele é o formato: responder como assistente e parar no lugar certo, o que se vê na demo |
| Os tokens do mid estão na faixa de 100M–300M? | Úteis, sim: ~294M (M) e ~292M (C). Com o preenchimento (*padding*), 374M e 360M |
| Como o pré-treino retoma sem repetir dados? | O checkpoint guarda a posição nos dados; a ordem é uma permutação fixa por semente; a LR depende dos tokens vistos |
