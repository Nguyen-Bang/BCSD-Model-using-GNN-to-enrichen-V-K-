# GNN-enriched binary code similarity

This repository implements the pipeline used in a bachelor's thesis on binary code similarity detection. It combines assembly transformers with control-flow graph information.

The code covers CFG extraction, static enrichment, tokenization, compact dataset building, training, fixed-pair ranking, and prefix-content controls. Datasets and checkpoints are not included.

## Model

The three-layer baseline is a RoFormer with hidden size 768. It uses the CLAP tokenizer and initializes its word embeddings from CLAP. Its transformer layers are trained from scratch.

A second backbone uses the full 12-layer pretrained CLAP RoFormer and freezes it. Both backbones use masked mean pooling, 132 static function features, and a normalized 768-dimensional output.

The graph model adds a three-layer GATv2 encoder. CFG nodes have 20 static features. Typed forward, reverse, and self-loop edges are supported.

Attention pooling creates ten 256-dimensional graph vectors. Each transformer layer projects them to key/value prefixes and learns one gate for each attention head and prefix token.

The set encoder replaces GATv2 with a per-node MLP and the same attention pooling. PalmTree configurations use 128-dimensional PalmTree node embeddings.

## Requirements and hardware

Use Python 3.13. The pinned environment was tested with PyTorch 2.11, PyTorch Geometric 2.8, and CUDA 12.8.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

On a machine without CUDA, skip the separate PyTorch command and install `requirements.txt` directly. CPU-only use is suitable for preprocessing and tests, not the supplied training runs.

| Stage | Requirement |
|---|---|
| CFG extraction | Licensed IDA Pro 9.3 with IDAPython and Python 3.13 |
| Enrichment and dataset building | CPU; no GPU required |
| Training | One NVIDIA GPU with BF16 support and the CUDA 12.8 environment |
| Evaluation | The same CUDA environment used for training |

The supplied training settings used about 11 GB of GPU memory. Dataset preload is usually the larger constraint.

A compact 23 GB dataset used about 43 GB of host RAM. Use at least 64 GB RAM; 96 GB gives useful headroom. BinaryCorp-3M requires storage and memory sized for that external corpus.

## Prepare a dataset

The input must contain at least two builds of the same function. Samples are paired by `(project, function_name)`, so function names must remain stable across compilation variants.

Place binaries directly below project directories:

```text
data/binaries/
  project_a/
    build_1
    build_2
  project_b/
    build_1
    build_2
```

### 1. Extract CFGs with IDA

Run this on the IDA machine. `--ida-path` may be an executable name or an absolute path.

```bash
python -m preprocessing.I_ida_extraction.batch_extract \
  data/binaries data/I_cfg \
  --ida-path /path/to/ida64
```

Each IDA process uses an isolated temporary database. The output mirrors the input project layout and contains one JSON file per binary.

### 2. Add static features

```bash
python -m preprocessing.II_static_enrichment.batch_enrich \
  data/I_cfg data/II_enriched
```

This adds 20 node features and 132 function features. The function vector has four scalar CFG properties and a 128-value opcode MinHash.

### 3. Tokenize assembly

```bash
python -m preprocessing.III_tokenization.batch_tokenize \
  data/II_enriched data/III_tokenized \
  --max-seq-length 1024
```

The tokenizer keeps at most 20 subword tokens per instruction and 1024 tokens per function.

### 4. Define project splits

Edit `configs/data_splits.yaml`, or create a YAML or JSON file with the same shape:

```yaml
train: [project_a, project_b]
validation: [project_c]
test: [project_d, project_e]
```

Projects must occur in exactly one split. Unknown project directories are skipped with a warning.

### 5. Build the compact dataset

```bash
python -m scripts.baseline.reduce_tokenized \
  --input_dir data/III_tokenized \
  --output_dir reduced_data \
  --split-manifest configs/data_splits.yaml \
  --workers 8
```

The result is stored as `reduced_data/{train,validation,test}/<project>/*.pt`.

### Optional: add PalmTree node embeddings

PalmTree is external. Clone its official repository and obtain its published checkpoint and vocabulary before running:

```bash
python -m preprocessing.III_tokenization.palmtree.augment_with_node_embeddings \
  --base-dir reduced_data \
  --tokenized-dir data/III_tokenized \
  --output-dir reduced_data_palmtree_full \
  --palmtree-checkout /path/to/PalmTree
```

The command writes inline float16 `[number_of_nodes, 128]` features to a separate dataset. It validates each function and node before replacing an output file and records the PalmTree checkout revision, model hash, and vocabulary hash.

## Preloading

Keep these settings for the supplied training path:

```yaml
data:
  data_dir: reduced_data
  preload: true
  num_workers: 0
```

Preloading keeps stored tensors compact and converts only the selected batch. Use the canonical disk copy, enough host RAM, and no swap.

Do not convert the dataset to Python lists, copy it into a memory-backed `/tmp`, or enable multiple loader workers. Multi-GPU training would preload one full dataset per process and is unsupported.

## Reproduce the experiments

All commands run from the repository root. The first model download needs network access. Use the same checkpoint selection rule for every model in a comparison.

### Three-layer baseline and typed GNN

Train the baseline once:

```bash
python -m scripts.baseline.train \
  --config configs/baseline_config.yaml \
  --device cuda:0 \
  --seed 0 \
  --pair-seed 0
```

Its checkpoint is `checkpoints/baseline_with_features/seed_0/best_model.pt`. Train and evaluate the typed GNN with three seeds:

```bash
for seed in 101 202 303; do
  python -m scripts.gnn.train \
    --config configs/gnn_v8_typed_edges.yaml \
    --backbone-checkpoint checkpoints/baseline_with_features/seed_0/best_model.pt \
    --device cuda:0 --seed "$seed" --pair-seed 0

  python -m scripts.gnn.evaluate \
    --config configs/gnn_v8_typed_edges.yaml \
    --checkpoint "checkpoints/gnn_v8_typed_edges/seed_${seed}/best_mrr.pt" \
    --device cuda:0 --output-dir "artifacts/typed_seed${seed}"
done
```

Seeds 101, 202, and 303 are stored in `seed_101`, `seed_202`, and `seed_303` below the configured checkpoint directory. Fresh runs refuse to overwrite existing training artifacts.

For three-seed tables, report the arithmetic mean and sample standard deviation (`ddof=1`) of each metric from the three evaluation JSON files.

Use `--resume <checkpoint.pt>` to restore optimizer, scheduler, scaler, structural heads, random state, and early stopping. The resume checkpoint fixes the output directory.

Other three-layer graph configurations are:

| Configuration | Purpose |
|---|---|
| `gnn_v8_untyped_edges.yaml` | GATv2 without edge types |
| `gnn_v8_setencoder.yaml` | Node MLP and attention pooling; no message passing |
| `gnn_v8_noedges.yaml` | Node features with self-loops only |
| `gnn_v8_noedges_noattn_dropout.yaml` | Self-loop control without attention dropout |
| `gnn_v8_noprefix.yaml` | Transformer without a structural prefix |
| `gnn_v8_register.yaml` | Learned content-free prefix |
| `gnn_v8_randomnodes.yaml` | CFG with deterministic random node content |
| `palmtree_setencoder.yaml` | Set encoder over PalmTree node embeddings |

`best_model.pt` is selected by validation loss. `best_mrr.pt` is selected by validation MRR. The fixed ranking evaluator writes `fixed_pair_ranking.json`.

### Prefix-content controls

`noise_perdim` is the thesis "Per-dim current-batch per-position/dimension matched-noise" control. It uses test batch 33, pair seed 0, and sampling seeds 0 through 4.

```bash
python -m scripts.gnn.evaluate_controls \
  --config configs/gnn_v8_typed_edges.yaml \
  --checkpoint checkpoints/gnn_v8_typed_edges/seed_101/best_mrr.pt \
  --batch-size 33 --pair-seed 0 \
  --sampling-seeds 0,1,2,3,4 \
  --modes noise_global,noise_perdim,shuffle \
  --device cuda:0 --output-dir artifacts/typed_seed101_controls
```

The output is `content_control_hardened.json`. It compares the full prefix, a disabled prefix, global matched noise, `noise_perdim`, and shuffled content on fixed compilation pairs.

### PalmTree typed GNN

After building `reduced_data_palmtree_full`, run the typed PalmTree configuration:

```bash
for seed in 101 202 303; do
  python -m scripts.gnn.train \
    --config configs/palmtree_typed_edges.yaml \
    --backbone-checkpoint checkpoints/baseline_with_features/seed_0/best_model.pt \
    --device cuda:0 --seed "$seed" --pair-seed 0

  python -m scripts.gnn.evaluate \
    --config configs/palmtree_typed_edges.yaml \
    --checkpoint "checkpoints/palmtree_typed_edges/seed_${seed}/best_mrr.pt" \
    --device cuda:0 --output-dir "artifacts/palmtree_typed_seed${seed}"
done
```

### Frozen CLAP baseline, register, and typed GNN

This helper trains and evaluates one frozen-CLAP configuration for all three seeds:

```bash
run_clap () {
  cfg="$1"
  run="$2"
  mkdir -p "artifacts/$run"
  for seed in 101 202 303; do
    python -m scripts.gnn.clap_frozen.train \
      --config "$cfg" --device cuda:0 --seed "$seed" --pair-seed 0
    python -m scripts.gnn.clap_frozen.evaluate \
      --config "$cfg" \
      --checkpoint "checkpoints/${run}/seed_${seed}/best_mrr.pt" \
      --split test --pair-seed 0 --device cuda:0 \
      --output "artifacts/${run}/seed_${seed}.json"
  done
}

run_clap configs/clap_frozen_baseline_config.yaml clap_frozen_baseline
run_clap configs/clap_frozen_gnn_register.yaml clap_frozen_register
run_clap configs/clap_frozen_gnn_config.yaml clap_frozen_gnn
```

These runs also use separate `seed_<seed>` directories and refuse to replace existing checkpoints unless `--resume` is used.

### Frozen CLAP diagnostics

The frozen CLAP GNN has 12 layers, 12 heads, and 10 prefix tokens. This command evaluates all 1,440 gate cells and supports `--resume`:

```bash
python -m scripts.gnn.ablate_gate_cells \
  --config configs/clap_frozen_gnn_config.yaml \
  --checkpoint checkpoints/clap_frozen_gnn/seed_101/best_mrr.pt \
  --backbone clap --device cuda:0 \
  --output-dir artifacts/clap_frozen_gnn/gate_cells_seed101
```

Measure cosine similarity for matched and deranged GNN prefixes with:

```bash
python -m scripts.gnn.prefix_cosine \
  --config configs/clap_frozen_gnn_config.yaml \
  --checkpoint checkpoints/clap_frozen_gnn/seed_101/best_mrr.pt \
  --backbone clap --device cuda:0 \
  --output-dir artifacts/clap_frozen_gnn/prefix_cosine_seed101
```

### BinaryCorp-3M

BinaryCorp-3M, its CSV manifests, extracted CFGs, and model checkpoints are not included. Obtain the corpus under its own terms and extract its binaries with IDA before running these steps.

Build compact files from extracted CFG JSON and the official manifests:

```bash
python -m preprocessing.binarycorp.build_binarycorp_dataset \
  --cfg-dir data/binarycorp_cfg \
  --train-csv /path/to/small_train.csv \
  --test-csv /path/to/small_test.csv \
  --output-dir reduced_data_binarycorp_raw \
  --workers 8 --recursive
```

Create a project-disjoint validation split and verify that each test pair has full pools:

```bash
python -m preprocessing.binarycorp.build_bc3m_split \
  --source-dir reduced_data_binarycorp_raw \
  --small-train-csv /path/to/small_train.csv \
  --small-test-csv /path/to/small_test.csv \
  --output-dir reduced_data_bc3m \
  --seed 0 --mode copy --check-test-pools --pool-size 10000
```

Train the projection-fusion, set-encoder, and typed-GNN configurations:

```bash
for name in projection_fusion set_encoder typed_gnn; do
  python -m scripts.gnn.clap_frozen.train \
    --config "configs/binarycorp_${name}.yaml" \
    --device cuda:0 --seed 0 --pair-seed 0
done
```

Evaluate zero-shot CLAP, the three BinaryCorp-trained modes, and the internal typed-GNN checkpoint without BinaryCorp retraining:

```bash
python -m scripts.binarycorp.evaluate \
  --model clap-zero-shot --data-dir reduced_data_bc3m \
  --pool-size 10000 --seed 0 --device cuda:0 \
  --output artifacts/binarycorp_clap_zero_shot.json

python -m scripts.binarycorp.evaluate \
  --model projection-fusion \
  --checkpoint checkpoints/binarycorp_projection_fusion/seed_0/best_mrr.pt \
  --model-config configs/binarycorp_projection_fusion.yaml \
  --data-dir reduced_data_bc3m --pool-size 10000 --seed 0 --device cuda:0 \
  --output artifacts/binarycorp_projection_fusion.json

python -m scripts.binarycorp.evaluate \
  --model set-encoder \
  --checkpoint checkpoints/binarycorp_set_encoder/seed_0/best_mrr.pt \
  --model-config configs/binarycorp_set_encoder.yaml \
  --data-dir reduced_data_bc3m --pool-size 10000 --seed 0 --device cuda:0 \
  --output artifacts/binarycorp_set_encoder.json

python -m scripts.binarycorp.evaluate \
  --model typed-gnn \
  --checkpoint checkpoints/binarycorp_typed_gnn/seed_0/best_mrr.pt \
  --model-config configs/binarycorp_typed_gnn.yaml \
  --data-dir reduced_data_bc3m --pool-size 10000 --seed 0 --device cuda:0 \
  --output artifacts/binarycorp_typed_gnn.json

python -m scripts.binarycorp.evaluate \
  --model typed-gnn \
  --checkpoint checkpoints/clap_frozen_gnn/seed_101/best_mrr.pt \
  --model-config configs/clap_frozen_gnn_config.yaml \
  --data-dir reduced_data_bc3m --pool-size 10000 --seed 0 --device cuda:0 \
  --output artifacts/binarycorp_internal_typed_gnn.json
```

The default protocol uses pool size 10,000 and seed 0. It equally averages O0-O3, O1-O3, O2-O3, O0-Os, O1-Os, and O2-Os with a pessimistic tie policy.

## Evaluation protocol

The fixed three-layer evaluator requires the test split, batch size 64, and pair seed 0. The controls require test batch 33 and pair seed 0. Both record pool and provenance data.

Never compare metrics from different project pools or pool sizes.

## Repository layout

```text
configs/        Training settings and split manifest
dataset/        Compact schema, dataset, sampler, and batch collation
evaluation/     Reusable evaluation runners and model adapters
models/         Transformers, graph/set encoders, and KV-prefix attention
preprocessing/  IDA extraction, enrichment, tokenization, and corpus builders
scripts/        Command-line entry points
training/       Shared training loops, losses, metrics, and checkpoints
tests/          Hermetic regression tests
```

`models/` contains reusable PyTorch modules. `scripts/` only turns those modules and the shared training or evaluation code into commands.

Run the tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

## External work

The full CLAP, PalmTree, and HermesSim implementations are not included.

- The tokenizer files in `preprocessing/clap_asm_tokenizer/` come from [hustcw/clap-asm](https://huggingface.co/hustcw/clap-asm) at revision `620f4beba2edce172e8f35e263399716494950c9`.
- The word-embedding initializer and frozen CLAP backbone use `model.safetensors` from that pinned revision. See [Hustcw/CLAP](https://github.com/Hustcw/CLAP) and its [paper](https://doi.org/10.1145/3650212.3652145).
- PalmTree is used only to generate optional node features. See [palmtreemodel/PalmTree](https://github.com/palmtreemodel/PalmTree).
- HermesSim informed the research, but its code and model files are not distributed here. See [NSSL-SJTU/HermesSim](https://github.com/NSSL-SJTU/HermesSim).
- CFG extraction requires [IDA Pro](https://hex-rays.com/ida-pro/). IDA and IDAPython are not distributed here.

### CLAP tokenizer license notice

```text
Copyright (c) 2024 Hustcw

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
