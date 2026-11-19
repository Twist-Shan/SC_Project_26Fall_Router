# SC_Project_26Fall_Router

A prompt-aware router learns which LLM response people tend to prefer for a given request. It fits a finite mixture of Bradley–Terry comparisons to pairwise votes. Prompt features determine the mixture weights, and each component estimates a separate vector of LLM skills. For a new prompt the router selects the candidate with the highest predicted average win probability against the other candidates.

EM assigns each vote a probability of belonging to each component. With those probabilities fixed, the skill update uses the weighted numbers of comparisons and wins for each component and unordered LLM pair. The implementation caches the pair indices and reuses them across EM iterations. It uses individual rows for the prompt-weight update and for skill fits that include response-specific covariates.

## Run a synthetic example

Use Python 3.12 in a fresh virtual environment. From the repository root run

```sh
python -m pip install -r requirements-reproduced.txt
python -m pip install --no-deps -e .
python examples/synthetic_router.py
```

The example generates training battles, fits the router and prints selected candidate IDs and their expected win probabilities on new prompts. The first column of the feature matrix is an intercept. The candidate set contains at least two distinct IDs seen during training.

## Experiments

The simulation experiments write tables and traces to `results/`. S1 varies the sample size and measures parameter recovery and routing regret. B1 compares optimisers on the same synthetic battles and initial parameters.

```sh
python experiments/sim_experiments.py S1
python experiments/em_benchmark.py B1
```

To study which comparisons are needed for routing, run `python experiments/coverage_study.py`. It trains on a connected path of 7 pairs or all 28 pairs among 8 candidates, at two levels of skill separation. Each setting uses 20 paired simulation replicates. Both the mixture and the global Bradley–Terry ranking are evaluated on the full candidate set. **Routing regret** is the difference in true mean win probability between the oracle choice and the fitted router's choice, averaged over new prompts. For example, under weak separation, broadening coverage from 7 pairs to all 28 reduced the mixture's mean regret from 0.0253 to 0.0085 at the same training budget of 2,400 votes. [Summary and Monte Carlo standard errors](results/coverage_study_summary.csv) · [Replicate results](results/coverage_study_runs.csv)

## Arena evaluation

The five-fold evaluation predicts the final recorded human preference in a decisive battle from its first prompt and the two candidate identities. It includes 39,716 battles across 36,235 distinct normalised first prompts. All battles with the same normalised prompt share an outer fold. Each fold learns its embedding and selects its hyperparameters using training prompts. The download script verifies the 184 MB input against the dataset publisher's SHA-256. [Arena dataset and revision](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/commit/e5a2158d14d30003ef7ec8d4756c86359715dbc2)

```sh
python data/download_data.py
python data/prepare_routing_input.py
```

### Reproduce the five-fold reference and pooled router

The pooled router gives equal weight to the EM mixture's pairwise win probability and a second estimate from the prompt's words, bigrams and surface features. The second branch fits candidate-specific effects and signed pair offsets using ridge-regularised logistic regression. It chooses the ridge penalty in each outer fold's inner validation split. At deployment `LexicalMixtureRouter.route` averages each candidate's two predicted mean win probabilities against the supplied opponents. `LexicalMixtureRouter` and `LexicalBT` are in `mixbt.lexical`.

| Method | Held-out pair accuracy | Log loss |
| --- | ---: | ---: |
| Global Bradley–Terry ranking | 64.732% | 0.619477 |
| EM mixture | 65.203% | 0.616697 |
| Lexical branch | 65.392% | 0.616656 |
| Equal pool, lexical branch and original EM mixture | 65.374% | 0.616032 |
| EM mixture with a frozen semantic representation (32 PCs) | 65.633% | 0.614977 |
| Family-shared semantic branch | 65.754% | 0.613361 |
| Equal pool, semantic branch and original EM mixture | **65.787%** | **0.612418** |

Relative to the original mixture, the lexical pool gained 0.171 percentage points in accuracy (paired 95% bootstrap interval 0.000 to 0.342) and reduced log loss by 0.000665 (95% interval 0.000310 to 0.001017). These intervals condition on the fitted out-of-fold predictions. [Lexical methods](results/lexical_refinement/metrics.csv) · [Paired intervals](results/lexical_refinement/contrasts.csv)

To reproduce the reported comparison, generate the reference predictions using the original five-fold source snapshot. Run these commands from the repository root after preparing the Arena data above. The 12 reference implementation files match the SHA-256 values recorded in the evaluation protocol. [Reference commit](https://github.com/Twist-Shan/SC_Project_26Fall_Router/tree/bfac1d6901f61783e263859ee6f2a4a8348d38c8)

```sh
git -c core.autocrlf=false worktree add --detach ../arena-router-reference bfac1d6901f61783e263859ee6f2a4a8348d38c8
python ../arena-router-reference/data/prepare_routing_input.py data/train.csv
python ../arena-router-reference/experiments/arena_nested.py --output ../arena-router-reference/results/nested --threads 2 --bootstrap 1000
python experiments/lexical_refinement.py --reference ../arena-router-reference/results/nested
python experiments/lexical_benchmark.py
```

The reference command writes `protocol.json` and `oof_predictions.npz` locally. The pooled evaluation checks battle order, exact first prompts and fold assignments, then reports paired differences with 1,000 prompt-group bootstrap draws. To evaluate the current mixture from scratch, run `python experiments/arena_nested.py --output results/nested --threads 2 --bootstrap 1000`. For a small smoke run, pass `--quick --rows 1500 --output results/smoke`. Run `python experiments/arena_descriptive.py --results-dir results/nested --threads 2` for descriptive summaries. Use a fresh output directory if the data, source or settings change.

### Semantic routing and the tie ablation

The semantic pool averages the original EM mixture with a second predictor that uses the prompt's words and a frozen multilingual sentence embedding. This branch estimates candidate-specific prompt effects and shares semantic effects within LLM families. Each training fold learns its own principal components and cluster centres. The ridge penalty and semantic feature scale are selected on grouped inner validation data. The deployment API is `SemanticMixtureRouter` in `mixbt.semantic`.

The semantic pool corrected 1,284 of the lexical pool's choices and introduced 1,120 new errors on the logged pairs. The net gain of 164 votes corresponds to **0.413 percentage points** (paired 95% interval 0.163 to 0.658). Log loss fell by **0.003614** (0.002930 to 0.004229). Among the 8,631 pairs where global BT predicted 45%–55%, accuracy rose from 53.85% to 55.64%, a net gain of 154. The other two prespecified groups contributed +13 and −3 correct choices. [Paired results](results/semantic_refinement/contrasts.csv) · [All three groups](results/semantic_refinement/difficulty.csv) · [Decision counts](results/decision_changes.csv)

Two ablations isolate different parts of the method. Replacing the original mixture's topic features with 32 embedding PCs raised EM accuracy from 65.203% to 65.633% at the same fold-specific component counts and penalties. A supplementary fit with 11 PCs matched the original mixture's 29 input columns in every fold and reached 65.744% accuracy (log loss 0.614312). The semantic pool correctly predicted 17 more votes than this matched-dimension EM fit, a difference of 0.043 percentage points (95% interval −0.227 to 0.295). Separately, a three-outcome Davidson model used eligible training ties to estimate a lexical branch and predicted decisive outcomes at evaluation. Its pool reached 65.339% and 0.615928. Compared with the lexical pool, the paired 95% intervals for both differences included zero. [32-PC mixture](results/semantic_mixture/contrasts.csv) · [11-PC comparison](results/semantic_mixture_capacity/contrasts.csv) · [Paired decision counts](results/decision_changes.csv) · [Tie ablation](results/tie_refinement/contrasts.csv)

#### Prepare frozen embeddings

Encoding uses a separate optional environment. The recorded GPU run used Python 3.10.20, PyTorch 2.5.1 with CUDA 12.1 and Transformers 4.48.3. The encoder checks hashes for five model and tokenizer files from revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, truncates to 128 tokens, mean-pools with the attention mask and normalises the 384-dimensional vectors. The 39,716 battles contained 36,580 distinct first-prompt strings. [Pinned MiniLM model](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42) · [PyTorch distributions](https://pytorch.org/get-started/previous-versions/#v251)

```sh
conda create -y -n arena-router-encoder python=3.10.20
conda run -n arena-router-encoder python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n arena-router-encoder python -m pip install -r requirements-encoder.txt
conda run -n arena-router-encoder python experiments/prepare_sentence_embeddings.py --device cuda --batch-size 96
```

A CPU PyTorch build and `--device cpu` use FP32. The recorded GPU run used mixed precision with FP32 pooling. The output is `data/semantic_minilm.npz`. With an offline model copy, pass `--model-dir` to the encoder.

#### Reproduce the semantic and tie results

After preparing the historical five-fold reference, the lexical tuning CSV and the embeddings above, run the following from the repository root using the main Python environment. The NPZ files used for the decision audit are local per-battle predictions. The plot reads the published aggregate CSVs and can be regenerated directly from a clean checkout.

```sh
python experiments/semantic_refinement.py --reference ../arena-router-reference/results/nested --save-predictions results/semantic_refinement/local_oof.npz
python experiments/semantic_mixture.py --reference ../arena-router-reference/results/nested --components 32 --workers 3
python experiments/semantic_mixture.py --reference ../arena-router-reference/results/nested --components 11 --workers 3
python experiments/tie_refinement.py --reference ../arena-router-reference/results/nested --save-predictions results/tie_refinement/local_oof.npz
python experiments/decision_changes.py --reference ../arena-router-reference/results/nested --semantic-oof results/semantic_refinement/local_oof.npz --semantic-mixture-oof results/semantic_mixture/local_oof.npz --capacity-matched-oof results/semantic_mixture_capacity/local_oof.npz --tie-oof results/tie_refinement/local_oof.npz
python experiments/plot_semantic_gains.py
```

## Training and routing time

After preparing the Arena input, run `python experiments/edge_mstep_benchmark.py` for the matched skill-update benchmark. With 31,773 training votes, 1,225 unordered pairs, 64 candidates and 6 components, the median paired speedup over seven repetitions was 10.08×. Median times were 74.7 ms using vote rows and 7.3 ms using pair counts. The one-time pair indexing took 3.6 ms. Across the seven runs, the largest held-out probability difference was below 4 × 10⁻¹⁶. These timings cover one skill update with fixed responsibilities. The script also measures a control with one vote per pair. [Timing data](results/edge_mstep_benchmark.csv)

The historical complete-fit benchmark times estimation of the global prior, EM and the final L-BFGS refinement on the same training fold. Each implementation receives the same initial parameters and precomputed prompt features. To reproduce that comparison with its original solver, run

```sh
git -c core.autocrlf=false worktree add --detach ../arena-router-edge d0572e0c2c79103d95f3d2cef5961a60652988bf
python ../arena-router-edge/data/prepare_routing_input.py data/train.csv
python ../arena-router-edge/experiments/edge_fit_benchmark.py --components 2
python ../arena-router-edge/experiments/edge_fit_benchmark.py --components 6
```

For two components, the median paired full-fit speedup was 1.16× across three repetitions. With the six components selected on this fold, the paired speedups were 0.90×, 1.17× and 1.24×. Both paths used 30 EM iterations and 319 L-BFGS iterations in the six-component fit. Their maximum probability difference across 7,943 held-out votes was 6.26 × 10⁻⁶ or less. [Two-component timings](results/edge_fit_benchmark.csv) · [Six-component timings](results/edge_fit_benchmark_k6.csv)

The current solver evaluates pair-specific likelihood terms once per observed pair and reuses the objective at accepted L-BFGS steps. Run `python experiments/score_reuse_benchmark.py` on the prepared Arena data. Against the previous edge-based fitter, median complete-fit time in the six-component fold fell from 29.76 s to 16.07 s across three paired runs (median paired speedup 1.86×). Joint score calls fell from 643 to 324. All three runs produced identical objective histories and held-out probabilities in the pinned environment. [Score-reuse benchmark](results/score_reuse_benchmark.csv)

With a fixed candidate pool the mixture can reuse each candidate's within-component mean win probabilities. Run `python experiments/routing_latency.py` to time single requests on synthetic embeddings. For 64 candidates and six components, median routing time was 314 µs without the cache and 27 µs with a warm cache (11.38× median paired speedup). This measures routing from a supplied embedding. [Routing latency by pool size](results/routing_latency.csv)

The added lexical branch took 5.63 s for one refit on 31,773 training votes at the ridge penalty selected in fold zero. Including tokenisation, the branch used a median of 0.205 ms per prompt for a logged pair and 0.259 ms for all 64 candidates. These are single-thread batch timings over 7,943 held-out prompts, processed in groups of 256. [Lexical branch fit and inference timings](results/lexical_refinement/branch_timing.csv)

## Verification

```sh
python -m pip install pytest
python -m pytest -q tests
```

The tests cover derivatives, EM ascent, row and pair-count equivalence, restart iteration counts, routing probabilities, prompt grouping, training-only feature transforms and tied evaluation scores. Core probability models are in `mixbt/`, experiment entry points are in `experiments/`, and scripts for obtaining the data are in `data/`.
