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
| Equal probability pool | 65.374% | 0.616032 |

Relative to the mixture, the pool gained 0.171 percentage points in accuracy (paired 95% bootstrap interval 0.000 to 0.342) and reduced log loss by 0.000665 (95% interval 0.000310 to 0.001017). These intervals condition on the fitted out-of-fold predictions. [All methods](results/lexical_refinement/metrics.csv) · [Paired intervals](results/lexical_refinement/contrasts.csv) · [Outer-fold results](results/lexical_refinement/folds.csv)

To reproduce the reported comparison, generate the reference predictions using the original five-fold source snapshot. Run these commands from the repository root after preparing the Arena data above. The 12 reference implementation files match the SHA-256 values recorded in the evaluation protocol. [Reference commit](https://github.com/Twist-Shan/SC_Project_26Fall_Router/tree/bfac1d6901f61783e263859ee6f2a4a8348d38c8)

```sh
git -c core.autocrlf=false worktree add --detach ../arena-router-reference bfac1d6901f61783e263859ee6f2a4a8348d38c8
python ../arena-router-reference/data/prepare_routing_input.py data/train.csv
python ../arena-router-reference/experiments/arena_nested.py --output ../arena-router-reference/results/nested --threads 2 --bootstrap 1000
python experiments/lexical_refinement.py --reference ../arena-router-reference/results/nested
python experiments/lexical_benchmark.py
```

The reference command writes `protocol.json` and `oof_predictions.npz` locally. The pooled evaluation checks battle order, exact first prompts and fold assignments, then reports paired differences with 1,000 prompt-group bootstrap draws. To evaluate the current mixture from scratch, run `python experiments/arena_nested.py --output results/nested --threads 2 --bootstrap 1000`. For a small smoke run, pass `--quick --rows 1500 --output results/smoke`. Run `python experiments/arena_descriptive.py --results-dir results/nested --threads 2` for descriptive summaries. Use a fresh output directory if the data, source or settings change.

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
