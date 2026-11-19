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

For the Arena evaluation, download the [Arena human-preference dataset](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) and extract its routing fields. The evaluation uses the first prompt and the two candidate identities to predict the final recorded preference in a decisive battle. Outer folds keep identical normalised first prompts together. Feature transforms and hyperparameter selection use training data.

```sh
python data/download_data.py
python data/prepare_routing_input.py
python experiments/arena_nested.py --output results/nested --threads 2 --bootstrap 1000
python experiments/arena_descriptive.py --results-dir results/nested --threads 2
```

For a small smoke run, use `python experiments/arena_nested.py --quick --rows 1500 --output results/smoke --threads 1`. A completed fold can be reused with the same data, source fingerprints and configuration. For a changed configuration, use a new output directory.

After preparing the Arena input, run `python experiments/edge_mstep_benchmark.py` for the matched skill-update benchmark. With 31,773 training votes, 1,225 unordered pairs, 64 candidates and 6 components, the median paired speedup over seven repetitions was 10.08×. Median times were 74.7 ms using vote rows and 7.3 ms using pair counts. The one-time pair indexing took 3.6 ms. Across the seven runs, the largest held-out probability difference was below 4 × 10⁻¹⁶. These timings cover one skill update with fixed responsibilities. The script also measures a control with one vote per pair. [Timing data](results/edge_mstep_benchmark.csv)

The complete-fit benchmark times estimation of the global prior, EM and the final L-BFGS refinement on the same training fold. Each implementation receives the same initial parameters and precomputed prompt features. Run both component counts with

```sh
python experiments/edge_fit_benchmark.py --components 2
python experiments/edge_fit_benchmark.py --components 6
```

For two components, the median paired full-fit speedup was 1.16× across three repetitions. With the six components selected on this fold, the paired speedups were 0.90×, 1.17× and 1.24×. Both paths used 30 EM iterations and 319 L-BFGS iterations in the six-component fit. Their maximum probability difference across 7,943 held-out votes was 6.26 × 10⁻⁶ or less. [Two-component timings](results/edge_fit_benchmark.csv) · [Six-component timings](results/edge_fit_benchmark_k6.csv)

## Verification

```sh
python -m pip install pytest
python -m pytest -q tests
```

The tests cover derivatives, EM ascent, row and pair-count equivalence, restart iteration counts, routing probabilities, prompt grouping, training-only feature transforms and tied evaluation scores. Core probability models are in `mixbt/`, experiment entry points are in `experiments/`, and scripts for obtaining the data are in `data/`.
