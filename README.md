# SC_Project_26Fall_Router

A prompt-aware router learns which LLM response people tend to prefer for a given request. It fits a finite mixture of Bradley–Terry comparisons to pairwise votes. Prompt features determine the mixture weights, and each component estimates a separate vector of LLM skills. For a new prompt the router selects the candidate with the highest predicted average win probability against the other candidates.

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

For the Arena evaluation, download the [Arena human-preference dataset](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) and extract its routing fields. The evaluation uses the first prompt and the two candidate identities to predict the final recorded preference in a decisive battle. Outer folds keep identical normalised first prompts together. Feature transforms and hyperparameter selection use training data.

```sh
python data/download_data.py
python data/prepare_routing_input.py
python experiments/arena_nested.py --output results/nested --threads 2 --bootstrap 1000
python experiments/arena_descriptive.py --results-dir results/nested --threads 2
```

For a small smoke run, use `python experiments/arena_nested.py --quick --rows 1500 --output results/smoke --threads 1`. A completed fold can be reused with the same data, source fingerprints and configuration. For a changed configuration, use a new output directory.

## Verification

```sh
python -m pip install pytest
python -m pytest -q tests
```

The tests cover derivatives, EM ascent, routing probabilities, prompt grouping, training-only feature transforms and tied evaluation scores. Core probability models are in `mixbt/`, experiment entry points are in `experiments/`, and scripts for obtaining the data are in `data/`.
