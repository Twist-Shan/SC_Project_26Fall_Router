"""Train and query a small synthetic prompt-aware router."""
import numpy as np

from mixbt import MixtureBT, make_params, simulate


def main():
    n_models, n_domains, n_features = 5, 2, 4
    weights, skills = make_params(n_models, n_domains, n_features, rng=17)
    train = simulate(1800, weights, skills, rng=43)
    new = simulate(4, weights, skills, rng=89)
    router = MixtureBT(K=n_domains, lam_W=0.1, lam_R=0.1, max_iter=60, seed=97)
    router.fit(train["F"], train["a"], train["b"], train["y"], M=n_models)
    candidates = np.arange(n_models)
    choices, scores = router.route(new["F"], candidates=candidates)
    for index, (choice, candidate_scores) in enumerate(zip(choices, scores), start=1):
        print(f"Prompt {index}: LLM {choice}, estimated mean win probability {candidate_scores.max():.3f}")


if __name__ == "__main__":
    main()
