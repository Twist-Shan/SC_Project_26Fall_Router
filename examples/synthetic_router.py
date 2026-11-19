"""Train and query a small synthetic prompt-aware router."""
import numpy as np

from mixbt import StyleMixtureBT, simulate


def main():
    # LLMs 0 and 1 specialise in opposite domains; LLM 2 is a generalist.
    weights = np.array([[0.0, 2.0], [0.0, 0.0]])
    skills = np.array([[1.6, -1.6], [-1.6, 1.6], [0.0, 0.0]])
    train = simulate(1800, weights, skills, rng=43)
    router = StyleMixtureBT(K=2, style=None, lam_R=2.0, max_iter=60, seed=97)
    router.fit(train["F"], train["a"], train["b"], train["y"], M=len(skills))

    # Each example has an intercept and one observed synthetic prompt coordinate.
    coordinates = [-2.0, -1.0, 1.0, 2.0]
    features = np.column_stack([np.ones(len(coordinates)), coordinates])
    choices, scores = router.route(features)
    for x, choice, row in zip(coordinates, choices, scores):
        print(f"x = {x:+.1f}   selected LLM {choice}   estimated mean win probability {row[choice]:.3f}")


if __name__ == "__main__":
    main()
