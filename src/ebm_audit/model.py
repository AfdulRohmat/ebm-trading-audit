"""Portable additive EBM inference and the frozen training recipe."""

import numpy as np

from .data import FEATURES


def contributions(model, x):
    x = np.asarray(x, dtype=float)
    if model["type"] != "ebm" or model["features"] != FEATURES:
        raise ValueError("Expected frozen additive EBM feature schema")
    if x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError("Wrong feature matrix shape")
    result = np.empty_like(x)
    for j, (cuts, scores) in enumerate(zip(model["cuts"], model["scores"])):
        idx = np.searchsorted(cuts, x[:, j], side="right") + 1
        idx[np.isnan(x[:, j])] = 0
        result[:, j] = np.asarray(scores)[idx]
    return result


def predict(model, x):
    # Preserve historical addition order for floating-point parity.
    terms = contributions(model, x)
    out = np.full(len(terms), model["intercept"])
    for column in terms.T:
        out += column
    return out


def create(max_rounds, seed=260926):
    from interpret.glassbox import ExplainableBoostingRegressor

    if max_rounds not in (200, 500):
        raise ValueError("Frozen grid contains only 200 and 500 rounds")
    return ExplainableBoostingRegressor(
        interactions=0,
        max_bins=32,
        outer_bags=1,
        inner_bags=0,
        validation_size=0,
        max_rounds=max_rounds,
        learning_rate=0.04,
        greedy_ratio=0,
        n_jobs=2,
        random_state=seed,
    )


def fit(train, validation, test, seed=260926):
    """Four prior years -> validation year -> five-year refit -> test year.

    Caller supplies chronological, nonoverlapping yearly splits; not used to
    regenerate the published result, which replays the frozen portable models.
    """
    assert train.entry_time.max() < validation.entry_time.min()
    assert validation.entry_time.max() < test.entry_time.min()
    assert train.exit_time.max() < validation.entry_time.min()
    assert validation.exit_time.max() < test.entry_time.min()
    x, v = train[FEATURES].to_numpy(), validation[FEATURES].to_numpy()
    scores = []
    for rounds in (200, 500):
        model = create(rounds, seed).fit(x, train.target)
        scores.append(
            (
                float(np.mean((model.predict(v) - validation.target.to_numpy()) ** 2)),
                rounds,
            )
        )
    rounds = min(scores)[1]
    model = create(rounds, seed).fit(
        np.concatenate([x, v]), np.r_[train.target, validation.target]
    )
    return model, model.predict(test[FEATURES].to_numpy()), scores
