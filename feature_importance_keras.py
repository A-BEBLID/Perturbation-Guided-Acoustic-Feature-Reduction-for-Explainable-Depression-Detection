import numpy as np

# === Core Strategies ===

def permutation_importance_keras(model, X, y, n_repeats=10):
    baseline_score = model.evaluate(X, y, verbose=0)[1]
    feature_importances = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            score = model.evaluate(X_permuted, y, verbose=0)[1]
            scores.append(score)
        feature_importances[i] = baseline_score - np.mean(scores)
    return feature_importances

def median_importance_keras(model, X, y):
    baseline_score = model.evaluate(X, y, verbose=0)[1]
    feature_importances = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        X_median = X.copy()
        median_value = np.median(X[:, i])
        X_median[:, i] = median_value
        score_median = model.evaluate(X_median, y, verbose=0)[1]
        feature_importances[i] = baseline_score - score_median
    return feature_importances

def mean_importance_keras(model, X, y):
    baseline_score = model.evaluate(X, y, verbose=0)[1]
    feature_importances = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        X_mean = X.copy()
        mean_value = np.mean(X[:, i])
        X_mean[:, i] = mean_value
        score_mean = model.evaluate(X_mean, y, verbose=0)[1]
        feature_importances[i] = baseline_score - score_mean
    return feature_importances

def max_importance_keras(model, X, y):
    baseline_score = model.evaluate(X, y, verbose=0)[1]
    feature_importances = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        X_max = X.copy()
        max_value = np.max(X[:, i])
        X_max[:, i] = max_value
        score_max = model.evaluate(X_max, y, verbose=0)[1]
        feature_importances[i] = baseline_score - score_max
    return feature_importances

def zero_importance_keras(model, X, y):
    baseline_score = model.evaluate(X, y, verbose=0)[1]
    feature_importances = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        X_zero = X.copy()
        X_zero[:, i] = 0.0
        score_zero = model.evaluate(X_zero, y, verbose=0)[1]
        feature_importances[i] = baseline_score - score_zero
    return feature_importances

def hybrid_importance_keras(model, X, y, n_repeats=10):
    baseline_score = model.evaluate(X, y, verbose=0)[1]
    feature_importances = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        drop_scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            score = model.evaluate(X_permuted, y, verbose=0)[1]
            drop_scores.append(baseline_score - score)

        X_min = X.copy()
        X_min[:, i] = np.min(X[:, i])
        score_min = model.evaluate(X_min, y, verbose=0)[1]
        drop_scores.append(baseline_score - score_min)

        X_max = X.copy()
        X_max[:, i] = np.max(X[:, i])
        score_max = model.evaluate(X_max, y, verbose=0)[1]
        drop_scores.append(baseline_score - score_max)

        feature_importances[i] = np.mean(drop_scores)
    return feature_importances

# === 3M Strategy ===
def three_m_importance_keras(model, X, y, seed=42):
    baseline_score = model.evaluate(X, y, verbose=0)[1]
    feature_importances = np.zeros(X.shape[1])
    rng = np.random.default_rng(seed)
    
    for i in range(X.shape[1]):
        choice = rng.integers(0, 3)  # 0=min, 1=median, 2=max
        X_mod = X.copy()
        if choice == 0:
            value = np.min(X[:, i])
        elif choice == 1:
            value = np.median(X[:, i])
        else:
            value = np.max(X[:, i])
        X_mod[:, i] = value
        score = model.evaluate(X_mod, y, verbose=0)[1]
        feature_importances[i] = baseline_score - score
    return feature_importances

# === Dispatcher ===
def get_feature_importance(strategy, model, X, y, n_repeats=10, all_importances_matrix=None):
    strategy = strategy.lower()
    if strategy == 'permutation':
        return permutation_importance_keras(model, X, y, n_repeats)
    elif strategy == 'median':
        return median_importance_keras(model, X, y)
    elif strategy == 'mean':
        return mean_importance_keras(model, X, y)
    elif strategy == 'max':
        return max_importance_keras(model, X, y)
    elif strategy == 'zero':
        return zero_importance_keras(model, X, y)
    elif strategy == 'hybrid':
        return hybrid_importance_keras(model, X, y, n_repeats)
    elif strategy == '3m':
        return three_m_importance_keras(model, X, y)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            "Choose from 'permutation', 'median', 'mean', 'max', 'zero', 'hybrid', or '3m'."
        )
