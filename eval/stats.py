"""Agreement statistics for ordinal relevance labels (0/1/2), without extra dependencies.

cohen_kappa: Cohen's kappa for two raters, unweighted or with linear/quadratic weights.
krippendorff_alpha: Krippendorff's alpha for any number of raters with missing values,
    nominal / ordinal / interval metric (Krippendorff 2011, "Computing Krippendorff's
    Alpha-Reliability"; tests reproduce its worked example).
"""

from collections import Counter
from itertools import permutations

LABELS = (0, 1, 2)


def raw_agreement(a, b):
    """Share of items with identical labels (a and b are equal-length sequences)."""
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else float("nan")


def confusion_matrix(a, b, categories=LABELS):
    """counts[i][j] = items labeled categories[i] by a and categories[j] by b."""
    index = {c: i for i, c in enumerate(categories)}
    m = [[0] * len(categories) for _ in categories]
    for x, y in zip(a, b):
        m[index[x]][index[y]] += 1
    return m


def cohen_kappa(a, b, weights=None, categories=LABELS):
    """Cohen's kappa; weights None (unweighted), "linear" or "quadratic" (ordinal categories)."""
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    n, k = len(a), len(categories)
    if n == 0:
        return float("nan")
    observed = confusion_matrix(a, b, categories)
    row = [sum(r) for r in observed]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if weights is None:
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1)
        return d if weights == "linear" else d * d if weights == "quadratic" else _bad(weights)

    num = sum(w(i, j) * observed[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * row[i] * col[j] / n for i in range(k) for j in range(k))
    return 1.0 - num / den if den else (1.0 if num == 0 else float("nan"))


def _bad(weights):
    raise ValueError(f"weights must be None, 'linear' or 'quadratic', got {weights!r}")


def krippendorff_alpha(units, level="ordinal"):
    """Krippendorff's alpha. units: one list of values per item (None = missing).

    Items with fewer than two values are not pairable and are ignored.
    """
    coincidence = Counter()
    for values in units:
        vals = [v for v in values if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for x, y in permutations(range(m), 2):
            coincidence[(vals[x], vals[y])] += 1.0 / (m - 1)
    if not coincidence:
        return float("nan")
    cats = sorted({c for pair in coincidence for c in pair})
    n_c = {c: sum(v for (x, _), v in coincidence.items() if x == c) for c in cats}
    n = sum(n_c.values())

    def delta2(c, k):
        if level == "nominal":
            return 0.0 if c == k else 1.0
        if level == "interval":
            return float(c - k) ** 2
        if level == "ordinal":
            lo, hi = sorted((cats.index(c), cats.index(k)))
            return (sum(n_c[cats[g]] for g in range(lo, hi + 1)) - (n_c[c] + n_c[k]) / 2.0) ** 2
        raise ValueError(f"level must be nominal, ordinal or interval, got {level!r}")

    d_o = sum(v * delta2(c, k) for (c, k), v in coincidence.items())
    d_e = sum(n_c[c] * n_c[k] * delta2(c, k) for c in cats for k in cats) / (n - 1)
    return 1.0 - d_o / d_e if d_e else (1.0 if d_o == 0 else float("nan"))
