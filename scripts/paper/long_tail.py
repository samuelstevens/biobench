import beartype
import numpy as np
import tyro


@beartype.beartype
def linregress(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T  # design matrix [x 1]
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


@beartype.beartype
def zipf_slope(counts):
    f = np.sort(counts)[::-1]
    slope, *_ = linregress(np.log(np.arange(1, len(f) + 1)), np.log(f))
    return -slope  # Zipf exponent s


@beartype.beartype
def shannon_eff(counts):
    p = counts[counts > 0] / counts.sum()
    H = -(p * np.log(p)).sum()
    return np.exp(H)  # effective classes N1


@beartype.beartype
def imbalance_ratio(counts):
    return counts.max() / counts.min()


@beartype.beartype
def gini(counts):
    y = np.sort(counts)
    n = len(y)
    cum = y.cumsum().astype(float)
    return 1 - 2 * (cum / cum[-1]).sum() / (n + 1)


@beartype.beartype
def tail_mass(counts, frac=0.9):
    y = np.sort(counts)[::-1]
    cum = y.cumsum()
    k = (cum >= frac * cum[-1]).argmax() + 1
    return k / len(y)


def main():
    pass


if __name__ == "__main__":
    tyro.cli(main)
