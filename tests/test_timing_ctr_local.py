"""Local-polynomial model-free FWHM/FWTM (spec 011 T15, FR-10): bias and replicate spread against spec 008's."""
import numpy as np
import pytest
from scipy.special import ndtr

from src.timing import ctr

pytestmark = pytest.mark.fr("011-FR-10")

EDGES = np.linspace(-3000.0, 3000.0, 601)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


def expected(sigma, tail, events=540_000):
    """Binned mean counts of a Gaussian (+ wider tail) peak on 1.3 % flat background, and its true widths."""
    shape = (1 - tail) * np.diff(ndtr(EDGES / sigma)) + tail * np.diff(ndtr(EDGES / (2.5 * sigma)))
    background = 0.013 * events / len(CENTERS)
    fine = np.linspace(-3000.0, 3000.0, 600_001)
    density = np.interp(fine, CENTERS, events * shape)
    widths = {f: np.ptp(fine[density >= f * density.max()]) for f in (0.5, 0.1)}
    return events * shape + background, background, widths


@pytest.mark.parametrize("tail", [0.0, 0.3])
def test_local_widths_unbiased_and_steadier(tail):
    mu, background, truth = expected(200.0, tail)
    rng = np.random.default_rng(3)
    widths = {rule: {0.5: [], 0.1: []} for rule in ("old", "local")}
    for _ in range(60):
        counts = rng.poisson(mu).astype(float)
        for fraction in (0.5, 0.1):
            for rule, crossings in (("old", ctr.level_crossings), ("local", ctr.local_crossings)):
                left, right, _ = crossings(counts, CENTERS, background, fraction)
                widths[rule][fraction].append(right - left)
    for fraction in (0.5, 0.1):
        local, old = np.array(widths["local"][fraction]), np.array(widths["old"][fraction])
        assert abs(local.mean() / truth[fraction] - 1) < 0.006
        assert local.std() < 0.8 * old.std()


def test_fit_histogram_rule_selection():
    mu, background, _ = expected(200.0, 0.0)
    counts = np.random.default_rng(4).poisson(mu).astype(float)
    default = ctr.fit_histogram(counts, EDGES)
    left, right, _ = ctr.level_crossings(counts, CENTERS, default["background_events"] / len(CENTERS), 0.5)
    assert default["model_free_fwhm_ps"] == pytest.approx(right - left)          # spec 008 rule unchanged
    local = ctr.fit_histogram(counts, EDGES, crossings=ctr.local_crossings)
    left, right, _ = ctr.local_crossings(counts, CENTERS, local["background_events"] / len(CENTERS), 0.5)
    assert local["model_free_fwhm_ps"] == pytest.approx(right - left)
    assert local["fwhm_ps"] == default["fwhm_ps"]                                  # Gaussian fit untouched
    assert ctr.local_crossings(np.full(len(CENTERS), 50.0), CENTERS, 50.0, 0.5) is None
