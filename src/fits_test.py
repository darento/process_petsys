from pytest import raises
import numpy as np

from fits import (
    shift_to_centres,
    gaussian,
    lorentzian,
    fit_gaussian,
    curve_fit_fn,
    mean_around_max,
)


def test_shift_to_centres():
    bins = np.array([0, 1, 2, 3, 4])
    expected = np.array([0.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(shift_to_centres(bins), expected)


def test_gaussian():
    mu = 0.0
    sig = 5.0
    x = np.linspace(-2000, 2000, 4000)
    g = gaussian(x, 1.0, mu, sig)

    std = np.average((x - mu) ** 2, weights=g)
    np.testing.assert_almost_equal(np.average(x, weights=g), mu)
    np.testing.assert_almost_equal(np.sqrt(std), sig)
    np.testing.assert_almost_equal(np.sum(g), 1.0, decimal=3)


def test_lorentzian():
    x0 = 0.0
    gamma = 5.0
    x = np.linspace(-2000, 2000, 4000)
    l = lorentzian(x, 1.0, x0, gamma)

    std = np.average((x - x0) ** 2, weights=l)
    np.testing.assert_almost_equal(np.average(x, weights=l), x0)
    np.testing.assert_almost_equal(np.sqrt(std), gamma)
    np.testing.assert_almost_equal(np.sum(l), 1.0, decimal=3)


def test_fit_gaussian():
    amp = 10000
    mu = 3
    sig = 2
    bin_wid = 0.2
    bin_vals, bin_edges = np.histogram(
        np.random.normal(mu, sig, amp), bins=np.arange(-50, 50, bin_wid)
    )
    bin_centres, gvals, pars, pcov, chi_ndf = fit_gaussian(bin_vals, bin_edges)

    np.testing.assert_allclose(bin_centres, bin_edges[:-1] + bin_wid / 2)
    rel_errors = np.sqrt(np.diag(pcov)) / pars
    assert np.isclose(np.sum(gvals), amp, rtol=3 * rel_errors[0])
    assert np.isclose(pars[1], mu, rtol=3 * rel_errors[1])
    assert np.isclose(pars[2], sig, rtol=3 * rel_errors[2])
    assert chi_ndf < 5


def test_fit_gaussian_raises():
    amp = 100
    mu = 3
    sig = 2
    bin_wid = 0.2
    bin_vals, bin_edges = np.histogram(
        np.random.normal(mu, sig, amp), bins=np.arange(-50, 50, bin_wid)
    )

    with raises(RuntimeError):
        dummy = fit_gaussian(bin_vals, bin_edges)


def test_fit_gaussian_peakfinder():
    amp = 10000
    mu = 3
    sig = 2
    bin_wid = 0.2
    g_shift = 20
    bin_vals, bin_edges = np.histogram(
        (np.random.normal(mu, sig, amp), np.random.normal(mu + g_shift, sig, amp)),
        bins=np.arange(-50, 50, bin_wid),
    )
    _, gvals, pars, pcov, chi_ndf = fit_gaussian(bin_vals, bin_edges, pk_finder="peak")

    rel_errors = np.sqrt(np.diag(pcov)) / pars
    assert np.isclose(np.sum(gvals), amp, rtol=3 * rel_errors[0])
    assert np.isclose(pars[1], mu + g_shift, rtol=3 * rel_errors[1])
    assert np.isclose(pars[2], sig, rtol=3 * rel_errors[2])
    assert chi_ndf < 5


def test_curve_fit_fn():
    xdata = np.linspace(-10, 10, 100)
    yerr = np.ones(100) * 0.1
    p0 = [1, 0]
    fn = lambda x, a, b: a * x + b
    ydata = fn(xdata, *p0)
    pars, pcov = curve_fit_fn(fn, xdata, ydata, yerr, p0)

    np.testing.assert_allclose(pars, p0, rtol=1e-5)
    assert pcov.shape == (2, 2)


def test_mean_around_max():
    mu = 0
    bin_vals, bin_edges = np.histogram(
        np.random.normal(mu, 2, 10000), bins=np.arange(-50, 50, 0.2)
    )
    edges = bin_edges[:-1]
    mean_val, wsum, x, y, _ = mean_around_max(bin_vals, edges, 6)

    max_indx = np.argmax(bin_vals)
    np.testing.assert_allclose(x, edges[max_indx - 6 : max_indx + 6])
    np.testing.assert_allclose(y, bin_vals[max_indx - 6 : max_indx + 6])
    # np.testing.assert_almost_equal(mean_val, edges[max_indx])
    np.testing.assert_almost_equal(wsum, np.sum(bin_vals[max_indx - 6 : max_indx + 6]))
