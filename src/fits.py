from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import Callable

import numpy as np


def shift_to_centres(bin_low_edge: np.ndarray) -> np.ndarray:
    """
    This function shifts the bin edges to the bin centres.

    Parameters:
        - bins (np.ndarray): A numpy array of the bin edges.

    Returns:
    np.ndarray: A numpy array of the bin centres.
    """
    return bin_low_edge[:-1] + np.diff(bin_low_edge) * 0.5


def gaussian(
    x: float | np.ndarray, amp: float, mu: float, sigma: float
) -> float | np.ndarray:
    """
    This function calculates the Gaussian distribution.

    Parameters:
        - x (float | np.ndarray): The input array or a single float number.
        - amp (float): The amplitude of the Gaussian.
        - mu (float): The mean of the Gaussian.

    Returns:
    np.ndarray: A numpy array of the bin centres.
    """
    if sigma <= 0.0:
        return np.inf
    return amp * np.exp(-0.5 * (x - mu) ** 2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)

def double_gaussian(
    x: float | np.ndarray, amp1: float, mu1: float, sigma1: float, amp2: float, mu2: float, sigma2: float
) -> float | np.ndarray:
    """
    This function calculates the double Gaussian distribution.

    Parameters:
        - x (float | np.ndarray): The input array or a single float number.
        - amp1 (float): The amplitude of the first Gaussian.
        - mu1 (float): The mean of the first Gaussian.
        - sigma1 (float): The standard deviation of the first Gaussian.
        - amp2 (float): The amplitude of the second Gaussian.
        - mu2 (float): The mean of the second Gaussian.
        - sigma2 (float): The standard deviation of the second Gaussian.

    Returns:
    float | np.ndarray: The calculated double Gaussian distribution.
    """
    if sigma1 <= 0.0 or sigma2 <= 0.0:
        return np.inf
    return amp1 * np.exp(-0.5 * (x - mu1) ** 2 / sigma1**2) / (
        np.sqrt(2 * np.pi) * sigma1
    ) + amp2 * np.exp(-0.5 * (x - mu2) ** 2 / sigma2**2) / (
        np.sqrt(2 * np.pi) * sigma2
    )

def lorentzian(
    x: float | np.ndarray, amp: float, x0: float, gamma: float
) -> float | np.ndarray:
    """
    This function calculates the Lorentzian distribution.

    Parameters:
        - x (float | np.ndarray): The input array or a single float number.
        - amp (float): The amplitude of the Lorentzian.
        - x0 (float): The location parameter of the Lorentzian.
        - gamma (float): The scale parameter of the Lorentzian.

    Returns:
    float | np.ndarray: The calculated Lorentzian distribution.
    """
    if gamma <= 0:
        return np.inf
    return amp * gamma**2 / ((x - x0) ** 2 + gamma**2)


def fit_gaussian(
    data: np.ndarray,
    bins: np.ndarray,
    cb: int = 8,
    min_peak: int = 50,
    yerr: np.ndarray | None = None,
    pk_finder: str = "max",
    gaussian_str: str = "gaussian", # double_gaussian
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function fits a Gaussian to the data.

    Parameters:
        - data (np.ndarray): A numpy array of the data.
        - bins (np.ndarray): A numpy array of the bin edges.
        - cb (int, optional): An integer that defines the range around the maximum value.
        - min_peak (int, optional): The minimum height of the peaks.
        - yerr (np.ndarray | None, optional): A numpy array of uncertainties in the data.
        - pk_finder (str, optional): The method to find the peaks in the data.
        - gaussian_str (str, optional): The type of Gaussian to fit to the data. It can be either "gaussian" or "double_gaussian".

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of five numpy arrays:
        - bin_centres: A numpy array of the bin centres.
        - gaussian(bin_centres, *pars): A numpy array of the Gaussian distribution.
        - pars: The optimal values for the parameters of the Gaussian distribution.
        - pcov: The estimated covariance of `pars`.
        - chi_ndf: The chi-squared per degree of freedom.
    """
    bin_centres = shift_to_centres(bins)

    # Define limits around maximum.
    if data[np.argmax(data)] < min_peak:
        raise RuntimeError("Peak max below requirement.")

    if "peak" in pk_finder:
        max_val = data.max()
        peaks, _ = find_peaks(
            data,
            height=max_val / 2,
            distance=cb,
            prominence=min_peak // 2,
            width=cb // 2,
        )
        if peaks.shape[0] == 0:
            mu0, wsum, x, y, err = mean_around_max(data, bin_centres, cb, yerr)
        else:
            pk_indx = peaks.max()  # Use the peak furthest to the right
            mu0 = bin_centres[pk_indx]
            min_indx = max(pk_indx - cb, 0)
            max_indx = min(pk_indx + cb, len(bin_centres))
            x = bin_centres[min_indx:max_indx]
            y = data[min_indx:max_indx]
            wsum = y.sum()
            if yerr is None:
                err = np.sqrt(y, out=np.abs(y).astype("float"), where=y >= 0)
            else:
                err = yerr[min_indx:max_indx]
    elif pk_finder == "max":
        mu0, wsum, x, y, err = mean_around_max(data, bin_centres, cb, yerr)

    if mu0 is None:
        raise RuntimeError("No useful data available.")

    ## Initial values
    sig0 = np.average((x - mu0) ** 2, weights=y)
    if wsum > 1:
        sig0 *= wsum / (wsum - 1)
    sig0 = np.sqrt(sig0)

    if gaussian_str == "gaussian":
        gaussian_fn = gaussian
        p0=[wsum, mu0, sig0]
    elif gaussian_str == "double_gaussian":
        gaussian_fn = double_gaussian
        p0=[wsum, mu0, sig0, wsum, mu0, sig0]

    pars, pcov = curve_fit(gaussian_fn, x, y, sigma=err, p0=p0)
    chi_ndf = np.square((y - gaussian_fn(x, *pars)) / err).sum() / (y.shape[0] - 3)
    return bin_centres, gaussian_fn(bin_centres, *pars), pars, pcov, chi_ndf


def curve_fit_fn(
    fn: Callable, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, p0: list
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function fits a curve to the data using the least squares method.

    Parameters:
        - fn (Callable): A callable function that defines the model to fit to the data.
        - x (np.ndarray): A numpy array of x-coordinates of the data.
        - y (np.ndarray): A numpy array of y-coordinates of the data.
        - yerr (np.ndarray): A numpy array of uncertainties in the y-coordinates of the data.
        - p0 (list): A list of initial guess for the parameters of `fn`.

    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays:
        - pars: Optimal values for the parameters of `fn` that minimize the least squares error.
        - pcov: The estimated covariance of `pars`.
    """
    pars, pcov = curve_fit(fn, x, y, sigma=yerr, p0=p0)
    return pars, pcov


def mean_around_max(
    data: np.ndarray, bins: np.ndarray, cb: int, yerr: np.ndarray = None
) -> list[float | None]:
    """
    This function calculates the mean around the maximum value of the data.

    Parameters:
        - data (np.ndarray): A numpy array of the data.
        - bins (np.ndarray): A numpy array of the bin edges.
        - cb (int): An integer that defines the range around the maximum value.
        - yerr (np.ndarray, optional): A numpy array of uncertainties in the data.

    Returns:
    list[float | None]: A list of five elements:
        - None: The function seems to be incomplete, so it's unclear what the first three elements of the tuple should be.
        - yerr: A numpy array of uncertainties in the data around the maximum value.
    """
    bin_centres = shift_to_centres(bins)
    max_indx = np.argmax(data)
    first_bin = max(max_indx - cb, 0)
    last_bin = min(max_indx + cb, len(bin_centres))
    x = bin_centres[first_bin:last_bin]
    y = data[first_bin:last_bin]
    if sum(y) <= 0:
        return None, None, None, None, None
    if yerr is not None:
        yerr = yerr[first_bin:last_bin]
    else:
        yerr = np.sqrt(y, out=np.abs(y).astype("float"), where=y >= 0)
    return *np.average(x, weights=y, returned=True), x, y, yerr
