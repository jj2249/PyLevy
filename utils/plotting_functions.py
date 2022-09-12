import numpy as np
import matplotlib.pyplot as plt
import numbers

def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False, 
           rug_length=0.05, rug_kwargs=None, font_size=14, title_size=14, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=title_size)
    x1 = x
    y1 = y
    if ax is None:
        ax = plt.gca()
    if quantiles is None:
        quantiles = min(len(x1), len(y1))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles1 = np.quantile(x1, quantiles, interpolation=interpolation)
    y_quantiles1 = np.quantile(y1, quantiles, interpolation=interpolation)
    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x1:
            ax.axvline(point, **rug_x_params)
        for point in y1:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot and compare with y = x
    ax.scatter(x_quantiles1, y_quantiles1, c="black", label="Q-Q plot", **kwargs)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against each other
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label="45 degree line")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()


def histogramplot(rvs, pdf_vals, num_bins = 100, xlabel="", ylabel="", plottitle="", plottlabel="", ax=None):
    """ Function to compare generated process with density at t = T_{horizon} """
    if ax is None:
        ax = plt.gca()
    x1 = rvs
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plottitle)
    binvals, _, _ = plt.hist(x1, num_bins, density=True, label="Histogram of Process at $t = T_{horizon}$")
    ax.plot(np.linspace(min(x1), max(x1), len(x1)), pdf_vals, label=plottlabel)
    ax.legend()


def plot_filtering_results(times, x, x_dot, estimated_signal):
    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    ax1.plot(times, x, label="True Signal")
    ax2.plot(times, x_dot, label="True Signal")
    ax1.plot(times, estimated_signal[0], label="Estimated Signal")
    ax2.plot(times, estimated_signal[1], label="Estimated Signal")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Acceleration")
    ax1.set_ylabel("Velocity")
    ax1.legend()
    ax2.legend()
    plt.show()