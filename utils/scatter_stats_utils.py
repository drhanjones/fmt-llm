import numpy as np
import matplotlib.pyplot as plt

def connect_dots(ax_obj, alpha=.4, col='grey', linewidth=2):
    """ for a given ax_object (from scatterplot or sns.stripplot or the like)
    connect the dots (arbtirary number of columns)"""
    cs = ax_obj.collections
    xcoords = np.vstack([this_cs.get_offsets()[:, 0] for this_cs in cs]).T
    ycoords = np.vstack([this_cs.get_offsets()[:, 1] for this_cs in cs]).T
    for (xcoord, ycoord) in zip(xcoords, ycoords):
        ax_obj.plot(xcoord, ycoord, alpha=alpha, color=col, linewidth=linewidth)

    return (ax_obj)


def fmt_boot_pval(pval, n_boots=10e4, scientific=False) -> str:
    """convertr bootstrap pvalues to expression that takes into account precision
    (e.g. p=0 will become p < x, with x being determined by number of bootstraps)"""
    if scientific:
        p_str = f'p={pval}' if pval > 0 else f'p < {round(1 / float(n_boots), int(np.log10(n_boots) + 1))}'
    else:
        if pval > 0:
            p_str = f'p={pval}'
        else:
            p_str = 'p < {atleast:.{decim}f}'.format(atleast=1 / float(n_boots), decim=int(np.log10(n_boots)))
    return (p_str)


def bootstrap_t_onesample(samps_in, pop_mean=0, tail='2s', n_boots=10e4, seed=123):
    """one-sample (paired) bootstrap t-test; returns p-value only

    in:
    - samps: nd.array, shape(n_samples)
        datapoints
    - pop_mean: float, Default=0
        mean to test against
    - tail: str, default: '2s'
        options: '2s','l','r' (for two-tailed,left or right-tailed)
    - n_boots: int; default=10e3
        number of bootstraps (determines precision)

    out:
    -pval: float
        fraction of instances where simulated null distribution returns
        test statistic that is at least as extreme as emprical test stat.
    see also:
    - fmt_boot_pval, function to format the pvalues, changes p=0 into P < (1/n_boots) statement
    dependencies: bootstrap from astropy
    """
    if seed is not None:
        np.random.seed(seed)
    # test stat
    t_func = lambda x, dim: (x.mean(dim) - pop_mean) / (x.std(dim) / np.sqrt(x.shape[dim]))
    # make null distribution
    null_boot_test = t_func(bootstrap(samps_in - samps_in.mean(0) + pop_mean, bootnum=int(n_boots)), 1)
    emp_test = t_func(samps_in, 0)
    # return p-value as probability of obtaining a test stat at least as extreme under the null
    if tail in ['2s', 'two', 'both']:
        left_pval = np.mean(null_boot_test < emp_test)
        right_pval = np.mean(null_boot_test > emp_test)
        return (2 * min(left_pval, right_pval))
    elif tail.lower() in ['l', 'left']:
        return (np.mean(null_boot_test < emp_test))
    elif tail.lower() in ['r', 'right']:
        return (np.mean(null_boot_test > emp_test))
    else:
        raise ValueError('tail not recognised!')


def bootstrap_analysis(data, pop_mean=0, tail='2s', n_boots=10000, seed=123):
    """
    Perform bootstrap analysis including one-sample t-test and compute 95% CI for the mean.

    Parameters:
    - data (array-like): Sample data.
    - pop_mean (float): Mean to test against.
    - tail (str): Type of the test ('2s' for two-tailed, 'l' for left-tailed, 'r' for right-tailed).
    - n_boots (int): Number of bootstrap samples.
    - seed (int): Seed for the random number generator.

    Returns:
    - dict: Contains the mean, 95% CI for the mean, and formatted p-value.
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute the p-value using the provided function
    p_value = bootstrap_t_onesample(data, pop_mean, tail, n_boots, seed)

    # Format the p-value
    formatted_p_value = fmt_boot_pval(p_value, n_boots)

    # Calculate the sample mean
    sample_mean = np.mean(data)

    # Generate bootstrap samples for the CI of the mean
    bootstrap_samples = bootstrap(data, bootnum=n_boots, bootfunc=np.mean)
    ci_lower, ci_upper = np.percentile(bootstrap_samples, [2.5, 97.5])

    # Compile results
    results = {
        'mean                    ': sample_mean,
        '95%_CI around mean      ': (ci_lower, ci_upper),
        'bootstrap-t-test p_value': formatted_p_value
    }

    return results


def bootstrap(data, bootnum=1000, samples=None, bootfunc=None, seed=False):
    """Performs bootstrap resampling on numpy arrays. (FUNCTION FROM ASTROPY)

    Bootstrap resampling is used to understand confidence intervals of sample
    estimates. This function returns versions of the dataset resampled with
    replacement ("case bootstrapping"). These can all be run through a function
    or statistic to produce a distribution of values which can then be used to
    find the confidence intervals.

    Parameters
    ----------
    data : numpy.ndarray
        N-D array. The bootstrap resampling will be performed on the first
        index, so the first index should access the relevant information
        to be bootstrapped.
    bootnum : int, optional
        Number of bootstrap resamples
    samples : int, optional
        Number of samples in each resample. The default `None` sets samples to
        the number of datapoints
    bootfunc : function, optional
        Function to reduce the resampled data. Each bootstrap resample will
        be put through this function and the results returned. If `None`, the
        bootstrapped data will be returned

    Returns
    -------
    boot : numpy.ndarray

        If bootfunc is None, then each row is a bootstrap resample of the data.
        If bootfunc is specified, then the columns will correspond to the
        outputs of bootfunc.

    """
    if seed != False:
        np.random.seed(seed)

    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)

    # create empty boot array
    boot = np.empty(resultdims)

    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)

        #         if seed != False: print(bootarr)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])

    return boot


def get_asterisk_map(pval):
    pval_c = float(pval.split('=')[1]) if 'p=' in pval else float(pval.split('<')[1])
    if pval_c < 0.001:
        return '***'
    elif pval_c < 0.01:
        return '**'
    elif pval_c < 0.05:
        return '*'
    else:
        return pval


def annotate_plot(ax, x1, x2, y, h, asterisk):
    # Ax = ax object
    # x1, x2 = x-coordinates of the bars, give absolute bar id location 0 to n_bar-1
    # y = y-coordinate of the bar, actual value in y axis where you want bar to be placed
    # h = height of the bar
    # asterisk = string to be placed on top of the bar, *,**,***, etc.

    if asterisk == '':
        return

    # Use annotate pairs to add bars on top of specific columns
    ax = plt.gca() if ax is None else ax

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')

    if asterisk in ['*', '**', '***']:
        ax.text((x1 + x2) * .5, y + h, asterisk, ha='center', va='bottom', color='black')
    else:
        ax.text((x1 + x2) * .5, y + h * 2, asterisk, ha='center', va='bottom', color='black', fontsize=8)


# add variable for comparison data (Validation Loss or BLIMP scores or Reading Time)
def annotate_given_colid(ax, col_id1, col_id2, plot_data, y, h, plotcolname, group_by="seed"):
    col_names = [tick.get_text() for tick in ax.get_xticklabels()]
    col_names = {i: col_name for i, col_name in enumerate(col_names)}
    # print(col_names)
    # Using the column names, get the data for each condition #Just do for 0,1 for now
    print("Comparing", col_names[col_id1], "and", col_names[col_id2])
    stats_test_subset = plot_data[plot_data["condition"].isin([col_names[col_id1], col_names[col_id2]])]
    # Get pairwise differences
    pairwise_diff = stats_test_subset.groupby(group_by).apply(
        lambda x: x[x['condition'] == col_names[col_id1]][plotcolname].values[0] -
                  x[x['condition'] == col_names[col_id2]][plotcolname].values[0])

    # perform bootstrap-t-test analysis on pairwise values (https://open.lnu.se/index.php/metapsychology/article/view/2058)
    res = bootstrap_analysis(pairwise_diff.values, )  # alternatively, do: tail='r' to test for delta>0 (one-sided)
    # print(res)
    # Use results to annotate the plot
    annotate_plot(ax, col_id1, col_id2, y, h, get_asterisk_map(res['bootstrap-t-test p_value']))
    return res


# ib = individual blimp
# Different way to connect dots for individual blimp scores, because pair-wise comp

def connect_dots_ib(ax_obj, alpha=.4, col='grey', linewidth=2):
    """ for a given ax_object (from scatterplot or sns.stripplot or the like)
    connect the dots (arbtirary number of columns)"""
    cs = ax_obj.collections
    xcoords = np.vstack([this_cs.get_offsets()[:, 0] for this_cs in cs]).T
    ycoords = np.vstack([this_cs.get_offsets()[:, 1] for this_cs in cs]).T
    for (xcoord, ycoord) in zip(xcoords, ycoords):
        xcoord = [xcoord[0::2], xcoord[1::2]]
        ycoord = [ycoord[0::2], ycoord[1::2]]
        ax_obj.plot(xcoord, ycoord, alpha=alpha, color=col, linewidth=linewidth)

    return (ax_obj)


def annotate_plot_ib(ax, x1, y, h, asterisk):
    # Ax = ax object
    # x1, x2 = x-coordinates of the bars, give absolute bar id location 0 to n_bar-1
    # y = y-coordinate of the bar, actual value in y axis where you want bar to be placed
    # h = height of the bar
    # asterisk = string to be placed on top of the bar, *,**,***, etc.

    if asterisk == '':
        return

    # Use annotate pairs to add bars on top of specific columns
    ax = plt.gca() if ax is None else ax

    # ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    # Given a x1, plot within its hue
    ax.plot([x1 - 0.2, x1 - 0.2, x1 + 0.2, x1 + 0.2], [y, y + h, y + h, y], lw=1.5, c='black')

    if asterisk in ['*', '**', '***']:
        ax.text((x1 + x1) * .5, y + h, asterisk, ha='center', va='bottom', color='black')
    else:
        ax.text((x1 + x1) * .5, y + h * 2, asterisk, ha='center', va='bottom', color='black', fontsize=8)


def annotate_given_colid_ib(ax, col_id1, plot_data, y, h, conditional_map):
    col_names = [tick.get_text() for tick in ax.get_xticklabels()]
    col_names = {i: col_name for i, col_name in enumerate(col_names)}

    # Annotate within hue
    # res = annotate_given_colid(ax, 0, 1, plot_data, 90, 2, "score", col_names[0])

    stats_test_subset = \
    plot_data[plot_data["condition"].isin(conditional_map.values())][["condition", "seed", "score"]][
        plot_data["blimp_task"] == col_names[col_id1]]

    pairwise_diff = stats_test_subset.groupby('seed').apply(
        lambda x: x[x['condition'] == list(conditional_map.values())[1]]["score"].values[0] -
                  x[x['condition'] == list(conditional_map.values())[0]]["score"].values[0])

    res = bootstrap_analysis(pairwise_diff.values, )  # alternatively, do: tail='r' to test for delta>0 (one-sided)
    annotate_plot_ib(ax, col_id1, y, h, get_asterisk_map(res['bootstrap-t-test p_value']))

    return res

