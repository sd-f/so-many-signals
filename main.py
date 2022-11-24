import signal

from pandas import read_csv
import matplotlib.pyplot as plt
from tsmoothie.smoother import *
from scipy.signal import savgol_filter
from scipy import signal
import statsmodels.api as sm


def main():
    # TODO cutoff meta data in original file
    series = read_csv('data/BOS_spatial_1s_after4_adrenalin_P2_2_eine ROI_5Hz.txt',
                      header=37,  # watch for blank lines
                      index_col=0,
                      parse_dates=True, sep="\t")



    # simple rolling mean
    # https://www.statology.org/rolling-mean-pandas/

    s1 = series.copy()
    s1['rolling_mean'] = s1['Mean F1'].rolling(window=30).mean()
    s1.plot(title="rolling mean", linewidth=1)
    plt.savefig("plots/s1.png")

    # twice rolling mean

    s2 = series.copy()
    s2['rolling_mean_2'] = s2['Mean F1'].rolling(window=500).mean().rolling(window=20).mean()
    s2.plot(title="rolling mean (2x)", linewidth=1)
    plt.savefig("plots/s2.png")

    # savgol filter
    # https://www.datatechnotes.com/2022/05/smoothing-example-with-savitzky-golay.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

    s3 = series.copy()
    s3['savgol_filter_25'] = savgol_filter(s3['Mean F1'], window_length=25, polyorder=2)
    s3.plot(title="savitzky golay", linewidth=1)
    plt.savefig("plots/s3.png")

    # convolution smoother
    # https://pypi.org/project/tsmoothie/

    s4 = series.copy()
    smoother = ConvolutionSmoother(window_len=25, window_type='ones')
    s4['tsmoothie'] = smoother.smooth(s4['Mean F1']).smooth_data[0]
    s4.plot(title="convolution smoother", linewidth=1)
    plt.savefig("plots/s4.png")

    #  Notch filter (remove by frequency)
    # https://www.geeksforgeeks.org/design-an-iir-notch-filter-to-denoise-signal-using-python/

    s5 = series.copy()
    samp_freq = 5  # Sample frequency (Hz)
    notch_freq = 0.02  # Frequency to be removed from signal (Hz)
    quality_factor = 20.0  # Quality factor

    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    s5['notch_filter'] = signal.filtfilt(b_notch, a_notch, s5['Mean F1'])

    s5.plot(title="notch filter", linewidth=1)
    plt.savefig("plots/s5.png")

    # lowess filter
    # https://towardsdatascience.com/lowess-regression-in-python-how-to-discover-clear-patterns-in-your-data-f26e523d7a35
    # https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html

    s6 = series.copy()
    a = sm.nonparametric.lowess(s6['Mean F1'], s6.index, frac=0.3)
    s6['lowess'] = a[:, 1]
    s6.plot(title="lowess filter", linewidth=1)
    plt.savefig("plots/s6.png")

    # plt.show()
    plt.close()
    pass


if __name__ == "__main__":
    main()
