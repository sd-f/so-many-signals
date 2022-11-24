from pandas import read_csv
import matplotlib.pyplot as plt
# from tsmoothie.smoother import *
# from scipy.signal import savgol_filter
# from scipy import signal
import statsmodels.api as sm


def main():
    # TODO cutoff meta data in original file
    series = read_csv('data/BOS_spatial_1s_after4_adrenalin_P2_2_eine ROI_5Hz.txt',
                      header=37,  # watch for blank lines
                      index_col=0,
                      parse_dates=True, sep="\t")

    # window_size = 50
    # original = series['Mean F1']

    # simple rolling mean
    # https://www.statology.org/rolling-mean-pandas/
    # series['rolling_mean'] = original.rolling(window=500).mean()
    # series['rolling_mean_2'] = series['rolling_mean'].rolling(window=20).mean()

    # savgol filter
    # https://www.datatechnotes.com/2022/05/smoothing-example-with-savitzky-golay.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    # series['savgol_filter'] = savgol_filter(original, window_length=window_size, polyorder=2)

    # convolution smoother
    # https://pypi.org/project/tsmoothie/
    # smoother = ConvolutionSmoother(window_len=window_size, window_type='ones')
    # series['tsmoothie'] = smoother.smooth(original).smooth_data[0]

    #  Notch filter (remove by frequency)
    # https://www.geeksforgeeks.org/design-an-iir-notch-filter-to-denoise-signal-using-python/
    # samp_freq = 5  # Sample frequency (Hz)
    # notch_freq = 0.02  # Frequency to be removed from signal (Hz)
    # quality_factor = 20.0  # Quality factor

    # b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    # series['notch_filter'] = signal.filtfilt(b_notch, a_notch, original)

    # lowess filter
    # https://towardsdatascience.com/lowess-regression-in-python-how-to-discover-clear-patterns-in-your-data-f26e523d7a35
    # https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    a = sm.nonparametric.lowess(series['Mean F1'], series.index, frac=0.3)
    series['lowess'] = a[:, 1]
    print(series.head())

    series.plot(linewidth=1)
    plt.show()
    plt.close()
    pass


if __name__ == "__main__":
    main()
