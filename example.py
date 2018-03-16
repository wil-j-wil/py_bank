# filter bank analysis of an audio signal
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import librosa as lb
import filterbanks as fb

y, fs = lb.load('stim118_puppy_whining.wav', sr=None)

DS = 2  # downsample rate
y = y[::DS]
fs = fs / DS

N = 16  # number of channels / filters
low_lim = 20  # centre freq. of lowest filter
high_lim = fs / 2  # centre freq. of highest filter
leny = y.shape[0]  # filter bank length

# %% Equal Rectangular Bandwidth example
# create an instance of the ERB filter bank class
erb_bank = fb.EqualRectangularBandwidth(leny, fs, N, low_lim, high_lim)

# generate subbands for signal y
erb_bank.generate_subbands(y)

# exclude the first (lowpass) and last (highpass) filters
# N.B. perfect reconstruction only possible with all filters
erb_subbands = erb_bank.subbands[:, 1:-1]

# calculate the envelopes using the Hilbert transform
erb_envs = np.transpose(np.absolute(sig.hilbert(np.transpose(erb_subbands))))

plt.plot(erb_bank.filters[:, 1:-1])  # plot the filter bank
plt.show()
plt.plot(erb_subbands)  # plot the subband signals
plt.show()
plt.plot(erb_envs)  # plot the subband envelopes
plt.show()

# %% Linear example
# create an instance of the linear filter bank class
linear_bank = fb.Linear(leny, fs, N, low_lim, high_lim)

# generate subbands for signal y
linear_bank.generate_subbands(y)

# exclude the first (lowpass) and last (highpass) filters
# N.B. perfect reconstruction only possible with all filters
linear_subbands = linear_bank.subbands[:, 1:-1]

# calculate the envelopes using the Hilbert transform
linear_envs = np.transpose(np.absolute(sig.hilbert(np.transpose(linear_subbands))))

plt.plot(linear_bank.filters[:, 1:-1])  # plot the filter bank
plt.show()
plt.plot(linear_subbands)  # plot the subband signals
plt.show()
plt.plot(linear_envs)  # plot the subband envelopes
plt.show()
