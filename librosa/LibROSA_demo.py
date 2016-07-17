# We'll need numpy for some mathematical operations
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Librosa for audio
import librosa

# matplotlib for displaying the output
import matplotlib.pyplot as plt

import tensorflow as tf

import seaborn
seaborn.set(style='ticks')

# and IPython.display for audio output
import IPython.display

audio_path = librosa.util.example_audio_file()

# or uncomment the line below and point it at your favorite song:

audio_path = './/BRITISH_ENGLISH.wav'

y, sr = librosa.load(audio_path)

print('HAS_SAMPLERATE: ', librosa.core.audio._HAS_SAMPLERATE)


####################
# Mel Spectrogram
####################
# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(S, ref_power=np.max)

# Make a new figure
# plt.figure(figsize=(12,4))
plt.figure()
plt.subplot(411)

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
# plt.tight_layout()
# plt.show()

####################
# MFCC
####################
# Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

# Let's pad on the first and second deltas while we're at it
delta_mfcc  = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

# How do they look?  We'll show each in its own subplot
# plt.figure(figsize=(12, 6))

plt.subplot(4,1,2)
librosa.display.specshow(mfcc)
plt.ylabel('MFCC')
plt.colorbar()

plt.subplot(4,1,3)
librosa.display.specshow(delta_mfcc)
plt.ylabel('MFCC-$\Delta$')
plt.colorbar()

plt.subplot(4,1,4)
librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
plt.ylabel('MFCC-$\Delta^2$')
plt.colorbar()

plt.tight_layout()

# For future use, we'll stack these together into one matrix
M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

plt.show()
