import matplotlib.pyplot as plt
from music_tagger import extract_features

real_path = '/Users/nkroher/Desktop/audio-data/GTZAN/Blues/blues.00020.wav'
fake_path = '/Users/nkroher/code/tmc2/misc/synthetic_music_mir/music/Blues/Blues_run_6_choice_2_0.wav'

real = extract_features(real_path)
fake = extract_features(fake_path)

plt.imshow(real.T, origin='lower', aspect='auto')
plt.title('real')
plt.colorbar()
plt.figure()
plt.imshow(fake.T, origin='lower', aspect='auto')
plt.title('fake')
plt.colorbar()
plt.show()