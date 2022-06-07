from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram





# Settings
original_audio_file = '../tests/1min.wav'
image_shape = (160,160)
spectrogram_save_path = './saved_spectrogram.png'

# Open as Audio
audio = Audio.from_file(original_audio_file)

# Convert into spectrogram
spectrogram = Spectrogram.from_audio(audio)

# Convert into image
image = spectrogram.to_image(shape=image_shape)

# Save image
image.save(spectrogram_save_path)




audio_path = '../tests/1min.wav'
audio_object = Audio.from_file(audio_path)




bandpassed = audio_object.bandpass(low_f = 1000, high_f = 5000)


length = audio_object.duration()
print(length)