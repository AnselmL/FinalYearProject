function[freq] = mel_to_freq(mel)

freq = 700*(10.^(mel/2595.0) - 1.0);
