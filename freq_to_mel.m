function[mel] = freq_to_mel(freq)

mel = 2595.0 * log10(1.0 + freq / 700.0);
