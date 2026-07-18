import numpy as np

NPZ_PATH = (
    "<LOG PATH>"
    "61-70968-0010__32_019778.wav/"
    "3ec3c2f6-9b28-58f7-9267-f212b54f314b_0_amplitude.npz"
)

d = np.load(NPZ_PATH)

print("keys:", d.files)
# ['sample_rate', 'amp_original', 'amp_processed', 'freqs_hz',
#  'spectrum_original', 'spectrum_processed']

sr = int(d["sample_rate"])      # 16000
orig = d["amp_original"]        # (64000,) amplitude original
proc = d["amp_processed"]       # (64000,) amplitude after blur
freq = d["freqs_hz"]            # (32001,) freauency axis Hz
so = d["spectrum_original"]     # (32001,) magnitude spectrum original
sp = d["spectrum_processed"]    # (32001,) magnitude spectrum after blur

print("sample_rate     :", sr)
print("amp_original    :", orig.shape, orig.dtype)
print("amp_processed   :", proc.shape, proc.dtype)
print("freqs_hz        :", freq.shape)
print()
print("RMS original    :", np.sqrt(np.mean(orig ** 2)))
print("RMS processed   :", np.sqrt(np.mean(proc ** 2)))
print("ผลต่างสูงสุด (max|orig-proc|):", np.max(np.abs(orig - proc)))
