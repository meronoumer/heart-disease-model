### **1\. Time-Domain and Spectral Features (Librosa)**

| Feature | What it tells you | Why it might matter |
| ----- | ----- | ----- |
| Mean RMS / Standard Dev RMS | How loud the heart sounds are and how much they vary | Murmurs may make heartbeats louder or more irregular |
| Skewness RMS | Whether sounds are louder earlier or later | E.g., AS may have crescendo-decrescendo patterns |
| Zero Crossing Rate | How often the signal crosses zero (rapid fluctuations) | May indicate turbulence or noise from regurgitation |
| Spectral Centroid / Bandwidth | Where the “energy” of the sound lies (low vs. high frequency) and how wide the spectrum is | Murmurs have higher frequency components than normal beats |
| Spectral Contrast | How much contrast there is between loud/quiet frequencies | Captures differences between normal and turbulent flow |
| MFCCs & MFCC deviation | Capture general *texture* and *tone shape* of the signal (inspired by speech) | Abnormal valve flow creates different frequency envelopes |
| Mel Spectrogram Mean/Dev | Time-frequency representation with perceptual scaling | Models can “see” murmur signatures in time+frequency |
| CQT Mean/Std/Skew | Captures pitch-related structure with high resolution at low frequencies | Heart sounds are low-pitched — CQT captures fine details |

### **2\. openSMILE Features (voice-quality, spectral, prosodic)**

| Feature | What it captures | Why it might help |
| ----- | ----- | ----- |
| loudness\_sma3\_amean / stddev | Overall loudness and variation | Helps detect weak vs. turbulent heartbeats |
| spectralFlux\_sma3\_amean | How quickly frequency content changes over time | Murmurs cause rapid change; normal heartbeats are stable |
| mfcc1–4\_sma3\_amean & stddevNorm | Lower-order MFCCs (macro structure of frequency) | Add robustness and redundancy to your MFCC set |
| HNR / amplitudeLogRelF0 | Ratio of harmonic (voiced) to noisy (unvoiced) components | Murmurs \= more noise, lower HNR |
| alphaRatio / hammarbergIndex | Balance between low and high frequency energy | These shift with turbulent flows in diseased hearts |
| slopeUV0-500, slopeUV500-1500 | How quickly unvoiced energy decreases in certain bands | More negative slopes \= sharper noise onset (e.g., murmurs) |
| loudnessPeaksPerSec / Segment Length | Rhythm & tempo of beats | Irregular beats can signal arrhythmias or valve delays |

### **3\.  Demographic \+ Label Features**

| Feature | Why it matters |
| ----- | ----- |
| Age, Gender, Smoker | Some heart diseases are more likely in older patients, men, or smokers |
| Labels (AS, MR, etc.) | These are your target outputs — what you're training the model to predict |

