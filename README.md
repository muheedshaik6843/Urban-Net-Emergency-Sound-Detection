# Urban-Net | Emergency Sound Detection

A desktop application for classifying emergency and urban sounds into ambulance, firetruck, and traffic categories. The Tkinter interface supports MFCC + chroma feature extraction, classifier evaluation, and WAV-file prediction.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`

## Run

```powershell
python -m pip install -r requirements.txt
python main.py
```

Select the `Dataset` folder in **Upload Dataset** before running feature extraction and the train/test split. Use **Predict Test Audio** to select a WAV file.

## Project contents

- `main.py` — Tkinter application and audio classification workflow
- `Dataset/` — labeled ambulance, firetruck, and traffic audio samples
- `testsounds/` — sample WAV files for prediction
- `model/` — pre-trained classifier and feature-extractor files
- `X.npy`, `Y.npy` — cached feature vectors and labels
