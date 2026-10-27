from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
import cv2


from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import Model
import numpy as np
import librosa
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils import resample
import os
import logging
from pathlib import Path

import tensorflow as tf
import tensorflow as tf
import soundfile as sf
import resampy



import os
import numpy as np
import soundfile as sf
import resampy

import tensorflow as tf

accuracy = []
precision = []
recall = []
fscore = []

model_folder = "model"


def Upload_Dataset():
    global file_name, categories, X, Y  # Use file_name instead of filename to match with feature pipeline
    text.delete('1.0', END)
    
    # Open folder selection dialog
    file_name = filedialog.askdirectory(initialdir=".")
    
    if not file_name:
        text.insert(END, "No directory selected.\n")
        return

    # Get category names from subdirectories
    categories = [d for d in os.listdir(file_name) if os.path.isdir(os.path.join(file_name, d))]
    
    if not categories:
        text.insert(END, "No class folders found in the selected directory.\n")
        return

    text.insert(END, 'Dataset loaded successfully.\n')
    text.insert(END, "Classes found in dataset: " + str(categories) + "\n")
    


def audio_feature_pipeline():
    global categories, file_name, X, Y

    def load_and_preprocess_audio(file_path, target_sr=16000):
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
        return audio.astype(np.float32), target_sr

    def extract_audio_features(audio, sr):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        # Flatten and concatenate only MFCC and Chroma features
        feature_vector = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1)
        ])
        
        return feature_vector


    if os.path.exists("X.npy") and os.path.exists("Y.npy"):
        print("Loading existing features and labels...")
        X = np.load("X.npy")
        Y = np.load("Y.npy")
        text.insert(END, f"Feature extraction loaded\n")
    else:
        print("Extracting handcrafted audio features...")
        features = []
        labels = []
        class_to_idx = {name: idx for idx, name in enumerate(categories)}

        for class_name in categories:
            class_dir = os.path.join(file_name, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.lower().endswith('.wav'):
                    file_path = os.path.join(class_dir, file)
                    print(file_path)
                    audio, sr = load_and_preprocess_audio(file_path)
                    if audio.size == 0:
                        continue
                    combined_features = extract_audio_features(audio, sr)
                    features.append(combined_features)
                    labels.append(class_to_idx[class_name])

        X = np.array(features)
        Y = np.array(labels)
        np.save("X.npy", X)
        np.save("Y.npy", Y)
        text.insert(END, f"Feature extraction completed and saved\n")

    return X, Y
    

def Train_Test_Splitting():
    global X,Y
    global x_train,y_train,x_test,y_test
    # Create a count plot
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END, "Feature extraction completed and saved\n")

    

def Calculate_Metrics(algorithm, predict, y_test, predict_proba=None):
    global categories, accuracy, precision, recall, fscore, text

    # Calculate metrics
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    # Store metrics
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    # Display metrics
    text.insert(END, algorithm + " Accuracy  :  " + str(a) + "\n")
    text.insert(END, algorithm + " Precision : " + str(p) + "\n")
    text.insert(END, algorithm + " Recall    : " + str(r) + "\n")
    text.insert(END, algorithm + " FScore    : " + str(f) + "\n")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predict)
    CR = classification_report(y_test, predict, target_names=categories)
    text.insert(END, algorithm + ' Classification Report \n')
    text.insert(END, algorithm + str(CR) + "\n\n")

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories, annot=True, cmap="magma", fmt="g")
    ax.set_ylim([0, len(categories)])
    plt.title(algorithm + " Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

    # ROC-AUC Score & Curve
    if predict_proba is not None:
        y_test_bin = label_binarize(y_test, classes=np.arange(len(categories)))

        # Check for at least two classes present in test set
        if np.unique(y_test).shape[0] < 2:
            text.insert(END, algorithm + " ROC AUC Score: Not defined (only one class in y_test)\n")
            return

        try:
            roc_auc = roc_auc_score(y_test_bin, predict_proba, average='macro', multi_class='ovr') * 100
            text.insert(END, algorithm + " ROC AUC Score: " + str(roc_auc) + "\n")

            # Plot ROC curves for each class
            plt.figure(figsize=(8, 6))
            for i in range(len(categories)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], predict_proba[:, i])
                roc_auc_i = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{categories[i]} (AUC = {roc_auc_i:.2f})')

            plt.plot([0, 1], [0, 1], 'k--')  # Diagonal
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(algorithm + ' ROC Curves')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

        except ValueError as e:
            text.insert(END, algorithm + f" ROC AUC")

from sklearn_lvq import GlvqModel

from sklearn_lvq import GlvqModel
import os
import joblib

def existing_classifier_GLVQ():
    global x_train, y_train, x_test, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "Existing_GLVQ_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading Existing GLVQ model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training GLVQ model...\n")
        mlmodel = GlvqModel()
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        text.insert(END, "Model saved at: " + model_filename + "\n")

    y_pred = mlmodel.predict(x_test)

    # Some versions may not support predict_proba
    try:
        y_proba = mlmodel.predict_proba(x_test)
    except AttributeError:
        y_proba = None

    Calculate_Metrics("Existing GLVQ", y_pred, y_test, y_proba)



from sklearn.linear_model import Perceptron
import joblib
import os

def existing_classifier_Perceptron():
    global x_train, y_train, x_test, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "Existing_Perceptron_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading Existing Perceptron model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training Perceptron model...\n")
        mlmodel = Perceptron(max_iter=8, eta0=1.0, random_state=42)
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        text.insert(END, "Model saved at: " + model_filename + "\n")

    y_pred = mlmodel.predict(x_test)

    # Perceptron doesn't support predict_proba; set y_proba to None
    y_proba = None

    Calculate_Metrics("Existing Perceptron", y_pred, y_test, y_proba)



from sklearn_lvq import GlvqModel
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import joblib
import os


###############################################  DNN  #######################################################

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib

import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tkinter import END


class _CompatibleGlorotUniform(tf.keras.initializers.GlorotUniform):
    """Accept the dtype field written by newer Keras versions in legacy H5 files."""

    def __init__(self, seed=None, dtype=None, **kwargs):
        super().__init__(seed=seed)


class _CompatibleZeros(tf.keras.initializers.Zeros):
    """Accept the dtype field written by newer Keras versions in legacy H5 files."""

    def __init__(self, dtype=None, **kwargs):
        super().__init__()


def load_compatible_model(path):
    """Load project H5 models across Keras versions with legacy initializer configs."""
    try:
        return load_model(path, compile=False)
    except (TypeError, ValueError):
        return load_model(
            path,
            compile=False,
            custom_objects={
                "GlorotUniform": _CompatibleGlorotUniform,
                "Zeros": _CompatibleZeros,
            },
        )

def DNN_Classifier():
    global x_train, y_train, x_test, y_test, model_folder, y_pred, y_pred_labels, text

    text.delete('1.0', END)

    model_path = os.path.join(model_folder, "DNN_model.h5")

    # One-hot encode labels
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    input_dim = x_train.shape[1]
    output_dim = y_train_cat.shape[1]

    # Try loading the model
    if os.path.exists(model_path):
        try:
            text.insert(END, "Loading existing DNN model...\n")
            dnn_model = load_compatible_model(model_path)
        except Exception as e:
            text.insert(END, f"Error loading model: {str(e)}\nRe-training model...\n")
            dnn_model = None
    else:
        dnn_model = None

    # Train if model not loaded
    if dnn_model is None:
        dnn_model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_dim, activation='softmax')
        ])

        dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        dnn_model.fit(
            x_train, y_train_cat,
            validation_data=(x_test, y_test_cat),
            epochs=15,
            batch_size=32,
            verbose=2
        )

        dnn_model.save(model_path)
        text.insert(END, f"DNN model trained and saved at: {model_path}\n")

    # Predictions
    y_pred = dnn_model.predict(x_test)
    y_pred_labels = y_pred.argmax(axis=1)

    Calculate_Metrics("MLP Classifier", y_pred_labels, y_test, y_pred)




###############################################  DNN with Perceptron  #######################################################


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import Perceptron
import numpy as np
import os
import joblib

def DNN_Perceptron():
    global x_train, y_train, x_test, y_test, model_folder
    text.delete('1.0', END)

    # One-hot encoding for DNN training
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    extractor_path = os.path.join(model_folder, "DNN_feature_extractor.h5")
    perceptron_path = os.path.join(model_folder, "DNN_Perceptron_model.pkl")

    # Load or train DNN Feature Extractor
    if os.path.exists(extractor_path):
        text.insert(END, "Loading existing DNN Feature Extractor...\n")
        feature_extractor = load_compatible_model(extractor_path)
    else:
        text.insert(END, "Training DNN model...\n")
        input_dim = x_train.shape[1]

        # Define full DNN model
        full_model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),  # Feature layer
            Dense(y_train_cat.shape[1], activation='softmax')
        ])
        full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        full_model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat),
                       epochs=30, batch_size=32, verbose=0,
                       callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

        # Save feature extractor (exclude final softmax layer)
        feature_extractor = Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
        feature_extractor.save(extractor_path)
        text.insert(END, "Feature extractor saved at: " + extractor_path + "\n")

    # Extract features from DNN
    x_train_feat = feature_extractor.predict(x_train)
    x_test_feat = feature_extractor.predict(x_test)

    # Load or train Perceptron classifier
    if os.path.exists(perceptron_path):
        text.insert(END, "Loading existing Perceptron model...\n")
        perceptron_model = joblib.load(perceptron_path)
    else:
        text.insert(END, "Training Perceptron model...\n")
        perceptron_model = Perceptron(max_iter=1000, tol=1e-3)
        perceptron_model.fit(x_train_feat, y_train)
        joblib.dump(perceptron_model, perceptron_path)
        text.insert(END, "Perceptron model saved at: " + perceptron_path + "\n")

    # Predict and evaluate
    y_pred = perceptron_model.predict(x_test_feat)
    

    # Perceptron does not support predict_proba
    Calculate_Metrics("DNN + Perceptron", y_pred, y_test)




###############################################  PREDICITION  #######################################################
import os
import numpy as np
import librosa
import soundfile as sf
import resampy
import joblib
import cv2
import matplotlib.pyplot as plt
import sounddevice as sd
from tkinter import filedialog, END


def Prediction():
    global model_folder, categories, text, sf, np, sd

    # Select test audio file
    filename = filedialog.askopenfilename(initialdir="Test Data", title="Select Audio File", filetypes=[("WAV files", "*.wav")])
    text.delete('1.0', END)

    if not filename or not filename.lower().endswith(".wav"):
        text.insert(END, "Invalid file format. Please select a .wav file.\n")
        return

    text.insert(END, f"{filename} Loaded\n")

    def load_and_preprocess_audio(file_path, target_sr=16000):
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
        return audio.astype(np.float32), target_sr

    def extract_audio_features(audio, sr):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        feature_vector = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1)
        ])
        return feature_vector

    # Load and extract features from test audio
    audio, sr = load_and_preprocess_audio(filename)
    if audio.size == 0:
        text.insert(END, "Audio loading failed.\n")
        return

    # Play the audio
    text.insert(END, "Playing audio...\n")
    sd.play(audio, sr)
    sd.wait()

    raw_features = extract_audio_features(audio, sr).reshape(1, -1)

    # Load the DNN feature extractor
    extractor_path = os.path.join(model_folder, "DNN_feature_extractor.h5")
    if not os.path.exists(extractor_path):
        text.insert(END, "DNN feature extractor model not found.\n")
        return
    feature_extractor = load_compatible_model(extractor_path)

    # Extract deep features using DNN
    dnn_features = feature_extractor.predict(raw_features)

    # Load the Perceptron model
    perceptron_path = os.path.join(model_folder, "DNN_Perceptron_model.pkl")
    if not os.path.exists(perceptron_path):
        text.insert(END, "Trained Perceptron model not found.\n")
        return
    perceptron_model = joblib.load(perceptron_path)

    # Predict the class
    prediction = perceptron_model.predict(dnn_features)[0]
    predicted_class = categories[prediction] if prediction < len(categories) else str(prediction)
    text.insert(END, f"Predicted Outcome From Test Audio: {predicted_class}\n")

    # Generate and show waveplot
    y, sr = librosa.load(filename, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    waveplot_path = os.path.join(model_folder, "waveplot.png")
    plt.savefig(waveplot_path)
    plt.close()

    waveplot_img = cv2.imread(waveplot_path)
    if waveplot_img is not None:
        cv2.putText(waveplot_img, f"Prediction: {predicted_class}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Waveplot with Prediction", waveplot_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        text.insert(END, "Failed to load waveplot image.\n")




        

def close():
    main.destroy()


# ========================== MODERN UI (LOGIC UNCHANGED) ==========================

import tkinter as tk
from tkinter import ttk
from tkinter import Text, Scrollbar
from PIL import Image, ImageTk

main = tk.Tk()
main.title("Urban-Net | Emergency Sound Detection")
main.state('zoomed')
main.configure(bg="#0f172a")  # Dark slate background

# ---------------------- STYLE CONFIG ----------------------
style = ttk.Style(main)
style.theme_use("clam")

style.configure("TButton",
                font=("Segoe UI", 12, "bold"),
                padding=10,
                background="#1e293b",
                foreground="white")

style.map("TButton",
          background=[("active", "#334155")])

style.configure("TLabel",
                background="#0f172a",
                foreground="white",
                font=("Segoe UI", 12))

style.configure("Header.TLabel",
                font=("Segoe UI", 18, "bold"),
                foreground="#38bdf8")

# ---------------------- HEADER ----------------------
header = ttk.Label(
    main,
    text="Urban-Net: A Low-Latency Deep Learning Framework for Emergency Signal Extraction in Noisy Environments",
    style="Header.TLabel",
    anchor="center"
)
header.pack(pady=20)

# ---------------------- MAIN CONTAINER ----------------------
container = tk.Frame(main, bg="#0f172a")
container.pack(fill="both", expand=True, padx=20, pady=10)

# ---------------------- LEFT PANEL (CONTROLS) ----------------------
left_panel = tk.Frame(container, bg="#020617", width=320)
left_panel.pack(side="left", fill="y", padx=(0, 15))
left_panel.pack_propagate(False)

section_label = ttk.Label(left_panel, text="Control Panel", style="Header.TLabel")
section_label.pack(pady=15)

btn_cfg = {"fill": "x", "padx": 20, "pady": 6}

ttk.Button(left_panel, text="Upload Dataset", command=Upload_Dataset).pack(**btn_cfg)
ttk.Button(left_panel, text="MFCC + Chroma Features", command=audio_feature_pipeline).pack(**btn_cfg)
ttk.Button(left_panel, text="Train Test Split", command=Train_Test_Splitting).pack(**btn_cfg)

ttk.Separator(left_panel).pack(fill="x", padx=20, pady=10)

ttk.Button(left_panel, text="GLVQ Classifier", command=existing_classifier_GLVQ).pack(**btn_cfg)
ttk.Button(left_panel, text="Perceptron Classifier", command=existing_classifier_Perceptron).pack(**btn_cfg)
ttk.Button(left_panel, text="MLP Classifier", command=DNN_Classifier).pack(**btn_cfg)
ttk.Button(left_panel, text="DNN + Perceptron", command=DNN_Perceptron).pack(**btn_cfg)

ttk.Separator(left_panel).pack(fill="x", padx=20, pady=10)

ttk.Button(left_panel, text="Predict Test Audio", command=Prediction).pack(**btn_cfg)
ttk.Button(left_panel, text="Exit", command=close).pack(**btn_cfg)

# ---------------------- RIGHT PANEL (OUTPUT LOG) ----------------------
right_panel = tk.Frame(container, bg="#020617")
right_panel.pack(side="right", fill="both", expand=True)

output_label = ttk.Label(right_panel, text="System Output", style="Header.TLabel")
output_label.pack(anchor="w", padx=10, pady=10)

output_frame = tk.Frame(right_panel, bg="#020617")
output_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

text = Text(
    output_frame,
    bg="#020617",
    fg="#e5e7eb",
    insertbackground="white",
    font=("Consolas", 12),
    relief="flat",
    wrap="word"
)
text.pack(side="left", fill="both", expand=True)

scrollbar = Scrollbar(output_frame, command=text.yview)
scrollbar.pack(side="right", fill="y")

text.configure(yscrollcommand=scrollbar.set)

main.mainloop()
