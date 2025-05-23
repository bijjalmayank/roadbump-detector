import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib
from catboost import CatBoostClassifier

# Function to add FFT features
import numpy as np
import pandas as pd

def rolling_fft_features(df, col, window_size=64, n_components=5):
    fft_feats = np.zeros((len(df), n_components))
    vals = df[col].values
    for i in range(len(df)):
        if i < window_size:
            # Zero pad start if not enough points
            window = np.pad(vals[:i+1], (window_size - (i+1), 0), mode='constant')
        else:
            window = vals[i - window_size + 1: i + 1]
        fft_vals = np.fft.rfft(window)
        fft_magnitudes = np.abs(fft_vals)
        # Take first n_components (or pad if not enough)
        for j in range(n_components):
            fft_feats[i, j] = fft_magnitudes[j] if j < len(fft_magnitudes) else 0
    # Add FFT features to dataframe
    for k in range(n_components):
        df[f'{col}_fft_{k+1}'] = fft_feats[:, k]
    return df


# Load data
df = pd.read_csv('speed_bump_dataset.csv')

# Add FFT features on acc_z_dashboard
df = rolling_fft_features(df, 'acc_z_dashboard', window_size=64, n_components=5)


# Select features (original + FFT)
features = ['acc_z_dashboard', 'speed'] + [f'acc_z_dashboard_fft_{i}' for i in range(1, 6)]
X = df[features]
y = df['speed_bumps']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to balance training set
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("✅ After SMOTE:", pd.Series(y_train_bal).value_counts())

# Initialize and train CatBoost model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='F1',
    random_seed=42,
    verbose=False
)
model.fit(X_train_bal, y_train_bal)

# Predict probabilities on test set
probs = model.predict_proba(X_test)

# Find best threshold by F1-score
thresholds = np.arange(0.0, 1.0, 0.01)
f1_scores = []
for thresh in thresholds:
    y_pred_thresh = (probs[:, 1] >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))

best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Best threshold by F1-score: {best_thresh:.3f}")
print(f"Best F1-score: {max(f1_scores):.3f}")

# Predict with best threshold
y_pred_best = (probs[:, 1] >= best_thresh).astype(int)

# Evaluation with best threshold
print("\nConfusion Matrix (best threshold):\n", confusion_matrix(y_test, y_pred_best))
print("\nClassification Report (best threshold):\n", classification_report(y_test, y_pred_best))

# Save model, scaler, and threshold
joblib.dump(model, 'catboost_bump_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_thresh, 'best_threshold.pkl')
print("✅ CatBoost model, scaler, and threshold saved.")
