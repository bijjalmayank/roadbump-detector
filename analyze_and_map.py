import pandas as pd
import numpy as np
import joblib
import folium

# Load dataset
df = pd.read_csv('speed_bump_dataset.csv')

# Load scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('bump_model.pkl')

# FFT function (5 components)
def compute_fft_features(acc_values, n_components=5):
    fft_vals = np.fft.rfft(acc_values)
    fft_mags = np.abs(fft_vals)
    return [fft_mags[i] if i < len(fft_mags) else 0 for i in range(n_components)]

window_size = 64
fft_features = []
acc_z_values = df['acc_z_dashboard'].to_numpy()

for i in range(len(df)):
    window = np.zeros(window_size)
    if i > 0:
        if i < window_size:
            window[-i:] = acc_z_values[:i]
        else:
            window = acc_z_values[i-window_size:i]
    # when i == 0, window stays zeros

    fft_feats = compute_fft_features(window, n_components=5)
    fft_features.append(fft_feats)

fft_df = pd.DataFrame(fft_features, columns=[
    'acc_z_dashboard_fft_1',
    'acc_z_dashboard_fft_2',
    'acc_z_dashboard_fft_3',
    'acc_z_dashboard_fft_4',
    'acc_z_dashboard_fft_5',
])

# Use only original features that the model expects
feature_df = df[['acc_z_dashboard', 'speed']]

# Scale features
features_scaled = scaler.transform(feature_df)


# Predict
probs = model.predict_proba(features_scaled)[:, 1]

BEST_THRESHOLD = 0.91
df['predicted_bump'] = (probs >= BEST_THRESHOLD).astype(int)
df['probability'] = probs

bumps = df[df['predicted_bump'] == 1]

# Ask user for map center
try:
    fixed_lat = float(input("Enter map center latitude (e.g. 40.7128): "))
    fixed_lon = float(input("Enter map center longitude (e.g. -74.0060): "))
except:
    fixed_lat, fixed_lon = 40.7128, -74.0060
    print("Invalid input, using default NYC coordinates")

# Create map
bump_map = folium.Map(location=[fixed_lat, fixed_lon], zoom_start=17)

if not bumps.empty:
    for _, row in bumps.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=(
                f"Z: {row['acc_z_dashboard']:.2f}, "
                f"Speed: {row['speed']} km/h, "
                f"Prob: {row['probability']:.2f}"
            ),
            color='red',
            fill=True,
            fill_opacity=0.8
        ).add_to(bump_map)
    bump_map.save('bump_map.html')
    print("✅ Map saved as bump_map.html")
else:
    print("❌ No bumps detected by the model.")
