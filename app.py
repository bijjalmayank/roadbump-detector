from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('catboost_bump_model.pkl')
scaler = joblib.load('scaler.pkl')
best_thresh = joblib.load('best_threshold.pkl')

def compute_fft_features(acc_window, n_components=5):
    fft_vals = np.fft.rfft(acc_window)
    fft_mags = np.abs(fft_vals)
    return [fft_mags[i] if i < len(fft_mags) else 0 for i in range(n_components)]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        acc_window = data.get('acc_z_sequence', [])
        speed = float(data.get('speed', 0))

        if len(acc_window) < 64:
            return jsonify({'error': 'Need at least 64 acceleration values'}), 400

        acc_window = acc_window[-64:]  # keep last 64
        fft_feats = compute_fft_features(acc_window, n_components=5)
        
        acc_z_latest = float(acc_window[-1])  # last value in window

        features = [acc_z_latest, speed] + fft_feats
        scaled_features = scaler.transform([features])
        
        proba = model.predict_proba(scaled_features)[0][1]
        prediction = int(proba >= best_thresh)

        return jsonify({
            'predicted_bump': prediction,
            'probability': proba,
            'threshold_used': best_thresh
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

