import numpy as np
import pandas as pd
import json
import os
import pickle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
import joblib
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class AdvancedLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_classes=3, dropout=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        context = self.attention(lstm_out)
        return self.classifier(context)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def calculate_optimized_features(df):
    """
    Calculate optimized features for hit/bounce detection.
    MUST be identical to training feature engineering.
    """
    df = df.copy()

    # === BASIC CLEANING ===
    df['x'] = df['x'].replace(0, np.nan).interpolate(method='cubic').bfill().ffill()
    df['y'] = df['y'].replace(0, np.nan).interpolate(method='cubic').bfill().ffill()

    # === SMOOTHING ===
    df['x_smooth'] = gaussian_filter1d(df['x'], sigma=1.5)
    df['y_smooth'] = gaussian_filter1d(df['y'], sigma=1.5)

    # === 1ST ORDER: VELOCITY ===
    df['vx'] = df['x_smooth'].diff()
    df['vy'] = df['y_smooth'].diff()
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['v_horizontal'] = df['vx'].abs()
    df['v_vertical'] = df['vy'].abs()

    # === 2ND ORDER: ACCELERATION ===
    df['ax'] = df['vx'].diff()
    df['ay'] = df['vy'].diff()
    df['accel_magnitude'] = np.sqrt(df['ax']**2 + df['ay']**2)
    df['accel_horizontal'] = df['ax'].abs()
    df['accel_vertical'] = df['ay'].abs()

    # === 3RD ORDER: JERK ===
    df['jerk_x'] = df['ax'].diff()
    df['jerk_y'] = df['ay'].diff()
    df['jerk_magnitude'] = np.sqrt(df['jerk_x']**2 + df['jerk_y']**2)

    # === ANGLE & DIRECTION ===
    df['angle'] = np.arctan2(df['vy'], df['vx'])
    df['delta_angle'] = df['angle'].diff().abs()
    df.loc[df['delta_angle'] > np.pi, 'delta_angle'] = \
        2*np.pi - df.loc[df['delta_angle'] > np.pi, 'delta_angle']
    df['angular_velocity'] = df['delta_angle'] / (df['speed'] + 1e-6)

    # === VERTICAL VELOCITY ANALYSIS ===
    df['vy_change'] = df['vy'].diff()
    df['vy_abs_change'] = df['vy_change'].abs()
    df['vy_sign_change'] = ((df['vy'] * df['vy'].shift(1)) < 0).astype(int)
    df['vy_acceleration'] = df['vy'].diff(2)

    # === HEIGHT & POSITION ===
    max_y = df['y'].max()
    min_y = df['y'].min()
    y_range = max_y - min_y if (max_y - min_y) > 0 else 1

    df['height_normalized'] = (max_y - df['y']) / y_range
    df['height_raw'] = max_y - df['y']

    center_x = (df['x'].max() + df['x'].min()) / 2
    df['dist_from_center'] = (df['x'] - center_x).abs()

    # === ENERGY & MOMENTUM ===
    df['kinetic_energy'] = df['speed']**2
    df['energy_change'] = df['kinetic_energy'].diff()
    df['energy_change_rate'] = df['energy_change'] / (df['kinetic_energy'].shift(1) + 1e-6)

    # === CURVATURE ===
    df['curvature'] = np.abs(df['ax'] * df['vy'] - df['ay'] * df['vx']) / \
                      (df['speed']**3 + 1e-6)

    # === TEMPORAL CONTEXT ===
    temporal_features = ['speed', 'vy', 'accel_magnitude', 'jerk_magnitude',
                        'height_normalized', 'delta_angle']

    for lag in [1, 2, 3, 5]:
        for feature in temporal_features:
            if feature in df.columns:
                df[f'{feature}_past_{lag}'] = df[feature].shift(lag)
                df[f'{feature}_future_{lag}'] = df[feature].shift(-lag)
                df[f'{feature}_diff_{lag}'] = df[feature] - df[feature].shift(lag)

    for window in [3, 5]:
        for feature in temporal_features:
            if feature in df.columns:
                df[f'{feature}_mean_{window}'] = df[feature].rolling(window, center=True).mean()
                df[f'{feature}_std_{window}'] = df[feature].rolling(window, center=True).std()
                df[f'{feature}_max_{window}'] = df[feature].rolling(window, center=True).max()
                df[f'{feature}_min_{window}'] = df[feature].rolling(window, center=True).min()

    df['speed_change'] = df['speed'].diff()
    df['accel_change'] = df['accel_magnitude'].diff()
    df['height_change'] = df['height_normalized'].diff()

    # === PEAK DETECTION ===
    df['is_speed_peak'] = 0
    df['is_accel_peak'] = 0
    df['is_jerk_peak'] = 0
    df['is_y_local_max'] = 0
    df['is_y_local_min'] = 0

    if len(df) > 5:
        try:
            speed_peaks, _ = find_peaks(df['speed'].values, distance=2, prominence=df['speed'].std()*0.5)
            accel_peaks, _ = find_peaks(df['accel_magnitude'].values, distance=2)
            jerk_peaks, _ = find_peaks(df['jerk_magnitude'].values, distance=2)
            y_maxs, _ = find_peaks(df['y_smooth'].values, distance=3)
            y_mins, _ = find_peaks(-df['y_smooth'].values, distance=3)

            if len(speed_peaks) > 0:
                df.iloc[speed_peaks, df.columns.get_loc('is_speed_peak')] = 1
            if len(accel_peaks) > 0:
                df.iloc[accel_peaks, df.columns.get_loc('is_accel_peak')] = 1
            if len(jerk_peaks) > 0:
                df.iloc[jerk_peaks, df.columns.get_loc('is_jerk_peak')] = 1
            if len(y_maxs) > 0:
                df.iloc[y_maxs, df.columns.get_loc('is_y_local_max')] = 1
            if len(y_mins) > 0:
                df.iloc[y_mins, df.columns.get_loc('is_y_local_min')] = 1
        except:
            pass

    # === COMPOSITE FEATURES ===
    df['hit_score'] = (
        df['jerk_magnitude'] *
        df['accel_magnitude'] *
        df['delta_angle'] *
        (1 + df['is_jerk_peak'])
    )

    df['bounce_score'] = (
        df['vy_abs_change'] *
        (1 - df['height_normalized']) *
        df['vy_sign_change'] *
        (1 + df['is_y_local_min'])
    )

    df['vertical_impact'] = df['vy_abs_change'] * df['accel_vertical']

    # === VELOCITY REVERSAL ===
    df['vy_reversed_recently'] = (
        (df['vy'] * df['vy_past_1'] < 0) |
        (df['vy_past_1'] * df['vy_past_2'] < 0)
    ).astype(int)

    df['vy_will_reverse'] = (
        (df['vy'] * df['vy_future_1'] < 0) |
        (df['vy_future_1'] * df['vy_future_2'] < 0)
    ).astype(int)

    df['vy_reversal_window'] = (
        df['vy_reversed_recently'] |
        df['vy_sign_change'] |
        df['vy_will_reverse']
    ).astype(int)

    # === SUSTAINED EVENTS ===
    df['jerk_sustained'] = (
        df['jerk_magnitude_past_3'] +
        df['jerk_magnitude'] +
        df['jerk_magnitude_future_3']
    ) / 3

    df['accel_sustained'] = (
        df['accel_magnitude_past_3'] +
        df['accel_magnitude'] +
        df['accel_magnitude_future_3']
    ) / 3

    # === COMPARATIVE ===
    for feature in ['speed', 'vy', 'height_normalized']:
        if feature in df.columns:
            past_mean = df[f'{feature}_mean_5']
            future_mean = df[feature].shift(-5).rolling(5).mean()

            df[f'{feature}_past_vs_future'] = past_mean - future_mean

            df[f'{feature}_is_peak'] = (
                (df[feature] > past_mean) &
                (df[feature] > future_mean)
            ).astype(int)

    df['trajectory_phase'] = np.arctan2(df['vy'], df['vx'] + 1e-6)
    df['trajectory_phase_change'] = df['trajectory_phase'].diff().abs()

    df['frame_idx'] = range(len(df))
    df['time_in_point'] = df['frame_idx'] / len(df)

    return df.fillna(0)

# ============================================================================
# UNSUPERVISED METHOD (Physics-Based)
# ============================================================================

def physics_method(df, params=None):
    """
    Enhanced physics-based detection with optimized thresholds.
    """
    df = df.copy()
    df['pred_action'] = 'air'
    
    # Default optimized parameters
    if params is None:
        params = {
            'angle_threshold': 0.35,
            'jerk_percentile': 92,
            'accel_percentile': 88,
            'height_threshold': 0.25,
            'min_speed': 3
        }
    
    # Calculate thresholds
    angle_threshold = params['angle_threshold']
    jerk_threshold = df['jerk_magnitude'].quantile(params['jerk_percentile'] / 100)
    accel_threshold = df['accel_magnitude'].quantile(params['accel_percentile'] / 100)
    height_threshold = params['height_threshold']
    
    # Find candidate events
    candidates_mask = (
        (df['delta_angle'] > angle_threshold) |
        (df['jerk_magnitude'] > jerk_threshold) |
        (df['accel_magnitude'] > accel_threshold)
    ) & (df['speed'] > params['min_speed'])
    
    candidates = df[candidates_mask]
    
    if len(candidates) == 0:
        return df
    
    # Classify BOUNCES: near ground + velocity reversal
    bounce_mask = (
        (candidates['height_normalized'] < height_threshold) &
        ((candidates['vy_sign_change'] == 1) | (candidates['vy_reversed_recently'] == 1)) &
        (candidates['vy_change'] > candidates['vy_change'].quantile(0.4))
    )
    
    bounce_indices = candidates[bounce_mask].index
    df.loc[bounce_indices, 'pred_action'] = 'bounce'
    
    # Classify HITS: large angle change OR high jerk
    remaining = candidates[~candidates.index.isin(bounce_indices)]
    hit_mask = (
        (remaining['delta_angle'] > angle_threshold) |
        (remaining['jerk_magnitude'] > jerk_threshold * 0.8)
    )
    
    hit_indices = remaining[hit_mask].index
    df.loc[hit_indices, 'pred_action'] = 'hit'
    
    # Post-processing: remove isolated predictions
    for action in ['hit', 'bounce']:
        action_indices = df[df['pred_action'] == action].index.tolist()
        for idx in action_indices:
            window = df.iloc[max(0, idx-3):min(len(df), idx+4)]['pred_action']
            if (window == action).sum() == 1:
                df.at[idx, 'pred_action'] = 'air'
    
    return df


def unsupervised_hit_bounce_detection(json_path, params=None):
    """
    Unsupervised detection using physics-based method.
    
    Args:
        json_path: Path to input JSON file (ball_data_i.json)
        params: Optional dict of detection parameters
        
    Returns:
        dict: Enriched JSON with predictions
    """
    print(f"Loading data from: {json_path}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        point_data = json.load(f)
    
    # Convert to DataFrame
    frames = sorted(point_data.keys(), key=int)
    rows = []
    for f_idx in frames:
        details = point_data[f_idx]
        rows.append({
            "frame": int(f_idx),
            "x": details.get("x"),
            "y": details.get("y"),
            "visible": details.get("visible"),
            "action": details.get("action")
        })
    
    df = pd.DataFrame(rows)
    
    # Calculate features
    print("Calculating features...")
    df = calculate_optimized_features(df)
    
    # Apply physics-based detection
    print("Applying physics-based detection...")
    
    if params is None:
        params = {
            'angle_threshold': 0.35,
            'jerk_percentile': 92,
            'accel_percentile': 88,
            'height_threshold': 0.25,
            'min_speed': 3
        }
    
    print(f"  Parameters:")
    print(f"    Angle threshold: {params['angle_threshold']}")
    print(f"    Jerk percentile: {params['jerk_percentile']}")
    print(f"    Accel percentile: {params['accel_percentile']}")
    print(f"    Height threshold: {params['height_threshold']}")
    print(f"    Min speed: {params['min_speed']}")
    
    df = physics_method(df, params)
    
    # Create enriched JSON
    enriched_data = {}
    for idx, row in df.iterrows():
        frame_num = str(row['frame'])
        enriched_data[frame_num] = {
            "x": float(row['x']),
            "y": float(row['y']),
            "visible": int(row['visible']),
            "action": str(row['pred_action'])
        }
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"  Total frames: {len(df)}")
    print(f"  Air: {(df['pred_action'] == 'air').sum()}")
    print(f"  Bounces: {(df['pred_action'] == 'bounce').sum()}")
    print(f"  Hits: {(df['pred_action'] == 'hit').sum()}")
    
    # If ground truth available, show comparison
    if 'action' in df.columns and not df['action'].isna().all():
        df_eval = df[df['action'].notna()].copy()
        if len(df_eval) > 0:
            from sklearn.metrics import classification_report
            
            # Convert ground truth to string format if numeric
            label_map = {0: 'air', 1: 'bounce', 2: 'hit'}
            
            # Check if ground truth is numeric
            if pd.api.types.is_numeric_dtype(df_eval['action']):
                df_eval['action'] = df_eval['action'].map(label_map)
            
            # Ensure both are strings
            df_eval['action'] = df_eval['action'].astype(str)
            df_eval['pred_action'] = df_eval['pred_action'].astype(str)
            
            print("\n" + "="*80)
            print("COMPARISON WITH GROUND TRUTH")
            print("="*80)
            
            try:
                print(classification_report(df_eval['action'], df_eval['pred_action'], zero_division=0))
            except Exception as e:
                print(f"Could not generate classification report: {e}")
                print("\nManual comparison:")
                print(f"  Total labeled frames: {len(df_eval)}")
                for label in ['air', 'bounce', 'hit']:
                    true_count = (df_eval['action'] == label).sum()
                    pred_count = (df_eval['pred_action'] == label).sum()
                    correct = ((df_eval['action'] == label) & (df_eval['pred_action'] == label)).sum()
                    if true_count > 0:
                        acc = correct / true_count * 100
                        print(f"  {label.capitalize()}: {correct}/{true_count} correct ({acc:.1f}%)")
    
    return enriched_data



# ============================================================================
# SUPERVISED METHOD (Unified: ML or LSTM)
# ============================================================================

def supervised_hit_bounce_detection(json_path, model_dir='models', model_type='lstm'):
    """
    Unified supervised detection using either LSTM or ML (XGBoost) model.
    
    Args:
        json_path: Path to input JSON file (ball_data_i.json)
        model_dir: Directory containing model files
        model_type: 'lstm' (default, better results) or 'ml' (XGBoost)
        
    Returns:
        dict: Enriched JSON with predictions
    """
    model_type = model_type.lower()
    
    if model_type not in ['lstm', 'ml']:
        raise ValueError(f"model_type must be 'lstm' or 'ml', got '{model_type}'")
    
    print(f"Using {model_type.upper()} model for prediction")
    
    if model_type == 'lstm':
        return _supervised_lstm_detection(json_path, model_dir)
    else:
        return _supervised_ml_detection(json_path, model_dir)


def _supervised_ml_detection(json_path, model_dir='models'):
    """
    Internal: ML (XGBoost) detection with confidence thresholding.
    """
    AIR_CONFIDENCE_THRESHOLD = 0.7
    
    print(f"Loading data from: {json_path}")
    print(f"Air confidence threshold: {AIR_CONFIDENCE_THRESHOLD}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        point_data = json.load(f)
    
    # Convert to DataFrame
    frames = sorted(point_data.keys(), key=int)
    rows = []
    for f_idx in frames:
        details = point_data[f_idx]
        rows.append({
            "frame": int(f_idx),
            "x": details.get("x"),
            "y": details.get("y"),
            "visible": details.get("visible"),
            "action": details.get("action")
        })
    
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} frames")
    
    # Calculate features
    print("Calculating features...")
    df = calculate_optimized_features(df)
    
    # Load model components
    print(f"Loading ML model from: {model_dir}/")
    
    model_path = os.path.join(model_dir, 'trained_xgboost_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    features_path = os.path.join(model_dir, 'feature_columns.pkl')
    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  ✓ Model loaded: {type(model).__name__}")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✓ Scaler loaded: {type(scaler).__name__}")
    
    # Load feature columns
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"  ✓ Feature columns loaded ({len(feature_names)} features)")
    
    # Load label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"  ✓ Label encoder loaded: {type(label_encoder).__name__}")
    
    # Ensure all required features exist
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        print(f"  ⚠ Warning: Missing {len(missing_features)} features, filling with zeros")
        for feat in missing_features:
            df[feat] = 0
    
    # Prepare features in the correct order
    X = df[feature_names].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply scaling
    print("Applying feature scaling...")
    
    if hasattr(scaler, 'transform'):
        X_scaled = scaler.transform(X)
    elif isinstance(scaler, np.ndarray):
        if scaler.shape[0] == 2 and scaler.shape[1] == X.shape[1]:
            mean = scaler[0]
            std = scaler[1]
            X_scaled = (X - mean) / (std + 1e-8)
        else:
            X_scaled = X
    else:
        X_scaled = X
    
    # Get prediction probabilities
    print("Making predictions with probability outputs...")
    
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_scaled)
        print(f"  ✓ Got prediction probabilities")
        print(f"  Probabilities shape: {probas.shape}")
    else:
        print(f"  ⚠ Model doesn't support predict_proba, using predict()")
        raw_preds = model.predict(X_scaled)
        preds_array = np.asarray(raw_preds)
        
        if preds_array.ndim == 2:
            probas = preds_array
        else:
            n_classes = 3
            probas = np.zeros((len(preds_array), n_classes))
            probas[np.arange(len(preds_array)), preds_array.astype(int)] = 1.0
    
    # Apply confidence threshold and get predictions
    predictions = _apply_confidence_threshold(
        probas, label_encoder, AIR_CONFIDENCE_THRESHOLD
    )
    
    df['pred_action'] = predictions
    
    # Create enriched JSON and print summary
    enriched_data = _create_output_json(df)
    _print_summary(df, probas, label_encoder, AIR_CONFIDENCE_THRESHOLD)
    
    return enriched_data



def _supervised_lstm_detection(json_path, model_dir='models'):
    """
    Internal: LSTM detection with confidence thresholding.
    """
    AIR_CONFIDENCE_THRESHOLD = 0.9
    
    print(f"Loading data from: {json_path}")
    print(f"Air confidence threshold: {AIR_CONFIDENCE_THRESHOLD}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        point_data = json.load(f)
    
    # Convert to DataFrame
    frames = sorted(point_data.keys(), key=int)
    rows = []
    for f_idx in frames:
        details = point_data[f_idx]
        rows.append({
            "frame": int(f_idx),
            "x": details.get("x"),
            "y": details.get("y"),
            "visible": details.get("visible"),
            "action": details.get("action")
        })
    
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} frames")
    
    # Calculate features
    print("Calculating features...")
    df = calculate_optimized_features(df)
    
    # Load model components
    print(f"Loading LSTM model from: {model_dir}/")
    
    model_path = os.path.join(model_dir, 'trained_lstm_model.pth')
    config_path = os.path.join(model_dir, 'lstm_model_config.pkl')
    scaler_path = os.path.join(model_dir, 'lstm_scaler.pkl')
    features_path = os.path.join(model_dir, 'lstm_feature_columns.pkl')
    label_encoder_path = os.path.join(model_dir, 'lstm_label_encoder.pkl')
    
    # Load config (usually doesn't have numpy issues)
    try:
        with open(config_path, 'rb') as f:
            model_config = pickle.load(f)
        print(f"  ✓ Model config loaded")
        print(f"    Sequence length: {model_config['sequence_length']}")
        print(f"    Input dim: {model_config['input_dim']}")
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        raise
    
    # Load scaler with compatibility fallback
    scaler = _load_pickle_with_fallback(scaler_path, "Scaler", fallback_type='scaler')
    
    # Load feature columns (usually doesn't have numpy issues)
    try:
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"  ✓ Feature columns loaded ({len(feature_names)} features)")
    except Exception as e:
        print(f"  ✗ Failed to load feature columns: {e}")
        raise
    
    # Load label encoder with compatibility fallback
    label_encoder = _load_pickle_with_fallback(
        label_encoder_path, 
        "Label encoder",
        fallback_type='label_encoder'
    )
    
    # Reconstruct model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    model = AdvancedLSTMClassifier(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout']
    ).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"  ✓ Model weights loaded")
    except RuntimeError as e:
        if "state_dict" in str(e) or "size mismatch" in str(e):
            print(f"  ⚠ Could not load model weights (architecture mismatch): {str(e)[:100]}...")
            print(f"  ⚠ Using model with random initialization (untrained)")
            model.eval()
        else:
            raise
    
    # Prepare features
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        print(f"  ⚠ Warning: Missing {len(missing_features)} features, filling with zeros")
        for feat in missing_features:
            df[feat] = 0
    
    X = df[feature_names].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply scaling with fallback
    print("Applying feature scaling...")
    
    if scaler is None:
        try:
            from sklearn.preprocessing import StandardScaler
            print("  ⚠ Fitting new StandardScaler on current data (fallback)")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            print(f"  ✗ Could not fit fallback scaler: {e}")
            print("  ⚠ Proceeding without scaling")
            X_scaled = X
    elif hasattr(scaler, 'transform'):
        try:
            X_scaled = scaler.transform(X)
            print("  ✓ Applied saved scaler")
        except Exception as e:
            print(f"  ⚠ Scaler transform failed: {e}, fitting new one")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
    else:
        print("  ⚠ Scaler has no transform method, proceeding without scaling")
        X_scaled = X
    
    # Create sequences
    sequence_length = model_config['sequence_length']
    print(f"\nCreating sequences (length={sequence_length})...")
    
    # Pad if necessary
    if len(X_scaled) < sequence_length:
        padding = np.zeros((sequence_length - len(X_scaled), X_scaled.shape[1]))
        X_scaled = np.vstack([padding, X_scaled])
    
    X_seq = []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_seq.append(X_scaled[i:i + sequence_length])
    
    X_seq = np.array(X_seq)
    print(f"  Created {len(X_seq)} sequences")
    
    # Convert to tensor and predict
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    print("Making predictions...")
    with torch.no_grad():
        logits = model(X_tensor)
        probas = torch.softmax(logits, dim=1).cpu().numpy()
    
    print(f"  Got predictions for {len(probas)} frames")
    
    # Apply confidence threshold
    predictions = _apply_confidence_threshold(
        probas, label_encoder, AIR_CONFIDENCE_THRESHOLD
    )
    
    # Map back to original frame indices
    start_frame = sequence_length - 1
    full_predictions = ['air'] * len(df)
    for i, pred in enumerate(predictions):
        full_predictions[start_frame + i] = pred
    
    df['pred_action'] = full_predictions
    
    # Create enriched JSON and print summary
    enriched_data = _create_output_json(df)
    _print_summary(df, probas, label_encoder, AIR_CONFIDENCE_THRESHOLD)
    
    return enriched_data

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _load_pickle_with_fallback(path, name, fallback_type=None):
    """Load pickle file with numpy compatibility fallback."""
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"  ✓ {name} loaded")
        if hasattr(obj, 'classes_'):
            print(f"    Classes: {obj.classes_}")
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"  ⚠ Pickle load failed: {e}. Attempting compatibility fix...")
        
        # More comprehensive numpy compatibility fix
        import importlib
        import sys
        
        # Create a mapping for legacy numpy modules
        try:
            import numpy as np
            
            # Map all numpy._core submodules to numpy.core
            core_module = importlib.import_module('numpy.core')
            multiarray_module = importlib.import_module('numpy.core.multiarray')
            
            legacy_mappings = {
                'numpy._core': core_module,
                'numpy._core.multiarray': multiarray_module,
                'numpy._core.umath': core_module,
                'numpy._core._multiarray_umath': multiarray_module,
            }
            
            for legacy_name, target_module in legacy_mappings.items():
                if legacy_name not in sys.modules:
                    sys.modules[legacy_name] = target_module
            
            # Also ensure numpy.core.multiarray has all necessary attributes
            if not hasattr(multiarray_module, 'scalar'):
                # Try to get scalar from numpy directly
                if hasattr(np, 'ndarray'):
                    multiarray_module.scalar = np.ndarray
                    
            if not hasattr(multiarray_module, '_reconstruct'):
                # Add _reconstruct function if missing
                def _reconstruct(subtype, shape, dtype):
                    return np.ndarray.__new__(subtype, shape, dtype)
                multiarray_module._reconstruct = _reconstruct
            
            print("  ✓ Applied numpy compatibility patches")
            
        except Exception as patch_error:
            print(f"  ⚠ Patching failed: {patch_error}")
        
        # Retry loading with patches applied
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            print(f"  ✓ {name} loaded after compatibility fix")
            if hasattr(obj, 'classes_'):
                print(f"    Classes: {obj.classes_}")
            return obj
        except Exception as e2:
            print(f"  ✗ Failed to load {name} after compatibility fix: {e2}")
            
            # Fallback strategies based on type
            if fallback_type == 'label_encoder':
                print("  ⚠ Creating fallback LabelEncoder with ['air','bounce','hit']")
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.fit(['air', 'bounce', 'hit'])
                return le
            
            elif fallback_type == 'scaler':
                print("  ⚠ Will fit new StandardScaler on data (fallback)")
                return None  # Will be handled in calling function
            
            return None

def _apply_confidence_threshold(probas, label_encoder, threshold):
    """Apply confidence thresholding to probabilities."""
    print(f"\nApplying confidence threshold (air < {threshold})...")
    
    predictions_numeric = np.zeros(len(probas), dtype=int)
    threshold_applied_count = 0
    
    # Determine air class index
    try:
        air_idx = list(label_encoder.classes_).index('air')
    except (ValueError, AttributeError):
        air_idx = 0
    
    for i in range(len(probas)):
        air_prob = probas[i][air_idx]
        
        if air_prob >= threshold:
            predictions_numeric[i] = air_idx
        else:
            non_air_indices = [idx for idx in range(len(probas[i])) if idx != air_idx]
            non_air_probs = [probas[i][idx] for idx in non_air_indices]
            best_non_air_idx = non_air_indices[np.argmax(non_air_probs)]
            predictions_numeric[i] = best_non_air_idx
            threshold_applied_count += 1
    
    print(f"  Threshold applied to {threshold_applied_count}/{len(probas)} frames ({threshold_applied_count/len(probas)*100:.1f}%)")
    
    # Decode predictions
    if hasattr(label_encoder, 'inverse_transform'):
        try:
            predictions = label_encoder.inverse_transform(predictions_numeric)
        except Exception:
            label_map = {0: 'air', 1: 'bounce', 2: 'hit'}
            predictions = [label_map.get(int(p), 'air') for p in predictions_numeric]
    elif hasattr(label_encoder, 'classes_'):
        try:
            predictions = [label_encoder.classes_[int(p)] for p in predictions_numeric]
        except Exception:
            label_map = {0: 'air', 1: 'bounce', 2: 'hit'}
            predictions = [label_map.get(int(p), 'air') for p in predictions_numeric]
    else:
        label_map = {0: 'air', 1: 'bounce', 2: 'hit'}
        predictions = [label_map.get(int(p), 'air') for p in predictions_numeric]
    
    return predictions


def _create_output_json(df):
    """Create enriched JSON output from DataFrame."""
    enriched_data = {}
    for idx, row in df.iterrows():
        frame_num = str(row['frame'])
        enriched_data[frame_num] = {
            "x": float(row['x']),
            "y": float(row['y']),
            "visible": int(row['visible']),
            "action": str(row['pred_action'])
        }
    return enriched_data


def _print_summary(df, probas, label_encoder, threshold):
    """Print prediction summary and ground truth comparison if available."""
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"  Total frames: {len(df)}")
    unique_preds = df['pred_action'].unique()
    for pred in sorted(unique_preds):
        count = (df['pred_action'] == pred).sum()
        percentage = count / len(df) * 100
        print(f"  {pred}: {count} ({percentage:.1f}%)")
    
    # Ground truth comparison if available
    if 'action' in df.columns and not df['action'].isna().all():
        df_eval = df[df['action'].notna()].copy()
        if len(df_eval) > 0:
            from sklearn.metrics import classification_report, confusion_matrix
            
            label_map_gt = {0: 'air', 1: 'bounce', 2: 'hit'}
            
            if pd.api.types.is_numeric_dtype(df_eval['action']):
                df_eval['action'] = df_eval['action'].map(label_map_gt)
            
            df_eval['action'] = df_eval['action'].astype(str)
            df_eval['pred_action'] = df_eval['pred_action'].astype(str)
            
            print("\n" + "="*80)
            print("COMPARISON WITH GROUND TRUTH")
            print("="*80)
            
            try:
                print(classification_report(df_eval['action'], df_eval['pred_action'], zero_division=0))
                
                cm = confusion_matrix(df_eval['action'], df_eval['pred_action'], labels=['air', 'bounce', 'hit'])
                print("\nConfusion Matrix:")
                print("                Predicted")
                print("                air    bounce   hit")
                for i, label in enumerate(['air', 'bounce', 'hit']):
                    print(f"True {label:8s}  {cm[i,0]:6d}  {cm[i,1]:6d}  {cm[i,2]:6d}")
            except Exception as e:
                print(f"Could not generate classification report: {e}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <method> <input_json> [model_dir] [model_type] [output_json]")
        print("  method: 'supervised' or 'unsupervised'")
        print("  model_type: 'lstm' (default) or 'ml' (only for supervised)")
        sys.exit(1)
    
    method = sys.argv[1].lower()
    input_json = sys.argv[2]
    
    if method == 'supervised':
        # Parse arguments for supervised method
        model_dir = 'models'
        model_type = 'lstm'  # Default
        output_json = None
        
        for arg in sys.argv[3:]:
            if arg.endswith('.json'):
                output_json = arg
            elif arg.lower() in ['lstm', 'ml']:
                model_type = arg.lower()
            else:
                model_dir = arg
        
        print(f"Running supervised detection with {model_type.upper()} model...")
        enriched_data = supervised_hit_bounce_detection(input_json, model_dir, model_type)
        
    elif method == 'unsupervised':
        output_json = sys.argv[3] if len(sys.argv) > 3 else None
        enriched_data = unsupervised_hit_bounce_detection(input_json)
        
    else:
        print(f"Error: Unknown method '{method}'. Use 'supervised' or 'unsupervised'")
        sys.exit(1)
    
    # Save output
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(enriched_data, f, indent=2)
        print(f"\nEnriched JSON saved to: {output_json}")
    else:
        print("\nSample output (first 5 frames):")
        sample_keys = sorted(enriched_data.keys(), key=int)[:5]
        for key in sample_keys:
            print(f"  Frame {key}: {enriched_data[key]}")
        print(f"  ... ({len(enriched_data)} total frames)")
    
    print("\nDone!")