import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy.stats import poisson
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Global Constants (can be configured externally) ---
# Define feature groups for later analysis
FEATURE_GROUPS = {
    'technical': ['volatility_30d', 'momentum_7d', 'momentum_30d', 'rsi_14', 'volume'],
    'onchain': ['active_addresses', 'hash_rate', 'exchange_inflow', 'whale_transactions'],
    'sentiment': ['twitter_sentiment', 'reddit_mentions', 'fear_greed_index']
}
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
SEQUENCE_LENGTH = 30 # Looking back this many days to predict the next day

def set_global_seeds(seed: int = 42):
    """Sets random seeds for numpy and tensorflow for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def generate_btc_data(n_days: int = 1460, start_date: str = '2021-01-01') -> pd.DataFrame:
    """
    Generates simulated daily Bitcoin data including price, volume,
    technical indicators, on-chain metrics, and sentiment indices.

    Args:
        n_days (int): Number of days to simulate (~4 years = 1460 days).
        start_date (str): Start date for the simulation.

    Returns:
        pd.DataFrame: DataFrame with simulated Bitcoin data.
    """
    dates = pd.date_range(start_date, periods=n_days, freq='D')

    # --- 1. Price: GBM with jumps and regime-dependent drift/vol ---
    price = [40000.0]  # Starting Bitcoin price
    log_returns = [0.0]

    for t in range(1, n_days):
        # Regime shifts: roughly 200 days bull, 165 days bear within a year cycle
        # This is a simplification; real regimes are more complex and unpredictable.
        regime = 'bull' if (t % 365) < 200 else 'bear'

        # Regime-dependent drift and volatility
        mu = 0.0005 if regime == 'bull' else 0.0003  # Daily drift
        sigma = 0.04 if regime == 'bull' else 0.06 # Daily volatility

        # Occasional jumps (e.g., major news events, market shocks)
        jump = np.random.choice([0, 0.05, -0.08], p=[0.97, 0.02, 0.01]) # 3% chance of a jump

        # Calculate daily return based on GBM and jump
        ret = mu + sigma * np.random.randn() + jump
        price.append(price[-1] * (1 + ret))
        log_returns.append(np.log(1 + ret)) # Store log return for later

    df = pd.DataFrame({'date': dates, 'price': price, 'log_return': log_returns})

    # --- 2. Volume ---
    df['volume'] = np.random.lognormal(mean=np.log(2.5e10), sigma=0.5, size=n_days) # High volume base

    # --- 3. Technical Features ---
    df['volatility_30d'] = df['log_return'].rolling(window=30).std() * np.sqrt(365) # Annualized
    df['momentum_7d'] = df['log_return'].rolling(window=7).sum()
    df['momentum_30d'] = df['log_return'].rolling(window=30).sum()

    # Simplified RSI (Relative Strength Index) - influenced by recent returns
    # Actual RSI calculation is more complex, but this captures the sentiment-like aspect
    df['rsi_14'] = 50 + 20 * (df['log_return'].rolling(window=14).sum() / df['log_return'].rolling(window=14).std().replace([np.inf, -np.inf], np.nan).fillna(0.1))
    df['rsi_14'] = np.clip(df['rsi_14'], 0, 100) # Clip RSI to standard range

    # --- 4. On-chain features (unique to crypto) ---
    # Active addresses: proxy for network adoption/usage
    df['active_addresses'] = np.random.lognormal(mean=np.log(5e5), sigma=0.3, size=n_days)
    # Hash rate: proxy for network security, tends to increase over time
    df['hash_rate'] = np.random.lognormal(mean=np.log(2e8), sigma=0.1, size=n_days) * (1 + 0.001 * np.arange(n_days))
    # Exchange inflow: selling pressure indicator
    df['exchange_inflow'] = np.random.lognormal(mean=np.log(1e4), sigma=0.8, size=n_days)
    # Whale transactions: large transactions, Poisson distributed for discrete events
    df['whale_transactions'] = poisson.rvs(mu=15, size=n_days) # Avg 15 whale tx per day

    # --- 5. Sentiment (from social media NLP) ---
    # Twitter sentiment: influenced by recent returns
    df['twitter_sentiment'] = np.clip(np.random.randn(n_days)*0.3 + 0.1 * df['log_return'].shift(1) * 20, -1, 1)
    # Reddit mentions: Poisson distributed
    df['reddit_mentions'] = poisson.rvs(mu=500, size=n_days)
    # Fear/Greed index: influenced by recent returns
    df['fear_greed_index'] = np.clip(50 + 30*np.random.randn(n_days) + 200*df['log_return'].shift(1), 0, 100)

    # --- 6. Target: next-day direction (1=up, 0=down) ---
    df['next_day_return'] = df['log_return'].shift(-1)
    df['next_day_direction'] = (df['next_day_return'] > 0).astype(int)

    df = df.dropna().reset_index(drop=True) # Drop NaNs introduced by shifts/rolling

    return df

def plot_price_history(df: pd.DataFrame) -> plt.Figure:
    """
    Plots the simulated Bitcoin price history with market regime annotations.

    Args:
        df (pd.DataFrame): DataFrame containing 'date' and 'price' columns.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['date'], df['price'], label='Bitcoin Price')

    # Annotate approximate regime shifts (hardcoded logic from data generation)
    num_years = len(df) // 365
    for year in range(num_years):
        bull_start = df['date'].iloc[year * 365]
        bull_end = df['date'].iloc[min(len(df)-1, year * 365 + 199)]
        bear_start = df['date'].iloc[min(len(df)-1, year * 365 + 200)]
        bear_end = df['date'].iloc[min(len(df)-1, (year + 1) * 365 - 1)]

        if bull_start <= bull_end:
            ax.axvspan(bull_start, bull_end, color='green', alpha=0.1, label='Bull Regime' if year == 0 else "")
        if bear_start <= bear_end:
            ax.axvspan(bear_start, bear_end, color='red', alpha=0.1, label='Bear Regime' if year == 0 else "")

    ax.set_title('Simulated Bitcoin Price History with Market Regimes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def prepare_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares data into sequences for LSTM model.

    Args:
        X (np.array): Scaled feature data.
        y (np.array): Target variable data.
        sequence_length (int): Number of past timesteps to consider.

    Returns:
        tuple: (Xs, ys) where Xs are 3D sequences, ys are corresponding targets.
    """
    Xs, ys = [], []
    for i in range(sequence_length, len(X)):
        Xs.append(X[i-sequence_length:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def preprocess_data(
    btc_data: pd.DataFrame,
    all_features: list,
    sequence_length: int = SEQUENCE_LENGTH,
    test_size: float = 0.25
) -> tuple:
    """
    Scales features and prepares data for both Random Forest and LSTM models.

    Args:
        btc_data (pd.DataFrame): The raw simulated Bitcoin data.
        all_features (list): List of feature column names.
        sequence_length (int): Length of sequences for LSTM.
        test_size (float): Proportion of data to use for testing.

    Returns:
        tuple: A tuple containing:
            - scaler (StandardScaler): The fitted scaler.
            - X_train_rf (np.array): Training features for Random Forest.
            - y_train_rf (np.array): Training target for Random Forest.
            - X_test_rf (np.array): Testing features for Random Forest.
            - y_test_rf (np.array): Testing target for Random Forest.
            - X_seq_train (np.array): Training sequences for LSTM.
            - y_seq_train (np.array): Training target for LSTM.
            - X_seq_test (np.array): Testing sequences for LSTM.
            - y_seq_test (np.array): Testing target for LSTM.
            - split_index (int): The index where the temporal split occurred.
            - test_log_returns_rf (np.array): Actual log returns for RF test period.
            - test_log_returns_lstm (np.array): Actual log returns for LSTM test period.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(btc_data[all_features])
    y = btc_data['next_day_direction'].values

    # Temporal train/test split
    split_index = int(len(btc_data) * (1 - test_size))

    # For Random Forest, which doesn't require sequences
    X_train_rf = X_scaled[:split_index]
    y_train_rf = y[:split_index]
    X_test_rf = X_scaled[split_index:]
    y_test_rf = y[split_index:]

    # For LSTM, prepare sequences after the split to avoid data leakage
    # Note: LSTM sequences start 'SEQUENCE_LENGTH' days after the split index
    X_seq_train, y_seq_train = prepare_sequences(X_scaled[:split_index], y[:split_index], sequence_length)
    X_seq_test, y_seq_test = prepare_sequences(X_scaled[split_index:], y[split_index:], sequence_length)

    # Actual log returns for evaluation, adjusted for respective test data lengths
    test_log_returns_rf = btc_data['next_day_return'].iloc[split_index:].values
    test_log_returns_lstm = btc_data['next_day_return'].iloc[split_index + sequence_length:].values

    print(f"Data split: {split_index} days for training, {len(btc_data) - split_index} days for testing.")
    print(f"Random Forest training features shape: {X_train_rf.shape}, target shape: {y_train_rf.shape}")
    print(f"Random Forest testing features shape: {X_test_rf.shape}, target shape: {y_test_rf.shape}")
    print(f"LSTM training sequences shape: {X_seq_train.shape}, target shape: {y_seq_train.shape}")
    print(f"LSTM testing sequences shape: {X_seq_test.shape}, target shape: {y_seq_test.shape}")

    return (scaler, X_train_rf, y_train_rf, X_test_rf, y_test_rf,
            X_seq_train, y_seq_train, X_seq_test, y_seq_test,
            split_index, test_log_returns_rf, test_log_returns_lstm)

def build_lstm_model(n_features: int, sequence_length: int = SEQUENCE_LENGTH) -> tf.keras.Model:
    """
    Builds a Sequential LSTM model for binary classification.

    Args:
        n_features (int): Number of input features per timestep.
        sequence_length (int): Length of the input sequences.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(units=64, input_shape=(sequence_length, n_features), return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(units=32, activation='relu'),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid') # Binary classification output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_crypto_model(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    actual_log_returns: np.ndarray, label: str
) -> dict:
    """
    Evaluates directional prediction for crypto and simulates a naive trading strategy.

    Args:
        y_true (np.array): Actual next day directions (0 or 1).
        y_pred (np.array): Predicted next day directions (0 or 1).
        y_prob (np.array): Predicted probabilities for next day up (class 1).
        actual_log_returns (np.array): Actual log returns for the test period.
        label (str): Label for the model (e.g., 'LSTM', 'Random Forest').

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    metrics = {}

    # Directional Accuracy
    metrics['directional_accuracy'] = accuracy_score(y_true, y_pred)

    # Naive Trading Strategy: Go long on 'up' predictions, flat otherwise
    # Positions: 1 for long, 0 for flat
    positions = np.where(y_pred == 1, 1, 0)

    # Strategy returns: positions * actual_returns
    # Assuming actual_log_returns are already next-day log returns
    strategy_log_returns = positions * actual_log_returns

    # Convert log returns to simple returns for portfolio calculations
    strategy_simple_returns = np.exp(strategy_log_returns) - 1

    # Cumulative product for total return
    metrics['total_return'] = np.prod(1 + strategy_simple_returns) - 1

    # Sharpe Ratio (annualized)
    # Using 0.001 as minimum std dev to avoid division by zero for flat returns
    daily_avg_return = np.mean(strategy_simple_returns)
    daily_std_dev = np.std(strategy_simple_returns)
    metrics['sharpe_ratio'] = (daily_avg_return / max(daily_std_dev, 1e-6)) * np.sqrt(365)

    # Buy-and-hold baseline
    bh_simple_returns = np.exp(actual_log_returns) - 1
    metrics['bh_total_return'] = np.prod(1 + bh_simple_returns) - 1
    bh_daily_avg_return = np.mean(bh_simple_returns)
    bh_daily_std_dev = np.std(bh_simple_returns)
    metrics['bh_sharpe_ratio'] = (bh_daily_avg_return / max(bh_daily_std_dev, 1e-6)) * np.sqrt(365)

    # Store strategy and buy-and-hold cumulative returns for equity curve
    metrics['strategy_cumulative_returns'] = np.cumprod(1 + strategy_simple_returns) - 1
    metrics['bh_cumulative_returns'] = np.cumprod(1 + bh_simple_returns) - 1

    print(f"\n--- {label} EVALUATION ---")
    print("=" * 50)
    print(f"Directional accuracy: {metrics['directional_accuracy']:.2%} (random = 50%)")
    print(f"Strategy Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Strategy total return: {metrics['total_return']:.2%}")
    print(f"Buy-and-hold Sharpe: {metrics['bh_sharpe_ratio']:.2f}")
    print(f"Buy-and-hold total return: {metrics['bh_total_return']:.2%}")
    print(f"AI edge over B&H (Sharpe): {metrics['sharpe_ratio'] - metrics['bh_sharpe_ratio']:.2f}")

    return metrics

def train_and_evaluate_models(
    X_train_rf: np.ndarray, y_train_rf: np.ndarray,
    X_test_rf: np.ndarray, y_test_rf: np.ndarray,
    test_log_returns_rf: np.ndarray,
    X_seq_train: np.ndarray, y_seq_train: np.ndarray,
    X_seq_test: np.ndarray, y_seq_test: np.ndarray,
    test_log_returns_lstm: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH
) -> tuple:
    """
    Trains and evaluates LSTM and Random Forest models.

    Args:
        X_train_rf, y_train_rf: Training data for Random Forest.
        X_test_rf, y_test_rf: Testing data for Random Forest.
        test_log_returns_rf: Actual log returns for RF test period.
        X_seq_train, y_seq_train: Training sequences for LSTM.
        X_seq_test, y_seq_test: Testing sequences for LSTM.
        test_log_returns_lstm: Actual log returns for LSTM test period.
        sequence_length (int): Length of sequences for LSTM.

    Returns:
        tuple: A tuple containing:
            - lstm_model (tf.keras.Model): The trained LSTM model.
            - rf_model (sklearn.ensemble.RandomForestClassifier): The trained Random Forest model.
            - lstm_probs (np.array): LSTM predicted probabilities.
            - lstm_preds (np.array): LSTM binary predictions.
            - rf_probs (np.array): Random Forest predicted probabilities.
            - rf_preds (np.array): Random Forest binary predictions.
            - lstm_eval_metrics (dict): Evaluation metrics for LSTM.
            - rf_eval_metrics (dict): Evaluation metrics for Random Forest.
    """
    # Build and train the LSTM model
    n_features_lstm = X_seq_train.shape[2]
    lstm_model = build_lstm_model(n_features=n_features_lstm, sequence_length=sequence_length)

    print("Training LSTM model...")
    history = lstm_model.fit(
        X_seq_train, y_seq_train,
        epochs=50,
        batch_size=32,
        validation_split=0.15,
        verbose=0 # Set to 1 to see training progress
    )
    print("LSTM model training complete.")

    # Make predictions on the test set
    lstm_probs = lstm_model.predict(X_seq_test, verbose=0).flatten()
    lstm_preds = (lstm_probs > 0.5).astype(int) # Convert probabilities to binary predictions

    # Build and train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,      # Number of trees in the forest
        max_depth=6,           # Maximum depth of the tree to prevent overfitting
        min_samples_leaf=20,   # Minimum number of samples required to be at a leaf node
        random_state=42,       # For reproducibility
        class_weight='balanced' # Handle potential class imbalance
    )

    print("\nTraining Random Forest model...")
    rf_model.fit(X_train_rf, y_train_rf)
    print("Random Forest model training complete.")

    # Make predictions on the test set
    rf_probs = rf_model.predict_proba(X_test_rf)[:, 1] # Probability of class 1 (up)
    rf_preds = (rf_probs > 0.5).astype(int) # Convert probabilities to binary predictions

    # Evaluate models
    lstm_eval_metrics = evaluate_crypto_model(y_seq_test, lstm_preds, lstm_probs, test_log_returns_lstm, 'LSTM')
    rf_eval_metrics = evaluate_crypto_model(y_test_rf, rf_preds, rf_probs, test_log_returns_rf, 'Random Forest')

    return (lstm_model, rf_model, lstm_probs, lstm_preds, rf_probs, rf_preds,
            lstm_eval_metrics, rf_eval_metrics)

def analyze_feature_importance(
    rf_model: RandomForestClassifier,
    all_features: list,
    feature_groups: dict = None
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Analyzes and visualizes feature importance for the Random Forest model.

    Args:
        rf_model (sklearn.ensemble.RandomForestClassifier): The trained Random Forest model.
        all_features (list): List of all feature names.
        feature_groups (dict, optional): Dictionary mapping group names to lists of features.
                                        Defaults to `FEATURE_GROUPS`.

    Returns:
        tuple: A tuple containing:
            - importance_df (pd.DataFrame): DataFrame with feature importances.
            - fig (plt.Figure): The generated matplotlib figure.
    """
    feature_groups = feature_groups if feature_groups is not None else FEATURE_GROUPS
    importance_df = pd.Series(rf_model.feature_importances_, index=all_features).sort_values(ascending=False).to_frame(name='importance')

    # Add feature group for visualization
    importance_df['group'] = importance_df.index.map(lambda x: next((group_name for group_name, features in feature_groups.items() if x in features), 'Other'))

    print("\n--- FEATURE IMPORTANCE (Random Forest) ---")
    print("=" * 50)
    for feat, row in importance_df.iterrows():
        group_label = row['group']
        imp = row['importance']
        bar = '#' * int(imp * 100) # Simple text-based bar
        print(f"[{group_label.upper():>9s}] {feat:<22s} {imp:.4f} {bar}")

    # Calculate group-level importance
    group_importance = importance_df.groupby('group')['importance'].sum().sort_values(ascending=False)
    print("\n--- GROUP-LEVEL IMPORTANCE ---")
    print("=" * 50)
    for group_name, total_imp in group_importance.items():
        print(f"{group_name.upper()} group total importance: {total_imp:.3f}")

    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y=importance_df.index, hue='group', data=importance_df, dodge=False, palette='viridis', ax=ax)
    ax.set_title('Random Forest Feature Importance by Group')
    ax.set_xlabel('Importance (Gini Importance)')
    ax.set_ylabel('Feature')
    ax.legend(title='Feature Group')
    fig.tight_layout()
    return importance_df, fig

def visualize_results(
    lstm_probs: np.ndarray, rf_probs: np.ndarray,
    lstm_eval_metrics: dict, rf_eval_metrics: dict,
    btc_data: pd.DataFrame, split_index: int, sequence_length: int = SEQUENCE_LENGTH
) -> tuple[pd.DataFrame, plt.Figure, plt.Figure]:
    """
    Generates visualizations of model performance and a comparison table.

    Args:
        lstm_probs (np.array): Predicted probabilities from LSTM.
        rf_probs (np.array): Predicted probabilities from Random Forest.
        lstm_eval_metrics (dict): Evaluation metrics for LSTM.
        rf_eval_metrics (dict): Evaluation metrics for Random Forest.
        btc_data (pd.DataFrame): Original DataFrame (for dates).
        split_index (int): Index where train/test split occurred for Random Forest.
        sequence_length (int): Length of sequences for LSTM.

    Returns:
        tuple: A tuple containing:
            - comparison_df (pd.DataFrame): DataFrame comparing model performance.
            - fig_confidence (plt.Figure): Figure for prediction confidence distribution.
            - fig_equity (plt.Figure): Figure for equity curves.
    """
    # Visualize Prediction Confidence Distribution
    fig_confidence, ax_confidence = plt.subplots(figsize=(10, 6))
    sns.histplot(lstm_probs, bins=20, kde=True, color='skyblue', label='LSTM Probabilities', ax=ax_confidence)
    sns.histplot(rf_probs, bins=20, kde=True, color='lightcoral', label='Random Forest Probabilities', alpha=0.7, ax=ax_confidence)
    ax_confidence.set_title('Distribution of Predicted Probabilities (Confidence)')
    ax_confidence.set_xlabel('Predicted Probability of Up-Day')
    ax_confidence.set_ylabel('Frequency')
    ax_confidence.axvline(0.5, color='gray', linestyle='--', label='Decision Threshold (0.5)')
    ax_confidence.legend()
    ax_confidence.grid(True, linestyle='--', alpha=0.6)
    fig_confidence.tight_layout()

    # Visualize Equity Curve
    fig_equity, ax_equity = plt.subplots(figsize=(14, 7))
    # LSTM strategy dates start after split_index + sequence_length
    ax_equity.plot(btc_data['date'].iloc[split_index + sequence_length:], lstm_eval_metrics['strategy_cumulative_returns'], label='LSTM Strategy', color='blue')
    # Random Forest strategy and Buy-and-Hold dates start after split_index
    ax_equity.plot(btc_data['date'].iloc[split_index:], rf_eval_metrics['strategy_cumulative_returns'], label='Random Forest Strategy', color='green')
    ax_equity.plot(btc_data['date'].iloc[split_index:], rf_eval_metrics['bh_cumulative_returns'], label='Buy-and-Hold Baseline', color='red', linestyle='--')
    ax_equity.set_title('Cumulative Returns: AI Strategies vs. Buy-and-Hold')
    ax_equity.set_xlabel('Date')
    ax_equity.set_ylabel('Cumulative Return')
    ax_equity.legend()
    ax_equity.grid(True)
    fig_equity.tight_layout()

    # Comparison Table
    comparison_data = {
        'Metric': ['Directional Accuracy', 'Sharpe Ratio', 'Total Return'],
        'LSTM': [f"{lstm_eval_metrics['directional_accuracy']:.2%}",
                 f"{lstm_eval_metrics['sharpe_ratio']:.2f}",
                 f"{lstm_eval_metrics['total_return']:.2%}"],
        'Random Forest': [f"{rf_eval_metrics['directional_accuracy']:.2%}",
                          f"{rf_eval_metrics['sharpe_ratio']:.2f}",
                          f"{rf_eval_metrics['total_return']:.2%}"],
        'Buy-and-Hold Baseline': ['N/A',
                                  f"{rf_eval_metrics['bh_sharpe_ratio']:.2f}",
                                  f"{rf_eval_metrics['bh_total_return']:.2%}"]
    }
    comparison_df = pd.DataFrame(comparison_data)

    print("\n--- Model Performance Comparison ---")
    print(comparison_df.to_markdown(index=False))

    return comparison_df, fig_confidence, fig_equity

def honest_assessment() -> str:
    """
    Provides an honest assessment of AI in crypto markets from a CFA's perspective.

    Returns:
        str: A multi-line string containing the assessment.
    """
    assessment_lines = []
    assessment_lines.append("=" * 70)
    assessment_lines.append("HONEST ASSESSMENT: AI IN CRYPTO MARKETS (A CFA's Perspective)")
    assessment_lines.append("=" * 70)
    assessment_lines.append("\nWHERE AI ADDS LIMITED VALUE (short-term price prediction):")
    assessment_lines.append(" - Short-term price prediction: 52-56% directional accuracy - barely better than a coin flip.")
    assessment_lines.append(" - Not sufficient for profitable trading after typical transaction costs (0.1-0.3%).")
    assessment_lines.append(" - Overfitting risk is extreme in noisy crypto data due to high signal-to-noise ratio (noise >> signal).")
    assessment_lines.append(" - Regime shifts invalidate trained models quickly (non-stationarity).")
    assessment_lines.append("\nWHERE AI ADDS GENUINE VALUE (risk management, anomaly detection, sentiment monitoring):")
    assessment_lines.append(" - On-chain anomaly detection: Identifying unusual whale movements, exchange inflows/outflows, or network hacks.")
    assessment_lines.append(" - Sentiment monitoring: Early warning of narrative shifts, extreme fear/greed sentiment, or coordinated FUD/FOMO campaigns.")
    assessment_lines.append(" - Risk management: Volatility forecasting, drawdown prediction, and portfolio optimization for digital assets.")
    assessment_lines.append(" - Fraud detection: Identifying wash trading, rug pull schemes, or scam projects.")
    assessment_lines.append(" - 24/7 monitoring: AI agents can continuously monitor markets for anomalies, which human analysts cannot do.")
    assessment_lines.append("\nKEY TAKEAWAY FOR CFA PROFESSIONALS:")
    assessment_lines.append(" - AI is NOT a crystal ball. In speculative markets with low signal-to-noise ratios, prediction is extremely difficult.")
    assessment_lines.append(" - The professional's role is to deploy AI where it adds genuine value (risk management, anomaly detection, sentiment insights).")
    assessment_lines.append(" - Resist the temptation to over-rely on return prediction or blindly trust 'AI crypto trading bots'.")
    assessment_lines.append(" - Apply CFA Standard V(A) - Diligence and Reasonable Basis: Always exercise due diligence on AI-driven claims as with any investment product.")
    assessment_lines.append("=" * 70)
    return "\n".join(assessment_lines)

def run_crypto_prediction_pipeline(
    n_days: int = 1460,
    start_date: str = '2021-01-01',
    sequence_length: int = SEQUENCE_LENGTH,
    test_size: float = 0.25,
    random_seed: int = 42
) -> dict:
    """
    Main function to run the entire cryptocurrency prediction and analysis pipeline.

    Args:
        n_days (int): Number of days for simulated data.
        start_date (str): Start date for data generation.
        sequence_length (int): Look-back period for LSTM.
        test_size (float): Proportion of data for testing.
        random_seed (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing various results, metrics, and plots.
              Plots are returned as matplotlib Figure objects.
    """
    set_global_seeds(random_seed)

    print("--- Starting Cryptocurrency Prediction Pipeline ---")

    # 1. Data Generation
    print("\n1. Generating simulated Bitcoin data...")
    btc_data = generate_btc_data(n_days=n_days, start_date=start_date)
    print(f"Simulated Bitcoin data generated for {len(btc_data)} days.")
    print(f"Total features: {len(ALL_FEATURES)} ({len(FEATURE_GROUPS['technical'])} technical, "
          f"{len(FEATURE_GROUPS['onchain'])} on-chain, {len(FEATURE_GROUPS['sentiment'])} sentiment)")
    print(f"Base rate (% up days): {btc_data['next_day_direction'].mean():.2%}")
    print("\nFirst 5 rows of simulated Bitcoin data:")
    print(btc_data.head())
    print("\nDescriptive statistics:")
    print(btc_data.describe())

    # Plot price history
    price_history_fig = plot_price_history(btc_data)

    # 2. Data Preprocessing
    print("\n2. Preprocessing data and preparing sequences...")
    (scaler, X_train_rf, y_train_rf, X_test_rf, y_test_rf,
     X_seq_train, y_seq_train, X_seq_test, y_seq_test,
     split_index, test_log_returns_rf, test_log_returns_lstm) = preprocess_data(
        btc_data, ALL_FEATURES, sequence_length, test_size
    )

    # 3. Model Training and Evaluation
    print("\n3. Training and evaluating models (LSTM and Random Forest)...")
    (lstm_model, rf_model, lstm_probs, lstm_preds, rf_probs, rf_preds,
     lstm_eval_metrics, rf_eval_metrics) = train_and_evaluate_models(
        X_train_rf, y_train_rf, X_test_rf, y_test_rf, test_log_returns_rf,
        X_seq_train, y_seq_train, X_seq_test, y_seq_test, test_log_returns_lstm,
        sequence_length
    )

    # 4. Feature Importance Analysis
    print("\n4. Analyzing Random Forest Feature Importance...")
    importance_df, feature_importance_fig = analyze_feature_importance(rf_model, ALL_FEATURES, FEATURE_GROUPS)

    # 5. Visualize Results and Comparison
    print("\n5. Visualizing model performance...")
    comparison_df, confidence_fig, equity_curve_fig = visualize_results(
        lstm_probs, rf_probs, lstm_eval_metrics, rf_eval_metrics,
        btc_data, split_index, sequence_length
    )

    # 6. Honest Assessment
    print("\n6. Providing an honest assessment of AI in crypto markets...")
    assessment_text = honest_assessment()
    print(assessment_text) # Print the assessment for console output

    print("\n--- Pipeline Completed ---")

    return {
        "btc_data": btc_data,
        "scaler": scaler,
        "lstm_model": lstm_model,
        "rf_model": rf_model,
        "lstm_eval_metrics": lstm_eval_metrics,
        "rf_eval_metrics": rf_eval_metrics,
        "importance_df": importance_df,
        "comparison_df": comparison_df,
        "price_history_plot": price_history_fig,
        "feature_importance_plot": feature_importance_fig,
        "confidence_distribution_plot": confidence_fig,
        "equity_curve_plot": equity_curve_fig,
        "assessment_text": assessment_text
    }

# Example of how this might be used in an app.py:
if __name__ == '__main__':
    # To run the pipeline and get all results
    results = run_crypto_prediction_pipeline(n_days=1000, test_size=0.2)

    # You can now access the results, e.g.:
    # print(results['comparison_df'])
    # print(results['assessment_text'])

    # To display the plots (in a script, or save them in an app)
    # plt.show()
    # Note: If running this script directly, the print statements from
    # evaluation functions and the assessment will be shown in the console.
    # The figures will pop up, and you'll need to close them to proceed
    # if using plt.show() at the end. For web apps, these figures would be
    # embedded or saved.
    print("\nAll plots generated. Displaying them now (close plots to exit script)...")
    plt.show() # This will display all figures that were created and returned.
