
# Streamlit Application Specification: Crypto Price Prediction Challenge

## 1. Application Overview

### Purpose of the Application

This Streamlit application serves as an interactive learning environment for CFA Charterholders and Investment Professionals to pragmatically assess the capabilities and limitations of Artificial Intelligence in the highly volatile cryptocurrency market, specifically Bitcoin price prediction. The core purpose is to move beyond AI hype, demonstrate a real-world workflow from data simulation to model evaluation, and provide a critical assessment of where AI genuinely adds value versus where its predictive power is modest and carries significant risks in speculative assets.

### High-Level Story Flow of the Application

The application guides the user through a structured workflow:

1.  **Introduction & Data Simulation**: The user starts by understanding the challenge and then simulates a realistic, historical Bitcoin dataset, incorporating various market characteristics and features (technical, on-chain, sentiment). This establishes the testing ground.
2.  **Model Building & Training**: The application prepares the simulated data for machine learning and then builds and trains two distinct models: an LSTM for sequence prediction and a Random Forest for feature-based prediction.
3.  **Performance Evaluation & Insights**: The trained models are evaluated using walk-forward validation and a naive trading strategy. Key performance metrics (directional accuracy, Sharpe ratio, total return) are displayed alongside a comparison to a buy-and-hold baseline. Feature importance analysis provides insights into what drives predictions, and confidence distributions illustrate model certainty.
4.  **Honest Assessment & Warning**: Finally, the application presents a critical "Honest Assessment" summary and a "Practitioner Warning" that directly confronts the limitations of AI in short-term crypto prediction, emphasizing its value in other areas like risk management and anomaly detection, aligning with a CFA professional's perspective.

## 2. Code Requirements

### Import Statements

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set random seed for reproducibility at the app level
np.random.seed(42)
tf.random.set_seed(42)

# Import all necessary functions and classes from source.py
from source import (
    generate_btc_data,
    feature_groups, # Globally defined in source.py
    all_features, # Globally defined in source.py
    StandardScaler, # Class
    prepare_sequences,
    build_lstm_model,
    RandomForestClassifier, # Class
    evaluate_crypto_model
    # Note: honest_assessment is a print-based function. Its content will be
    # translated into st.markdown calls in app.py to display properly.
    # Plotting functions are not defined as separate functions in source.py,
    # but rather as code blocks that use plt.show(). Streamlit's st.pyplot()
    # can capture the current active figure if plt.show() is suppressed or
    # if the plotting code is executed and the figure is captured before closure.
)
```

### `st.session_state` Design

The `st.session_state` will be used to maintain the application's state across user interactions and page navigations.

**Initialization (at the top of `app.py` before any page logic):**

```python
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Introduction & Data"
if 'btc_data' not in st.session_state:
    st.session_state.btc_data = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'split_index' not in st.session_state:
    st.session_state.split_index = None
if 'X_train_rf' not in st.session_state:
    st.session_state.X_train_rf = None
if 'y_train_rf' not in st.session_state:
    st.session_state.y_train_rf = None
if 'X_test_rf' not in st.session_state:
    st.session_state.X_test_rf = None
if 'y_test_rf' not in st.session_state:
    st.session_state.y_test_rf = None
if 'X_seq_train' not in st.session_state:
    st.session_state.X_seq_train = None
if 'y_seq_train' not in st.session_state:
    st.session_state.y_seq_train = None
if 'X_seq_test' not in st.session_state:
    st.session_state.X_seq_test = None
if 'y_seq_test' not in st.session_state:
    st.session_state.y_seq_test = None
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'lstm_probs' not in st.session_state:
    st.session_state.lstm_probs = None
if 'lstm_preds' not in st.session_state:
    st.session_state.lstm_preds = None
if 'rf_probs' not in st.session_state:
    st.session_state.rf_probs = None
if 'rf_preds' not in st.session_state:
    st.session_state.rf_preds = None
if 'lstm_eval_metrics' not in st.session_state:
    st.session_state.lstm_eval_metrics = None
if 'rf_eval_metrics' not in st.session_state:
    st.session_state.rf_eval_metrics = None
if 'test_log_returns_lstm' not in st.session_state:
    st.session_state.test_log_returns_lstm = None
if 'test_log_returns_rf' not in st.session_state:
    st.session_state.test_log_returns_rf = None
if 'importance_df' not in st.session_state:
    st.session_state.importance_df = None
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'metrics_evaluated' not in st.session_state:
    st.session_state.metrics_evaluated = False
```

**Keys and Usage:**

*   `current_page`:
    *   **Initialized**: `"Introduction & Data"`
    *   **Updated**: Via `st.sidebar.selectbox` selection.
    *   **Read**: To conditionally render the content of the active page.
*   `btc_data`:
    *   **Initialized**: `None`
    *   **Updated**: When `generate_btc_data()` is called (Page 1).
    *   **Read**: For data preparation (Page 2), plotting (Page 1), and defining `test_log_returns` (Page 3).
*   `scaler`, `X_scaled`, `y`, `split_index`, `X_train_rf`, `y_train_rf`, `X_test_rf`, `y_test_rf`, `X_seq_train`, `y_seq_train`, `X_seq_test`, `y_seq_test`:
    *   **Initialized**: `None`
    *   **Updated**: When data preparation logic is executed (Page 2, after `btc_data` is available).
    *   **Read**: For model training (Page 2) and evaluation (`y` for `y_true` in `evaluate_crypto_model` on Page 3).
*   `lstm_model`, `rf_model`:
    *   **Initialized**: `None`
    *   **Updated**: After `build_lstm_model().fit()` and `RandomForestClassifier().fit()` calls (Page 2).
    *   **Read**: For predictions (Page 2) and feature importance (`rf_model.feature_importances_` on Page 3).
*   `lstm_probs`, `lstm_preds`, `rf_probs`, `rf_preds`:
    *   **Initialized**: `None`
    *   **Updated**: After `lstm_model.predict()` and `rf_model.predict_proba()` calls (Page 2).
    *   **Read**: For evaluation (`y_pred`, `y_prob` in `evaluate_crypto_model` on Page 3) and confidence distribution plot (Page 3).
*   `lstm_eval_metrics`, `rf_eval_metrics`:
    *   **Initialized**: `None`
    *   **Updated**: After `evaluate_crypto_model()` calls (Page 3).
    *   **Read**: For comparison table (Page 3), equity curve plot (Page 3), and honest assessment (Page 4).
*   `test_log_returns_lstm`, `test_log_returns_rf`:
    *   **Initialized**: `None`
    *   **Updated**: When extracted from `btc_data` for evaluation (Page 3).
    *   **Read**: As `actual_log_returns` in `evaluate_crypto_model()` (Page 3).
*   `importance_df`:
    *   **Initialized**: `None`
    *   **Updated**: After calculating Random Forest feature importances (Page 3).
    *   **Read**: For feature importance bar chart (Page 3).
*   `data_generated`, `models_trained`, `metrics_evaluated`:
    *   **Initialized**: `False`
    *   **Updated**: Set to `True` upon successful completion of respective steps.
    *   **Read**: To enable/disable buttons and conditionally display content/warnings for prerequisites.

### Application Structure and Flow

The application will use a sidebar selectbox for navigation, simulating a multi-page experience within a single `app.py` file.

```python
st.set_page_config(layout="wide", page_title="AI in Crypto: A CFA's Perspective")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    st.session_state.current_page = st.selectbox(
        "Go to",
        ["Introduction & Data", "Model Building & Training", "Performance & Insights", "Assessment & Warning"]
    )

# --- Conditional Page Rendering ---

# --- Page 1: Introduction & Data Simulation ---
if st.session_state.current_page == "Introduction & Data":
    st.title("1. Introduction: Navigating Bitcoin's Volatility with AI")
    st.markdown(f"As a **CFA Charterholder and Investment Professional** at a leading asset management firm, I'm constantly evaluating new technologies and data sources to gain an edge in financial markets. Our firm is particularly keen on understanding the capabilities and limitations of Artificial Intelligence in the highly volatile and speculative cryptocurrency market, specifically Bitcoin. There's a lot of hype around AI trading bots, and my role is to cut through that noise to provide a pragmatic assessment of where AI genuinely adds value versus where it presents significant risks.")
    st.markdown(f"This application will guide us through a real-world workflow to simulate historical Bitcoin data, build machine learning models (LSTM and Random Forest) to predict next-day price direction, and critically evaluate their performance. Our goal isn't just to build models, but to understand *why* their predictive power is modest in this noisy environment and what valuable lessons this teaches us about responsible AI deployment in speculative assets. This is a crucial 'humility exercise' to calibrate our expectations and inform risk management strategies.")

    st.header("2. Simulating Realistic Bitcoin Data: Building Our Testbed")
    st.markdown(f"To assess AI's potential, we need a robust dataset that reflects the complex dynamics of Bitcoin. Real historical data can be messy and subject to data availability issues. For controlled experimentation and to encapsulate specific market behaviors, simulating data allows us to incorporate key characteristics like extreme volatility, regime shifts (bull/bear markets), occasional large price jumps, and the influence of novel data sources like on-chain metrics and social sentiment. This step is crucial for an investment professional because it provides a realistic, yet controlled, environment to test hypotheses about predictive modeling without relying on potentially biased real-world data or dealing with data sourcing complexities during the initial research phase.")
    st.markdown(f"The simulation will include:")
    st.markdown(f"*   **Price:** Modeled using Geometric Brownian Motion (GBM) with dynamic drift ($\\mu$) and volatility ($\\sigma$) to simulate bull and bear market regimes, and occasional large jumps.")
    st.markdown(f"*   **Volume:** Reflects trading activity.")
    st.markdown(f"*   **Technical Indicators:** Common metrics derived from price and volume.")
    st.markdown(f"*   **On-chain Metrics:** Unique to crypto, reflecting network activity (e.g., active addresses, hash rate, whale transactions).")
    st.markdown(f"*   **Sentiment Indices:** Capturing market mood from social media.")
    st.markdown(f"*   **Target Variable:** The crucial `next_day_direction` (up or down) for our prediction models.")

    st.markdown(r"The mathematical formulation for the price generation, a modified Geometric Brownian Motion, can be expressed as:$$ S_t = S_{{t-1}} \cdot \exp((\mu - \frac{{1}}{{2}}\sigma^2)\Delta t + \sigma \sqrt{{\Delta t}} Z_t + J_t) $$")
    st.markdown(r"where $S_t$ is the price at time $t$, $\mu$ is the drift, $\sigma$ is the volatility, $\Delta t$ is the time step (1 day), $Z_t$ is a standard normal random variable, and $J_t$ represents occasional jumps. The values of $\mu$ and $\sigma$ will vary based on the market regime (bull or bear).")

    if st.button("Generate Simulated Bitcoin Data", disabled=st.session_state.data_generated):
        with st.spinner("Generating data... This may take a moment."):
            # UI Interaction calls generate_btc_data from source.py
            st.session_state.btc_data = generate_btc_data(n_days=1460, start_date='2021-01-01')
            st.session_state.data_generated = True
            st.success("Bitcoin data simulated successfully!")
            
    if st.session_state.btc_data is not None:
        st.markdown(f"Simulated Bitcoin data generated for {len(st.session_state.btc_data)} days.")
        st.markdown(f"Total features: {len(all_features)} ({len(feature_groups['technical'])} technical, "
                    f"{len(feature_groups['onchain'])} on-chain, {len(feature_groups['sentiment'])} sentiment)")
        st.markdown(f"Base rate (% up days): {st.session_state.btc_data['next_day_direction'].mean():.2%}")
        st.markdown(f"**First 5 rows of simulated Bitcoin data:**")
        st.dataframe(st.session_state.btc_data.head())
        st.markdown(f"**Descriptive statistics:**")
        st.dataframe(st.session_state.btc_data.describe())

        st.subheader("Visualizing Price History with Market Regimes (V1)")
        st.markdown(f"To better understand the simulated market environment, we visualize the Bitcoin price history and visually identify the bull/bear regimes. This helps us confirm if our simulation realistically captures market cycles, which is critical for evaluating model robustness.")
        
        # Plot V1: Bitcoin Price Chart with Market Regimes
        # The plotting logic from source.py's code block for V1 will be invoked.
        # Streamlit's st.pyplot() is expected to capture the Matplotlib figure.
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(st.session_state.btc_data['date'], st.session_state.btc_data['price'], label='Bitcoin Price')

        num_years = len(st.session_state.btc_data) // 365
        for year in range(num_years):
            bull_start = st.session_state.btc_data['date'].iloc[year * 365]
            bull_end = st.session_state.btc_data['date'].iloc[min(len(st.session_state.btc_data)-1, year * 365 + 199)]
            bear_start = st.session_state.btc_data['date'].iloc[min(len(st.session_state.btc_data)-1, year * 365 + 200)]
            bear_end = st.session_state.btc_data['date'].iloc[min(len(st.session_state.btc_data)-1, (year + 1) * 365 - 1)]

            if bull_start <= bull_end:
                ax.axvspan(bull_start, bull_end, color='green', alpha=0.1, label='Bull Regime' if year == 0 else "")
            if bear_start <= bear_end:
                ax.axvspan(bear_start, bear_end, color='red', alpha=0.1, label='Bear Regime' if year == 0 else "")

        ax.set_title('Simulated Bitcoin Price History with Market Regimes')
        ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig) # Render the plot

# --- Page 2: Model Building & Training ---
elif st.session_state.current_page == "Model Building & Training":
    st.title("2. Model Building & Training")

    st.header("3. Preparing Data for Machine Learning Models")
    st.markdown(f"Before feeding our data into machine learning models, proper preparation is essential. For time-series models like LSTMs, we need to transform our data into sequences, as LSTMs learn from the order and patterns over time. For traditional models like Random Forest, we need to ensure features are scaled appropriately, although tree-based models are less sensitive to scaling. Importantly, we must segment our data into training and testing sets *temporally*. This means using past data to train and future data to test, mirroring how an investment professional would apply a model in real-time trading – you can't use future information to predict the past.")
    st.markdown(f"Scaling features helps algorithms converge faster and prevents features with larger numerical ranges from dominating the learning process. For sequence models, the input data for each sample needs to be a 3D array (`(samples, timesteps, features)`). This process ensures that both our LSTM and Random Forest models receive data in the format they expect, enabling fair and robust evaluation.")

    if not st.session_state.data_generated:
        st.warning("Please generate Bitcoin data first on the 'Introduction & Data' page.")
    else:
        # Data Preparation (executed once and stored in session state)
        if st.session_state.X_scaled is None: # Only prepare if not already done
            with st.spinner("Preparing data for models..."):
                # Initialize StandardScaler from source.py
                st.session_state.scaler = StandardScaler()
                st.session_state.X_scaled = st.session_state.scaler.fit_transform(st.session_state.btc_data[all_features])
                st.session_state.y = st.session_state.btc_data['next_day_direction'].values

                # Temporal train/test split: 75% for training, 25% for testing
                st.session_state.split_index = int(len(st.session_state.btc_data) * 0.75)

                # For Random Forest
                st.session_state.X_train_rf = st.session_state.X_scaled[:st.session_state.split_index]
                st.session_state.y_train_rf = st.session_state.y[:st.session_state.split_index]
                st.session_state.X_test_rf = st.session_state.X_scaled[st.session_state.split_index:]
                st.session_state.y_test_rf = st.session_state.y[st.session_state.split_index:]

                # For LSTM, prepare sequences after the split
                SEQUENCE_LENGTH = 30 # Defined in source.py context
                st.session_state.X_seq_train, st.session_state.y_seq_train = prepare_sequences(
                    st.session_state.X_scaled[:st.session_state.split_index],
                    st.session_state.y[:st.session_state.split_index],
                    SEQUENCE_LENGTH
                )
                st.session_state.X_seq_test, st.session_state.y_seq_test = prepare_sequences(
                    st.session_state.X_scaled[st.session_state.split_index:],
                    st.session_state.y[st.session_state.split_index:],
                    SEQUENCE_LENGTH
                )
            st.success("Data prepared!")

        st.markdown(f"Data split: {st.session_state.split_index} days for training, {len(st.session_state.btc_data) - st.session_state.split_index} days for testing.")
        st.markdown(f"Random Forest training features shape: {st.session_state.X_train_rf.shape}, target shape: {st.session_state.y_train_rf.shape}")
        st.markdown(f"Random Forest testing features shape: {st.session_state.X_test_rf.shape}, target shape: {st.session_state.y_test_rf.shape}")
        st.markdown(f"LSTM training sequences shape: {st.session_state.X_seq_train.shape}, target shape: {st.session_state.y_seq_train.shape}")
        st.markdown(f"LSTM testing sequences shape: {st.session_state.X_seq_test.shape}, target shape: {st.session_state.y_seq_test.shape}")
        st.markdown(f"This meticulous preparation ensures no future data 'leaks' into the training set, a critical safeguard for an investment professional to prevent over-optimistic (and ultimately disastrous) backtesting results.")

        st.header("4. LSTM Model for Sequence Prediction")
        st.markdown(f"As an investment professional, I understand that financial time series often exhibit temporal dependencies. The Long Short-Term Memory (LSTM) network, a type of recurrent neural network, is specifically designed to capture these long-term dependencies in sequential data. Unlike simpler models, LSTMs can 'remember' information over extended periods, making them suitable for forecasting where past price movements, on-chain activities, or sentiment shifts might influence future directions. Applying an LSTM allows us to test if the sequential nature of our features provides a predictive edge in Bitcoin. We are building a model that can theoretically learn patterns like 'if Bitcoin has seen X active addresses and Y sentiment over the last 30 days, it tends to move in Z direction.'")
        st.markdown(r"The core idea of an LSTM involves several 'gates' that control the flow of information:")
        st.markdown(r"*   **Forget Gate:** Decides what information to discard from the cell state.$$ f_t = \sigma(W_f \cdot [h_{{t-1}}, x_t] + b_f) $$")
        st.markdown(r"*   **Input Gate:** Decides what new information to store in the cell state.$$ i_t = \sigma(W_i \cdot [h_{{t-1}}, x_t] + b_i) \\ \tilde{{C}}_t = \tanh(W_C \cdot [h_{{t-1}}, x_t] + b_C) $$")
        st.markdown(r"*   **Cell State Update:** Updates the cell state based on the forget and input gates.$$ C_t = f_t * C_{{t-1}} + i_t * \tilde{{C}}_t $$")
        st.markdown(r"*   **Output Gate:** Decides what part of the cell state to output.$$ o_t = \sigma(W_o \cdot [h_{{t-1}}, x_t] + b_o) \\ h_t = o_t * \tanh(C_t) $$")
        st.markdown(r"where $x_t$ is the input at time $t$, $h_{{t-1}}$ is the previous hidden state, $C_{{t-1}}$ is the previous cell state, $W$ and $b$ are weights and biases, and $\sigma$ is the sigmoid activation function.")

        st.header("5. Random Forest for Feature-Based Prediction")
        st.markdown(f"While LSTMs excel at sequential patterns, sometimes the direct, non-linear relationships between features and the target variable are more important than their temporal ordering. A Random Forest classifier is an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes (classification) of the individual trees. This approach is powerful for its ability to handle complex non-linear interactions, automatically perform feature selection, and provide insights into feature importance. For a CFA Charterholder, using a Random Forest provides a benchmark against a non-sequence-aware model and helps us understand which individual features (technical, on-chain, sentiment) are most influential, regardless of their past sequence. This model acts as a direct competitor to the LSTM, allowing us to compare their utility.")
        st.markdown(r"The Random Forest algorithm averages out the predictions of multiple decision trees, reducing overfitting. Each decision tree is built using a random subset of the training data (bagging) and a random subset of features for each split, as illustrated conceptually by:$$ \hat{{y}} = \frac{{1}}{{K}} \sum_{{k=1}}^K h_k(x) $$")
        st.markdown(r"where $\hat{{y}}$ is the final prediction, $K$ is the number of trees, and $h_k(x)$ is the prediction of the $k$-th decision tree.")

        if st.button("Train Models", disabled=st.session_state.models_trained):
            with st.spinner("Training LSTM and Random Forest models... This may take several minutes."):
                # UI Interaction calls build_lstm_model and RandomForestClassifier from source.py
                # LSTM Model
                lstm_model = build_lstm_model(n_features=st.session_state.X_seq_train.shape[2], sequence_length=SEQUENCE_LENGTH)
                lstm_model.fit(
                    st.session_state.X_seq_train, st.session_state.y_seq_train,
                    epochs=50, batch_size=32, validation_split=0.15, verbose=0
                )
                st.session_state.lstm_probs = lstm_model.predict(st.session_state.X_seq_test, verbose=0).flatten()
                st.session_state.lstm_preds = (st.session_state.lstm_probs > 0.5).astype(int)
                st.session_state.lstm_model = lstm_model # Store the trained model

                # Random Forest Model
                rf_model = RandomForestClassifier(
                    n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=42, class_weight='balanced'
                )
                rf_model.fit(st.session_state.X_train_rf, st.session_state.y_train_rf)
                st.session_state.rf_probs = rf_model.predict_proba(st.session_state.X_test_rf)[:, 1]
                st.session_state.rf_preds = (st.session_state.rf_probs > 0.5).astype(int)
                st.session_state.rf_model = rf_model # Store the trained model
                
                st.session_state.models_trained = True
                st.success("Models trained and predictions generated!")
        
        if st.session_state.models_trained:
            st.markdown(f"The LSTM model has been trained on our sequences of historical Bitcoin data. We obtain raw probabilities and then convert them into binary predictions (1 for up, 0 for down) based on a threshold of 0.5. For an investment professional, the choice of this threshold can be critical for constructing a trading strategy. A higher threshold implies higher conviction for a 'buy' signal, potentially reducing false positives but also missed opportunities. This step provides the direct output that our trading strategy will leverage.")
            st.markdown(f"The Random Forest model has been trained and has generated its predictions for the test set. Similar to the LSTM, we obtain probabilities and convert them to binary directional predictions. This provides us with a second, distinct model's output to evaluate. The ability of Random Forest to explain feature importance will be critical later, giving an investment professional insights into *why* a prediction is made, which is often as important as the prediction itself for trust and risk management.")

# --- Page 3: Performance Evaluation & Insights ---
elif st.session_state.current_page == "Performance & Insights":
    st.title("3. Performance Evaluation & Insights")

    st.header("6. Walk-Forward Performance Evaluation & Strategy Simulation")
    st.markdown(f"Evaluating models in financial markets requires more rigor than a simple train-test split. **Walk-forward validation** simulates a real-world trading scenario by iteratively re-training the model on an expanding window of historical data and testing it on the immediate next period. This methodology captures how a model's performance might degrade over time due to market regime shifts or concept drift, providing a more realistic assessment of its viability for an investment professional.")
    st.markdown(f"We'll define a naive trading strategy: 'go long' (buy) Bitcoin if the model predicts an 'up' day, and remain 'flat' (do nothing) if it predicts a 'down' day. This simple strategy allows us to translate model predictions into tangible portfolio performance, using metrics like **directional accuracy**, **Sharpe ratio**, and **total return**. These are standard metrics for an investment professional to assess a strategy's profitability and risk-adjusted returns compared to a **buy-and-hold baseline**.")
    st.markdown(r"Directional accuracy is defined as:$$ \text{{Directional Accuracy}} = \frac{{\text{{Number of Correct Directional Predictions}}}}{{\text{{Total Number of Predictions}}}} $$")
    st.markdown(r"The Sharpe Ratio for a trading strategy is given by:$$ \text{{Sharpe Ratio}} = \frac{{E[R_s - R_f]}}{{\sigma_s}} \approx \frac{{\text{{Mean Daily Strategy Return}}}}{{\text{{Standard Deviation of Daily Strategy Return}}}} \cdot \sqrt{{\text{{Trading Days per Year}}}} $$")
    st.markdown(r"where $R_s$ is the strategy's daily return, $R_f$ is the risk-free rate (assumed 0 for simplicity in this context), and $\sigma_s$ is the standard deviation of strategy daily returns.")

    if not st.session_state.models_trained:
        st.warning("Please train the models first on the 'Model Building & Training' page.")
    else:
        # Evaluate Models (executed once and stored in session state)
        if not st.session_state.metrics_evaluated: # Only evaluate if not already done
            with st.spinner("Evaluating model performance..."):
                SEQUENCE_LENGTH = 30 # Defined in source.py context
                # Determine the actual log returns for the test periods
                # For LSTM, predictions start SEQUENCE_LENGTH days into the test set
                st.session_state.test_log_returns_lstm = st.session_state.btc_data['next_day_return'].iloc[st.session_state.split_index + SEQUENCE_LENGTH:].values
                # For Random Forest, predictions start at the beginning of the test set
                st.session_state.test_log_returns_rf = st.session_state.btc_data['next_day_return'].iloc[st.session_state.split_index:].values

                # UI Interaction calls evaluate_crypto_model from source.py
                st.session_state.lstm_eval_metrics = evaluate_crypto_model(
                    st.session_state.y_seq_test,
                    st.session_state.lstm_preds,
                    st.session_state.lstm_probs,
                    st.session_state.test_log_returns_lstm,
                    'LSTM'
                )
                st.session_state.rf_eval_metrics = evaluate_crypto_model(
                    st.session_state.y_test_rf,
                    st.session_state.rf_preds,
                    st.session_state.rf_probs,
                    st.session_state.test_log_returns_rf,
                    'Random Forest'
                )
                
                st.session_state.metrics_evaluated = True
            st.success("Model performance evaluated!")

        if st.session_state.metrics_evaluated:
            st.markdown(f"The output presents the directional accuracy, Sharpe ratio, and total return for both the LSTM and Random Forest strategies, as well as the buy-and-hold baseline. We can now see how each model performs as a naive trading strategy. Critically, for an investment professional, this immediately highlights whether the models offer any *meaningful* edge over a simple passive strategy, especially in terms of risk-adjusted returns (Sharpe ratio). We often observe that the directional accuracy is barely above 50%, and the Sharpe ratios might not significantly outperform (or even underperform) buy-and-hold, leading us to question the actual profitability after transaction costs.")

            st.subheader("Model Performance Comparison (V2)")
            comparison_data = {
                'Metric': ['Directional Accuracy', 'Sharpe Ratio', 'Total Return'],
                'LSTM': [f"{st.session_state.lstm_eval_metrics['directional_accuracy']:.2%}",
                         f"{st.session_state.lstm_eval_metrics['sharpe_ratio']:.2f}",
                         f"{st.session_state.lstm_eval_metrics['total_return']:.2%}"],
                'Random Forest': [f"{st.session_state.rf_eval_metrics['directional_accuracy']:.2%}",
                                  f"{st.session_state.rf_eval_metrics['sharpe_ratio']:.2f}",
                                  f"{st.session_state.rf_eval_metrics['total_return']:.2%}"],
                'Buy-and-Hold Baseline': ['N/A',
                                          f"{st.session_state.rf_eval_metrics['bh_sharpe_ratio']:.2f}",
                                          f"{st.session_state.rf_eval_metrics['bh_total_return']:.2%}"]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)

            st.header("7. Feature Importance Analysis")
            st.markdown(f"Understanding *why* a model makes certain predictions is as crucial as the predictions themselves for an investment professional. Feature importance analysis, particularly from tree-based models like Random Forest, helps us identify which input variables contribute most to the model's decision-making process. This insight is invaluable for several reasons: it can validate our hypotheses about relevant crypto-specific data (on-chain, sentiment), guide future research into more impactful features, and build trust in the model by showing its reliance on interpretable factors. It answers the question: 'Are our novel data sources (on-chain, sentiment) actually providing predictive signal beyond traditional technicals?'")
            st.markdown(r"The importance of a feature in a Random Forest is typically measured by the average reduction in impurity (Gini importance).")

            # Calculate feature importance if not already done
            if st.session_state.importance_df is None:
                st.session_state.importance_df = pd.Series(
                    st.session_state.rf_model.feature_importances_,
                    index=all_features
                ).sort_values(ascending=False).to_frame(name='importance')
                st.session_state.importance_df['group'] = st.session_state.importance_df.index.map(
                    lambda x: next((group_name for group_name, features in feature_groups.items() if x in features), 'Other')
                )

            st.subheader("Random Forest Feature Importance by Group (V3)")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='importance', y=st.session_state.importance_df.index, hue='group', data=st.session_state.importance_df, dodge=False, palette='viridis', ax=ax)
            ax.set_title('Random Forest Feature Importance by Group')
            ax.set_xlabel('Importance (Gini Importance)')
            ax.set_ylabel('Feature')
            ax.legend(title='Feature Group')
            fig.tight_layout()
            st.pyplot(fig) # Render the plot

            st.subheader("Prediction Confidence Distribution (V4)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.lstm_probs, bins=20, kde=True, color='skyblue', label='LSTM Probabilities', ax=ax)
            sns.histplot(st.session_state.rf_probs, bins=20, kde=True, color='lightcoral', label='Random Forest Probabilities', alpha=0.7, ax=ax)
            ax.set_title('Distribution of Predicted Probabilities (Confidence)')
            ax.set_xlabel('Predicted Probability of Up-Day')
            ax.set_ylabel('Frequency')
            ax.axvline(0.5, color='gray', linestyle='--', label='Decision Threshold (0.5)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig) # Render the plot

            st.subheader("Strategy Equity Curve (V5)")
            fig, ax = plt.subplots(figsize=(14, 7))
            # Align dates for LSTM strategy which starts later
            lstm_dates = st.session_state.btc_data['date'].iloc[st.session_state.split_index + 30:]
            rf_dates = st.session_state.btc_data['date'].iloc[st.session_state.split_index:]

            ax.plot(lstm_dates, st.session_state.lstm_eval_metrics['strategy_cumulative_returns'], label='LSTM Strategy', color='blue')
            ax.plot(rf_dates, st.session_state.rf_eval_metrics['strategy_cumulative_returns'], label='Random Forest Strategy', color='green')
            ax.plot(rf_dates, st.session_state.rf_eval_metrics['bh_cumulative_returns'], label='Buy-and-Hold Baseline', color='red', linestyle='--')
            ax.set_title('Cumulative Returns: AI Strategies vs. Buy-and-Hold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig) # Render the plot


# --- Page 4: Honest Assessment & Warning ---
elif st.session_state.current_page == "Assessment & Warning":
    st.title("4. Honest Assessment & Warning")

    st.header("8. Critical Assessment & Discussion: The Humility Exercise")
    st.markdown(f"After building and evaluating our models, the most critical step for an investment professional is a candid, honest assessment of the results. This is our 'humility exercise.' The modest predictive power (typically 52-56% directional accuracy) we observe in speculative markets like Bitcoin is a profound lesson. It teaches us that AI is not a crystal ball. Understanding *why* short-term prediction is so difficult in crypto, even with sophisticated models and novel data, is paramount for responsible AI deployment and risk management. This section will connect our findings to key financial concepts and provide a pragmatic warning against unrealistic expectations.")

    st.subheader("Why Crypto Prediction Is Hard: A Mathematical Formulation")
    st.markdown(f"*   **Signal-to-noise ratio:** Daily crypto returns often have extremely high volatility ($\\sigma$) relative to their expected mean return ($|\\mu|$). If, for example, daily crypto returns have $\\sigma \\approx 4\\%$ and $|\\mu| \\approx 0.05\\%$, then the signal-to-noise ratio is approximately $\\sigma/|\\mu| \\approx 80$. This means the noise is about 80 times larger than the signal, making it incredibly difficult for any model to reliably extract predictive information. A perfect model would need enormous sample sizes to distinguish signal from noise reliably.")
    st.markdown(f"*   **Non-stationarity:** Crypto markets frequently undergo rapid regime shifts (bull/bear, bubble/crash) far more often than traditional equity markets. A model trained on past bull market data will likely fail catastrophically in a bear market, violating the stationarity assumption underlying many ML models.")
    st.markdown(f"*   **Reflexivity:** Crypto prices are heavily influenced by narratives and sentiment, which are themselves influenced by prices. This creates feedback loops that violate the independent and identically distributed (i.i.d.) assumption crucial for most ML models.")
    st.markdown(f"*   **Expected Directional Accuracy:** For well-designed models in noisy, speculative markets, a realistic directional accuracy is often only 52-56%. Anything below 52% suggests no meaningful signal, while anything above 58% in out-of-sample testing should trigger skepticism and a search for data leakage or overfitting.")
    st.markdown(f"The ultimate output for our firm is not just model performance, but a clear understanding of AI's practical value and its limitations in specific market contexts.")

    if not st.session_state.metrics_evaluated:
        st.warning("Please evaluate model performance first on the 'Performance & Insights' page.")
    else:
        st.subheader("Honest Assessment Summary (V6)")
        st.markdown("---")
        st.markdown("### HONEST ASSESSMENT: AI IN CRYPTO MARKETS (A CFA's Perspective)")
        st.markdown("---")
        st.markdown("#### WHERE AI ADDS LIMITED VALUE (short-term price prediction):")
        st.markdown("- Short-term price prediction: 52-56% directional accuracy - barely better than a coin flip.")
        st.markdown("- Not sufficient for profitable trading after typical transaction costs (0.1-0.3%).")
        st.markdown("- Overfitting risk is extreme in noisy crypto data due to high signal-to-noise ratio (noise >> signal).")
        st.markdown("- Regime shifts invalidate trained models quickly (non-stationarity).")
        st.markdown("#### WHERE AI ADDS GENUINE VALUE (risk management, anomaly detection, sentiment monitoring):")
        st.markdown("- On-chain anomaly detection: Identifying unusual whale movements, exchange inflows/outflows, or network hacks.")
        st.markdown("- Sentiment monitoring: Early warning of narrative shifts, extreme fear/greed sentiment, or coordinated FUD/FOMO campaigns.")
        st.markdown("- Risk management: Volatility forecasting, drawdown prediction, and portfolio optimization for digital assets.")
        st.markdown("- Fraud detection: Identifying wash trading, rug pull schemes, or scam projects.")
        st.markdown("- 24/7 monitoring: AI agents can continuously monitor markets for anomalies, which human analysts cannot do.")
        st.markdown("#### KEY TAKEAWAY FOR CFA PROFESSIONALS:")
        st.markdown("- AI is NOT a crystal ball. In speculative markets with low signal-to-noise ratios, prediction is extremely difficult.")
        st.markdown("- The professional's role is to deploy AI where it adds genuine value (risk management, anomaly detection, sentiment insights).")
        st.markdown("- Resist the temptation to over-rely on return prediction or blindly trust 'AI crypto trading bots'.")
        st.markdown("- Apply CFA Standard V(A) - Diligence and Reasonable Basis: Always exercise due diligence on AI-driven claims as with any investment product.")
        st.markdown("---")

        st.subheader("Practitioner Warning")
        st.warning(f"**Beware the 'AI crypto trading bot' hype.** Numerous products claim AI-powered crypto trading with extraordinary returns. The reality, as demonstrated in this lab, is that **52-56% directional accuracy is the state of the art for well-designed academic models in highly noisy, speculative markets like Bitcoin.** After accounting for typical transaction costs (0.1-0.3% per trade in crypto), even 56% accuracy may not be profitable. Any product claiming 70%+ accuracy in crypto should be treated with **extreme skepticism**. Such claims often involve:")
        st.markdown(f"*   **Survivorship bias:** Showing only winning periods.")
        st.markdown(f"*   **In-sample fitting presented as out-of-sample:** Data leakage or outright fabrication.")
        st.markdown(f"*   **Over-optimization:** Tuning a model so perfectly to past data that it fails in the future.")
        st.warning(f"**Apply the same due diligence standards (CFA Standard V(A)) to AI trading claims as to any other investment product.** Understand the methodology, data sources, and evaluation metrics before committing capital. AI is a powerful tool, but its utility is highly dependent on the signal-to-noise ratio of the domain. In crypto price prediction, the noise often dominates the signal.")

