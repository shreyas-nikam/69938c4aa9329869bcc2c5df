from source import (
    generate_btc_data,
    feature_groups,  # alias provided in source_generated.py
    all_features,    # alias provided in source_generated.py
    SEQUENCE_LENGTH,
    StandardScaler,
    prepare_sequences,
    build_lstm_model,
    RandomForestClassifier,
    evaluate_crypto_model,
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
import os

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed for reproducibility at the app level
np.random.seed(42)
tf.random.set_seed(42)

# Import all necessary functions and classes from source.py (generated compatibility wrapper)

st.set_page_config(
    layout="wide", page_title="QuLab: Lab 55 - AI in Crypto: A CFA's Perspective")
st.title("QuLab: Lab 55 - AI in Crypto: A CFA's Perspective - An Honest Assessment")
st.divider()
# ----------------------------
# Sidebar: Navigation + Controls
# ----------------------------
with st.sidebar:
    st.image("https://www.quantuniversity.com/assets/img/logo5.jpg")

    st.divider()
    st.header("Navigation")
    page = st.selectbox(
        "Go to",
        ["Introduction & Data", "Model Building & Training",
            "Performance & Insights", "Assessment & Warning"],
        key="nav_selectbox",
    )

    st.divider()
    st.subheader("Controls")
    if st.button("Reset Lab State"):
        keep_page = st.session_state.get(
            "nav_selectbox", "Introduction & Data")
        for k in list(st.session_state.keys()):
            if k not in {"nav_selectbox"}:
                del st.session_state[k]
        st.session_state["nav_selectbox"] = keep_page
        st.rerun()

    st.caption("Tip: Run the workflow in order (1 → 4).")

# ----------------------------
# Session State Initialization
# ----------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Introduction & Data"

# Data + prep
if "btc_data" not in st.session_state:
    st.session_state.btc_data = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "y" not in st.session_state:
    st.session_state.y = None
if "split_index" not in st.session_state:
    st.session_state.split_index = None

if "X_train_rf" not in st.session_state:
    st.session_state.X_train_rf = None
if "y_train_rf" not in st.session_state:
    st.session_state.y_train_rf = None
if "X_test_rf" not in st.session_state:
    st.session_state.X_test_rf = None
if "y_test_rf" not in st.session_state:
    st.session_state.y_test_rf = None

if "X_seq_train" not in st.session_state:
    st.session_state.X_seq_train = None
if "y_seq_train" not in st.session_state:
    st.session_state.y_seq_train = None
if "X_seq_test" not in st.session_state:
    st.session_state.X_seq_test = None
if "y_seq_test" not in st.session_state:
    st.session_state.y_seq_test = None

# Models + predictions
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = None
if "rf_model" not in st.session_state:
    st.session_state.rf_model = None

if "lstm_probs" not in st.session_state:
    st.session_state.lstm_probs = None
if "lstm_preds" not in st.session_state:
    st.session_state.lstm_preds = None
if "rf_probs" not in st.session_state:
    st.session_state.rf_probs = None
if "rf_preds" not in st.session_state:
    st.session_state.rf_preds = None

# Evaluation artifacts
if "lstm_eval_metrics" not in st.session_state:
    st.session_state.lstm_eval_metrics = None
if "rf_eval_metrics" not in st.session_state:
    st.session_state.rf_eval_metrics = None
if "test_log_returns_lstm" not in st.session_state:
    st.session_state.test_log_returns_lstm = None
if "test_log_returns_rf" not in st.session_state:
    st.session_state.test_log_returns_rf = None
if "importance_df" not in st.session_state:
    st.session_state.importance_df = None

# Flags
if "data_generated" not in st.session_state:
    st.session_state.data_generated = False
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "metrics_evaluated" not in st.session_state:
    st.session_state.metrics_evaluated = False

# Sync current page with nav
st.session_state.current_page = page

# ----------------------------
# Helpers
# ----------------------------


def _footer():
    st.divider()
    st.write("© 2025 QuantUniversity. All Rights Reserved.")
    st.caption(
        "The purpose of this demonstration is solely for educational use and illustration. "
        "To access the full legal documentation, please visit the official QuantUniversity documentation. "
        "Any reproduction of this demonstration requires prior written consent from QuantUniversity."
    )
    st.caption(
        "This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, "
        "which may contain inaccuracies or errors."
    )


def _plot_price_with_regimes(btc_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(btc_df["date"], btc_df["price"], label="Bitcoin Price")

    num_years = len(btc_df) // 365
    for year in range(num_years):
        bull_start = btc_df["date"].iloc[year * 365]
        bull_end = btc_df["date"].iloc[min(len(btc_df) - 1, year * 365 + 199)]
        bear_start = btc_df["date"].iloc[min(
            len(btc_df) - 1, year * 365 + 200)]
        bear_end = btc_df["date"].iloc[min(
            len(btc_df) - 1, (year + 1) * 365 - 1)]

        if bull_start <= bull_end:
            ax.axvspan(bull_start, bull_end, color="green", alpha=0.1,
                       label="Bull Regime" if year == 0 else "")
        if bear_start <= bear_end:
            ax.axvspan(bear_start, bear_end, color="red", alpha=0.1,
                       label="Bear Regime" if year == 0 else "")

    ax.set_title("Simulated Bitcoin Price History with Market Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def _metric_card(label: str, value: str, help_text: str):
    st.metric(label=label, value=value, help=help_text)


# ----------------------------
# Page 1: Introduction & Data
# ----------------------------
if st.session_state.current_page == "Introduction & Data":

    st.markdown(
        "There's a lot of hype around AI trading bots, and the professional obligation is to cut through that noise. "
        "Our goal isn't just to build models, but to understand *why* their predictive power is typically modest in this noisy environment "
        "and what valuable lessons this teaches us about responsible AI deployment in speculative assets. "
        "This is a crucial **humility exercise** to calibrate expectations and inform risk management."
    )

    st.markdown(
        "As a CFA Charterholder and Investment Professional at a leading asset management firm, "
        "I'm constantly evaluating new technologies and data sources to gain an edge in financial markets. "
        "Our firm is particularly keen on understanding the capabilities and limitations of Artificial Intelligence "
        "in the highly volatile and speculative cryptocurrency market, specifically Bitcoin."
    )

    st.markdown(
        "This application guides us through a real-world workflow to simulate historical Bitcoin data, "
        "build machine learning models (**LSTM** and **Random Forest**) to predict **next-day price direction**, "
        "and critically evaluate results using both model metrics and a simple strategy backtest."
    )

    st.header("2. Simulating Realistic Bitcoin Data: Building Our Testbed")

    st.markdown(
        "To assess AI's potential, we need a dataset that reflects the complex dynamics of Bitcoin. "
        "Real historical data can be messy and subject to availability or survivorship issues. "
        "For controlled experimentation, simulation lets us explicitly incorporate **extreme volatility**, **regime shifts** (bull/bear), "
        "**occasional jumps**, and the influence of **novel data sources** like on-chain metrics and sentiment. "
        "For an investment professional, this provides a realistic *yet controlled* environment to test modeling hypotheses before committing "
        "to production-grade data sourcing and governance."
    )

    st.markdown("The simulation will include:")
    st.markdown(
        "* **Price:** Modified GBM with regime shifts and occasional jumps.")
    st.markdown(
        "* **Volume**, **Technicals**, **On-chain metrics**, **Sentiment indices**.")
    st.markdown("* **Target:** `next_day_direction` (up/down).")

    st.markdown(
        r"Modified GBM with jumps:"
        r"""
$$ 
S_t = S_{t-1}\cdot\exp\Big((\mu-\tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z_t + J_t\Big)
$$"""
    )

    st.markdown(
        r"where $S_t$ is the price at time $t$, $\\mu$ is drift, $\\sigma$ is volatility, $\\Delta t$ is the time step (1 day), "
        r"$Z_t$ is a standard normal shock, and $J_t$ represents occasional jumps. "
        r"In our simulation, $\\mu$ and $\\sigma$ vary by regime (bull vs. bear), which intentionally stresses model robustness."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        n_days = st.number_input(
            "Number of simulated days", min_value=365, max_value=3000, value=1460, step=30)
    with c2:
        start_date = st.text_input(
            "Start date (YYYY-MM-DD)", value="2021-01-01")
    with c3:
        seed = st.number_input("Random seed", min_value=0,
                               max_value=10_000, value=42, step=1)

    if st.button("Generate Simulated Bitcoin Data", disabled=st.session_state.data_generated):
        with st.spinner("Generating data..."):
            np.random.seed(int(seed))
            tf.random.set_seed(int(seed))
            st.session_state.btc_data = generate_btc_data(
                n_days=int(n_days), start_date=str(start_date))
            st.session_state.data_generated = True
        st.success("Bitcoin data simulated successfully!")

    if st.session_state.btc_data is not None:
        btc = st.session_state.btc_data
        st.markdown(
            f"Simulated Bitcoin data generated for **{len(btc)}** days.")
        st.markdown(
            f"Total features: **{len(all_features)}** "
            f"({len(feature_groups['technical'])} technical, "
            f"{len(feature_groups['onchain'])} on-chain, "
            f"{len(feature_groups['sentiment'])} sentiment)"
        )
        st.markdown(
            f"Base rate (% up days): **{btc['next_day_direction'].mean():.2%}**")

        st.markdown("**First 5 rows:**")
        st.dataframe(btc.head(), use_container_width=True)

        st.markdown("**Descriptive statistics:**")
        st.dataframe(btc.describe(), use_container_width=True)

        st.subheader("Visualizing Price History with Market Regimes (V1)")

        st.markdown(
            "To better understand the simulated market environment, we visualize the Bitcoin price history and visually identify the "
            "bull/bear regimes. This helps confirm whether the simulation captures market cycles—critical for evaluating model robustness."
        )
        st.pyplot(_plot_price_with_regimes(btc))

    _footer()

# ----------------------------
# Page 2: Model Building & Training
# ----------------------------
elif st.session_state.current_page == "Model Building & Training":

    st.header("3. Preparing Data for Machine Learning Models")

    st.markdown(
        "Before feeding data into models, preparation is essential. For time-series models like LSTMs, we must transform data into sequences "
        "because the model learns from patterns across time. For Random Forest, we focus on feature relationships without sequence ordering. "
        "Most importantly, we split data **temporally** (past → train, future → test) to mirror real trading use and avoid look-ahead bias."
    )

    st.markdown(
        "Scaling features helps learning stability and prevents variables with larger numeric ranges from dominating. "
        "For LSTM, each sample must be a 3D tensor shaped `(samples, timesteps, features)`. "
        "This ensures both models receive inputs in the formats they expect, enabling fair evaluation."
    )
    st.markdown(
        "We scale features and create LSTM sequences, while ensuring a **temporal train/test split** "
        "(past → train, future → test) to avoid look-ahead bias."
    )

    if not st.session_state.data_generated:
        st.warning(
            "Please generate Bitcoin data first on the 'Introduction & Data' page.")
        _footer()
    else:
        if st.session_state.X_scaled is None:
            with st.spinner("Preparing data for models..."):
                st.session_state.scaler = StandardScaler()
                st.session_state.X_scaled = st.session_state.scaler.fit_transform(
                    st.session_state.btc_data[all_features])
                st.session_state.y = st.session_state.btc_data["next_day_direction"].values

                st.session_state.split_index = int(
                    len(st.session_state.btc_data) * 0.75)

                st.session_state.X_train_rf = st.session_state.X_scaled[: st.session_state.split_index]
                st.session_state.y_train_rf = st.session_state.y[: st.session_state.split_index]
                st.session_state.X_test_rf = st.session_state.X_scaled[st.session_state.split_index:]
                st.session_state.y_test_rf = st.session_state.y[st.session_state.split_index:]

                st.session_state.X_seq_train, st.session_state.y_seq_train = prepare_sequences(
                    st.session_state.X_scaled[: st.session_state.split_index],
                    st.session_state.y[: st.session_state.split_index],
                    SEQUENCE_LENGTH,
                )
                st.session_state.X_seq_test, st.session_state.y_seq_test = prepare_sequences(
                    st.session_state.X_scaled[st.session_state.split_index:],
                    st.session_state.y[st.session_state.split_index:],
                    SEQUENCE_LENGTH,
                )
            st.success("Data prepared!")

        st.markdown(
            f"Random Forest train shape: **{st.session_state.X_train_rf.shape}** | "
            f"test shape: **{st.session_state.X_test_rf.shape}**"
        )
        st.markdown(
            f"LSTM sequence train shape: **{st.session_state.X_seq_train.shape}** | "
            f"test shape: **{st.session_state.X_seq_test.shape}**"
        )

        st.markdown(
            "This meticulous preparation ensures no future data *leaks* into training—an essential safeguard to prevent over-optimistic "
            "backtests that can lead to poor investment decisions."
        )

        st.header("4. LSTM Model for Sequence Prediction")

        st.markdown(
            "Financial time series can exhibit temporal dependencies. The Long Short-Term Memory (LSTM) network is designed to capture "
            "long-term dependencies in sequential data by controlling what information is forgotten, stored, and exposed. "
            "Here we test whether the *sequence* of price/volume, on-chain activity, and sentiment provides incremental predictive signal."
        )

        st.markdown(
            "The core idea of an LSTM involves several *gates* that control information flow:")
        st.markdown(
            "* **Forget Gate:** decides what information to discard from the cell state.")
        st.markdown("* **Input Gate:** decides what new information to store.")
        st.markdown(
            "* **Cell State Update:** updates the internal memory using forget + input gates.")
        st.markdown("* **Output Gate:** decides what part of memory to output.")

        st.markdown(
            "For investment workflows, these mechanics matter because they encode a structured way to learn patterns like: "
            "‘if the last 30 days show a volatility regime + on-chain stress + sentiment shift, the probability of an up-day changes.’"
        )
        st.latex(r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)")
        st.latex(r"C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t")

        st.header("5. Random Forest for Feature-Based Prediction")

        st.markdown(
            "While LSTMs emphasize sequences, sometimes the non-linear relationships between features and outcomes matter more than ordering. "
            "A Random Forest is an ensemble of decision trees that captures complex interactions, offers robustness through averaging, "
            "and supports feature-importance analysis."
        )

        st.markdown(
            "For a CFA Charterholder, Random Forest serves two purposes: (1) a benchmark against a non-sequence-aware model, and "
            "(2) interpretability—helping answer *which* variables drive decisions (technical vs. on-chain vs. sentiment)."
        )
        st.latex(r"\hat{y} = \frac{1}{K}\sum_{k=1}^{K} h_k(x)")

        st.subheader("Train the Models")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            epochs = st.number_input(
                "LSTM epochs", min_value=5, max_value=200, value=50, step=5)
        with c2:
            batch_size = st.number_input(
                "LSTM batch size", min_value=8, max_value=256, value=32, step=8)
        with c3:
            threshold = st.slider(
                "Decision threshold", min_value=0.3, max_value=0.7, value=0.5, step=0.01)

        if st.button("Train Models", disabled=st.session_state.models_trained):
            with st.spinner("Training LSTM and Random Forest models..."):
                lstm_model = build_lstm_model(
                    n_features=st.session_state.X_seq_train.shape[2],
                    sequence_length=SEQUENCE_LENGTH,
                )
                lstm_model.fit(
                    st.session_state.X_seq_train,
                    st.session_state.y_seq_train,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    validation_split=0.15,
                    verbose=0,
                )
                st.session_state.lstm_probs = lstm_model.predict(
                    st.session_state.X_seq_test, verbose=0).flatten()
                st.session_state.lstm_preds = (
                    st.session_state.lstm_probs > float(threshold)).astype(int)
                st.session_state.lstm_model = lstm_model

                rf_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=20,
                    random_state=42,
                    class_weight="balanced",
                )
                rf_model.fit(st.session_state.X_train_rf,
                             st.session_state.y_train_rf)
                st.session_state.rf_probs = rf_model.predict_proba(
                    st.session_state.X_test_rf)[:, 1]
                st.session_state.rf_preds = (
                    st.session_state.rf_probs > float(threshold)).astype(int)
                st.session_state.rf_model = rf_model

                st.session_state.models_trained = True
            st.success("Models trained and predictions generated!")

            st.markdown(
                "The LSTM outputs probabilities which we convert to binary signals using a decision threshold. "
                "For practitioners, the threshold governs conviction: higher thresholds may reduce false positives but increase missed opportunities."
            )

            st.markdown(
                "The Random Forest similarly produces probabilities and binary signals. Its feature-importance analysis can improve trust: "
                "understanding *why* a prediction is made is often as important as the prediction itself for governance and risk oversight."
            )

        _footer()

# ----------------------------
# Page 3: Performance & Insights
# ----------------------------
elif st.session_state.current_page == "Performance & Insights":

    st.header("6. Walk-Forward Performance Evaluation & Strategy Simulation")

    st.markdown(
        "Evaluating models in financial markets requires more rigor than a single split. **Walk-forward validation** simulates a real-world "
        "deployment pattern by repeatedly training on an expanding window and testing on the next period. This highlights performance decay "
        "under regime shifts and concept drift—central concerns in crypto."
    )

    st.markdown(
        "We translate predictions into a naive strategy: go **long** when the model predicts an up-day, and stay **flat** otherwise. "
        "This maps model outputs to portfolio outcomes and allows comparison to **buy-and-hold** using directional accuracy, Sharpe ratio, and total return."
    )
    st.markdown(
        "We translate predictions into a naive trading strategy and evaluate performance with "
        "**directional accuracy**, **Sharpe ratio**, and **total return**, compared to **buy-and-hold**."
    )

    st.latex(
        r"\text{Directional Accuracy}=\frac{\#\ \text{Correct}}{\#\ \text{Total}}")

    st.latex(
        r"\text{Sharpe Ratio}=\frac{\mathbb{E}[R_s - R_f]}{\sigma_s}\;\approx\;\frac{\text{Mean Daily Strategy Return}}{\text{Std Dev of Daily Strategy Return}}\sqrt{\text{Trading Days/Year}}")
    st.markdown(
        r"where $R_s$ is the strategy's daily return, $R_f$ is the risk-free rate (assume 0 for simplicity here), and $\sigma_s$ is the standard deviation of daily returns."
    )

    if not st.session_state.models_trained:
        st.warning(
            "Please train the models first on the 'Model Building & Training' page.")
        _footer()
    else:
        if not st.session_state.metrics_evaluated:
            with st.spinner("Evaluating model performance..."):
                st.session_state.test_log_returns_lstm = (
                    st.session_state.btc_data["next_day_return"].iloc[st.session_state.split_index +
                                                                      SEQUENCE_LENGTH:].values
                )
                st.session_state.test_log_returns_rf = (
                    st.session_state.btc_data["next_day_return"].iloc[st.session_state.split_index:].values
                )

                st.session_state.lstm_eval_metrics = evaluate_crypto_model(
                    st.session_state.y_seq_test,
                    st.session_state.lstm_preds,
                    st.session_state.lstm_probs,
                    st.session_state.test_log_returns_lstm,
                    "LSTM",
                )
                st.session_state.rf_eval_metrics = evaluate_crypto_model(
                    st.session_state.y_test_rf,
                    st.session_state.rf_preds,
                    st.session_state.rf_probs,
                    st.session_state.test_log_returns_rf,
                    "Random Forest",
                )
                st.session_state.metrics_evaluated = True
            st.success("Model performance evaluated!")

            st.markdown(
                "These outputs let us judge whether either model provides a *meaningful* edge over buy-and-hold, particularly in risk-adjusted terms. "
                "If accuracy is barely above 50% and Sharpe does not improve, any real-world profitability after transaction costs is questionable."
            )

        if st.session_state.metrics_evaluated:
            st.subheader("Model Performance Comparison (V2)")
            comparison_df = pd.DataFrame(
                {
                    "Metric": ["Directional Accuracy", "Sharpe Ratio", "Total Return"],
                    "LSTM": [
                        f"{st.session_state.lstm_eval_metrics['directional_accuracy']:.2%}",
                        f"{st.session_state.lstm_eval_metrics['sharpe_ratio']:.2f}",
                        f"{st.session_state.lstm_eval_metrics['total_return']:.2%}",
                    ],
                    "Random Forest": [
                        f"{st.session_state.rf_eval_metrics['directional_accuracy']:.2%}",
                        f"{st.session_state.rf_eval_metrics['sharpe_ratio']:.2f}",
                        f"{st.session_state.rf_eval_metrics['total_return']:.2%}",
                    ],
                    "Buy-and-Hold Baseline": [
                        "N/A",
                        f"{st.session_state.rf_eval_metrics['bh_sharpe_ratio']:.2f}",
                        f"{st.session_state.rf_eval_metrics['bh_total_return']:.2%}",
                    ],
                }
            )
            st.dataframe(comparison_df, use_container_width=True)

            st.header("7. Feature Importance Analysis")

            st.markdown(
                "Understanding *why* a model predicts is as important as the prediction itself. Feature importance helps test whether crypto-native "
                "signals (on-chain, sentiment) add incremental value beyond technical indicators. This supports research prioritization and governance."
            )

            st.markdown(
                "In Random Forests, importance is typically measured via average impurity reduction (Gini importance)."
            )
            if st.session_state.importance_df is None:
                st.session_state.importance_df = (
                    pd.Series(
                        st.session_state.rf_model.feature_importances_, index=all_features)
                    .sort_values(ascending=False)
                    .to_frame(name="importance")
                )
                st.session_state.importance_df["group"] = st.session_state.importance_df.index.map(
                    lambda x: next(
                        (g for g, feats in feature_groups.items() if x in feats), "Other")
                )

            st.subheader("Random Forest Feature Importance by Group (V3)")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(
                x="importance",
                y=st.session_state.importance_df.index,
                hue="group",
                data=st.session_state.importance_df,
                dodge=False,
                palette="viridis",
                ax=ax,
            )
            ax.set_title("Random Forest Feature Importance by Group")
            ax.set_xlabel("Importance (Gini Importance)")
            ax.set_ylabel("Feature")
            ax.legend(title="Feature Group")
            fig.tight_layout()
            st.pyplot(fig)

            st.subheader("Prediction Confidence Distribution (V4)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.lstm_probs, bins=20,
                         kde=True, label="LSTM Probabilities", ax=ax)
            sns.histplot(st.session_state.rf_probs, bins=20, kde=True,
                         label="Random Forest Probabilities", alpha=0.7, ax=ax)
            ax.axvline(0.5, color="gray", linestyle="--",
                       label="Decision Threshold (0.5)")
            ax.set_title(
                "Distribution of Predicted Probabilities (Confidence)")
            ax.set_xlabel("Predicted Probability of Up-Day")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()
            st.pyplot(fig)

            st.subheader("Strategy Equity Curve (V5)")
            fig, ax = plt.subplots(figsize=(14, 7))
            lstm_dates = st.session_state.btc_data["date"].iloc[st.session_state.split_index + SEQUENCE_LENGTH:]
            rf_dates = st.session_state.btc_data["date"].iloc[st.session_state.split_index:]

            ax.plot(
                lstm_dates, st.session_state.lstm_eval_metrics["strategy_cumulative_returns"], label="LSTM Strategy")
            ax.plot(
                rf_dates, st.session_state.rf_eval_metrics["strategy_cumulative_returns"], label="Random Forest Strategy")
            ax.plot(
                rf_dates, st.session_state.rf_eval_metrics["bh_cumulative_returns"], label="Buy-and-Hold Baseline", linestyle="--")
            ax.set_title("Cumulative Returns: AI Strategies vs. Buy-and-Hold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Return")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)

        _footer()

# ----------------------------
# Page 4: Assessment & Warning
# ----------------------------
elif st.session_state.current_page == "Assessment & Warning":

    st.header("8. Critical Assessment & Discussion: The Humility Exercise")

    st.subheader("Why Crypto Prediction Is Hard: A Mathematical Formulation")
    st.markdown(
        "* **Signal-to-noise ratio:** daily crypto returns often exhibit volatility far larger than their expected mean return, "
        "making it difficult to extract stable predictive signal."
    )
    st.markdown(
        "* **Non-stationarity:** frequent regime shifts (bull/bear, bubble/crash) invalidate models trained on prior periods."
    )
    st.markdown(
        "* **Reflexivity:** narratives and sentiment influence prices, and prices in turn influence narratives—creating feedback loops that violate i.i.d. assumptions."
    )
    st.markdown(
        "* **Expected directional accuracy:** for well-designed models in noisy speculative markets, a realistic out-of-sample range is often only **52–56%**. "
        "Below ~52% suggests no meaningful signal; sustained results above ~58% should trigger skepticism and checks for data leakage or overfitting."
    )
    st.markdown(
        "The professional output is not just performance numbers—it is a clear understanding of where AI is useful *and* where it can mislead."
    )
    st.markdown(
        "This final section confronts the key lesson: AI is not a crystal ball in speculative assets. "
        "The right professional use is often **risk management and monitoring**, not short-term return prediction."
    )

    if not st.session_state.metrics_evaluated:
        st.warning(
            "Please evaluate model performance first on the 'Performance & Insights' page.")
        _footer()
    else:
        st.subheader("Honest Assessment Summary (V6)")
        st.markdown("---")
        st.markdown(
            "### HONEST ASSESSMENT: AI IN CRYPTO MARKETS (A CFA's Perspective)")
        st.markdown("---")
        st.markdown(
            "#### WHERE AI ADDS LIMITED VALUE (short-term price prediction):")
        st.markdown(
            "- Short-term price prediction: 52-56% directional accuracy - barely better than a coin flip.")
        st.markdown(
            "- Not sufficient for profitable trading after typical transaction costs (0.1-0.3%).")
        st.markdown(
            "- Overfitting risk is extreme in noisy crypto data due to high signal-to-noise ratio (noise >> signal).")
        st.markdown(
            "- Regime shifts invalidate trained models quickly (non-stationarity).")

        st.markdown(
            "#### WHERE AI ADDS GENUINE VALUE (risk management, anomaly detection, sentiment monitoring):")
        st.markdown(
            "- On-chain anomaly detection: Identifying unusual whale movements, exchange inflows/outflows, or network hacks.")
        st.markdown(
            "- Sentiment monitoring: Early warning of narrative shifts, extreme fear/greed sentiment, or coordinated FUD/FOMO campaigns.")
        st.markdown(
            "- Risk management: Volatility forecasting, drawdown prediction, and portfolio optimization for digital assets.")
        st.markdown(
            "- Fraud detection: Identifying wash trading, rug pull schemes, or scam projects.")
        st.markdown(
            "- 24/7 monitoring: AI agents can continuously monitor markets for anomalies, which human analysts cannot do.")

        st.markdown("#### KEY TAKEAWAY FOR CFA PROFESSIONALS:")
        st.markdown(
            "- AI is NOT a crystal ball. In speculative markets with low signal-to-noise ratios, prediction is extremely difficult.")
        st.markdown(
            "- The professional's role is to deploy AI where it adds genuine value (risk management, anomaly detection, sentiment insights).")
        st.markdown(
            "- Resist the temptation to over-rely on return prediction or blindly trust 'AI crypto trading bots'.")
        st.markdown("- Apply CFA Standard V(A) - Diligence and Reasonable Basis: Always exercise due diligence on AI-driven claims as with any investment product.")
        st.markdown("---")

        st.subheader("Practitioner Warning")
        st.warning(
            "**Beware the 'AI crypto trading bot' hype.** Numerous products claim AI-powered crypto trading with extraordinary returns. "
            "The reality, as demonstrated in this lab, is that **52-56% directional accuracy is the state of the art for well-designed models "
            "in highly noisy, speculative markets like Bitcoin.** After accounting for typical transaction costs (0.1-0.3% per trade in crypto), "
            "even 56% accuracy may not be profitable. Any product claiming 70%+ accuracy in crypto should be treated with **extreme skepticism**."
        )
        st.markdown("* **Survivorship bias:** Showing only winning periods.")
        st.markdown(
            "* **In-sample fitting presented as out-of-sample:** Data leakage or outright fabrication.")
        st.markdown(
            "* **Over-optimization:** Tuning perfectly to past data that it fails in the future.")
        st.warning(
            "**Apply the same due diligence standards (CFA Standard V(A)) to AI trading claims as to any other investment product.** "
            "Understand methodology, data sources, and evaluation metrics before committing capital."
        )

        st.caption(
            "This was developed with the assistance of QCreate.ai, an AI-powered platform that supported content structuring, formatting, "
            "and the creation of illustrative narratives under human supervision. All analysis, interpretation, and final editorial judgment "
            "were made by the author(s). The case is intended solely for educational use. For feedback or corrections, please contact info@qusandbox.com."
        )

        _footer()
