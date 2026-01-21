# Regime-Augmented Deep Reinforcement Learning for Volatility Trading

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A novel algorithmic trading architecture that integrates **Wavelet Denoising**, **Probabilistic Regime Inference (HMM)**, and **Deep Reinforcement Learning (PPO)**. This system decouples signal processing from decision-making, allowing the agent to dynamically adapt its aggression based on latent market states.

## 📄 Key Resources
* **[Read the Full Whitepaper (PDF)](./papers/Regime_Augmented_Wavelet_Enhanced_DRL_Whitepaper.pdf)** – Detailed methodology, mathematical formulation, and stress test results.
* **[View the Implementation Code](./Regime_Aware_Wavelet_Enhanced_DRL_Trading.ipynb)** – Complete Python notebook including data pipeline, HMM training, and PPO agent execution.

---

## 🏗️ Architecture Overview
The system employs a sequential pipeline to transform raw, noisy market microstructure data into a context-aware state representation.

1.  **Spectral Decomposition (Signal Purification):** Raw signals (VIX, GEX, DIX) are processed using Discrete Wavelet Transform (`db4`, Level-2) to remove high-frequency noise.
2.  **Latent Inference (Context Injection):** A Gaussian Hidden Markov Model (HMM) infers the probability of the current market regime ($P(z_t|x_{1:t})$) and feeds this context to the agent.
3.  **Deep Policy Network:** A PPO agent utilizing **Multi-Head Attention** and **Bi-LSTM** to weigh regime context against price signals.

---

## 📊 Performance Highlights
Tested on S&P 500 E-Mini Futures (2011–2025), Out-of-Sample.

| Model | Total Return | Sharpe Ratio | Max Drawdown | Alpha |
| :--- | :--- | :--- | :--- | :--- |
| **RL Proposed (Signal + Regime)** | **91.51%** | **0.80** | **14.5%** | **0.27** |
| RL Baseline (Signal Only) | 52.68% | 0.98 | 13.1% | 0.14 |
| Buy & Hold (Benchmark) | 43.46% | 0.71 | 18.9% | -0.00 |

> **Key Insight:** The inclusion of HMM regime probabilities generated **0.13 excess Alpha** compared to the baseline agent.

---

## 🚀 Usage
1. Clone the repository.
2. Install dependencies: `pip install numpy pandas matplotlib scikit-learn torch stable-baselines3 shimmy hmmlearn pywavelets gymnasium`
3. Run the `Regime_Augmented_DRL_Implementation.ipynb` notebook.

---

## 📂 Dataset Info
Utilizes **S&P 500 E-Mini Futures** daily data combined with **Market Microstructure Indicators** (DIX, GEX) provided by [SqueezeMetrics](https://squeezemetrics.com/).

---

## ⚠️ Disclaimer
This codebase is for **educational and research purposes only**. It is not financial advice.
