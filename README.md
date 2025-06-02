# Neural Network Prediction Liminality and Stability Analysis

This Jupyter Notebook (`LiminalityStabilityAnalysis.ipynb` - *assuming this would be the file name*) delves into the dynamic behavior of neural network predictions during training, specifically focusing on two key metrics: "Liminality" and "Stability." It provides a custom `LiminalityTracker` class to monitor and visualize how these metrics evolve as a simple Keras model learns on synthetic data.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Key Concepts](#key-concepts)
3.  [Methodology](#methodology)
4.  [Experiment Scenarios](#experiment-scenarios)
5.  [Visualizations](#visualizations)
6.  [Requirements](#requirements)
7.  [How to Run](#how-to-run)

## Project Overview

The goal of this project is to provide a comparative view of how different techniques identify important features and how consistently those features exhibit significance across iterations or different interpretability methods. This concept of consistency or reliability in feature selection is termed "Feature Certainty" or "Liminality."

## Key Concepts

* **Liminality:** Defined as the percentage of model predictions that fall within a narrow range around the decision boundary (e.g., 0.5 for binary classification). A high liminality might indicate a model that is uncertain about a large portion of its predictions, while a decreasing liminality suggests increasing confidence.
* **Stability:** Defined as the variance of the model's predictions. A high variance could imply unstable predictions, while a decreasing variance might indicate that the model's predictions are becoming more consistent or stable.

## Methodology

The core of the analysis is the `LiminalityTracker` class:
* It takes a Keras model, input data, and labels during initialization.
* During the `fit` method (which wraps the model's training loop), it periodically calculates and stores:
    * **Liminality:** By checking how many predictions are close to `0.5` (for a sigmoid output).
    * **Stability:** By computing the variance of the predictions.
* The `plot_metrics` method visualizes the recorded liminality and stability scores over epochs.

## Experiment Scenarios

The notebook presents several scenarios using synthetic data and variations in model complexity or data noise to observe their effects on liminality and stability:

1.  **Basic Tracking:** Introduces the `LiminalityTracker` and plots liminality over epochs for a simple model.
2.  **Combined Metrics:** Extends the tracker to include and plot both liminality and stability on a dual-axis graph.
3.  **Underpowered Model with Noise:** Explores the behavior of metrics when a simpler model (fewer layers/neurons) is trained on data with added noise.
4.  **Very Simple Model with High Noise:** Further simplifies the model and increases data noise to observe extreme cases of liminality and stability.
5.  **Constant Output Model:** Demonstrates the metrics for a model initialized to output a constant value, providing a baseline for stable but potentially non-converging behavior.

Each scenario showcases how model architecture, data characteristics, and training progress influence the model's predictive confidence and consistency.

## Visualizations

The notebook generates line plots showing:
* "Liminality Over Epochs" (percentage of predictions near the decision boundary).
* "Stability Over Epochs" (variance of predictions).
* In some sections, these are combined into a single plot with dual Y-axes for comparative analysis.

## Requirements

To run this notebook, you will need the following Python libraries:

* `numpy`
* `matplotlib`
* `keras` (or `tensorflow.keras`)

You can install them using pip:

```bash
pip install numpy matplotlib keras # If using TensorFlow 2.x, keras is usually included with tensorflow
```
## How to Run

1.  **Clone the repository** (if applicable) or download the `LiminalityStabilityAnalysis.ipynb` file.
2.  **Ensure you have Jupyter Notebook or JupyterLab installed.** If not, you can install it via pip:

    ```bash
    pip install notebook  # or pip install jupyterlab
    ```

3.  **Navigate to the directory** containing the notebook in your terminal.
4.  **Launch Jupyter Notebook/Lab:**

    ```bash
    jupyter notebook # or jupyter lab
    ```

5.  **Open `LiminalityStabilityAnalysis.ipynb`** from the Jupyter interface.
6.  **Run all cells** to execute the code and generate the plots.
