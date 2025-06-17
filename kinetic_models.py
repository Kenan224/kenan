"""
Kinetic modeling functions for PFO and PSO analysis.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from typing import List, Tuple, Dict


def find_stable_points(y: pd.Series, t: pd.Series, threshold: float = 0.1) -> List[int]:
    """
    Automatically identifies a range of 'stable' points in the data based on
    the change in slope. This is useful for selecting the linear region
    often used in kinetic model fitting.

    Args:
        y: The dependent variable data (e.g., ln(A/A0))
        t: The independent variable data (e.g., time)
        threshold: A threshold to determine when the slope significantly changes

    Returns:
        List of indices corresponding to the stable points
    """
    stable_indices = [0]  # Always start with the first point
    initial_slope = None
    previous_slope = 0

    for i in range(1, len(y)):
        delta_y = y.iloc[i] - y.iloc[stable_indices[-1]]
        delta_t = t.iloc[i] - t.iloc[stable_indices[-1]]

        if delta_t == 0:
            # Skip if time doesn't change to avoid division by zero
            continue

        current_slope = delta_y / delta_t

        if initial_slope is None:
            # Set the initial slope using the first valid segment
            initial_slope = current_slope
        else:
            # Check for slope change based on threshold or sign change
            if initial_slope != 0:  # Avoid division by zero if initial_slope is 0
                slope_ratio = abs(current_slope / initial_slope)
                if slope_ratio < threshold or (current_slope * previous_slope < 0):
                    # Break if slope significantly flattens or changes direction
                    break
            else:  # If initial_slope is 0, check if current_slope deviates significantly
                if abs(current_slope) > threshold:  # If current slope is not near zero
                    break

        stable_indices.append(i)
        previous_slope = current_slope

    return stable_indices


def pfo_model(t: np.ndarray, k1: float) -> np.ndarray:
    """
    Pseudo-First Order kinetic model: ln(A/A0) = -k1 * t

    Args:
        t: Time array
        k1: Pseudo-first order rate constant

    Returns:
        Predicted ln(A/A0) values
    """
    return -k1 * t


def pso_model(t: np.ndarray, k2: float, A0: float) -> np.ndarray:
    """
    Pseudo-Second Order kinetic model: 1/A = 1/A0 + k2 * t

    Args:
        t: Time array
        k2: Pseudo-second order rate constant
        A0: Initial concentration

    Returns:
        Predicted 1/A values
    """
    return (1 / A0) + k2 * t


def fit_pfo_model(selected_data: pd.DataFrame) -> Tuple[float, pd.DataFrame, float, float]:
    """
    Fit the Pseudo-First Order model to the selected data.

    Args:
        selected_data: DataFrame with selected stable points

    Returns:
        Tuple of (k1, predictions_df, mape, r2)
    """
    # Fit the PFO model
    pfo_params, _ = curve_fit(pfo_model, selected_data['т, мин'], selected_data['ln_A_A0'])
    k1 = pfo_params[0]

    # Calculate predictions
    predictions = selected_data.copy()
    predictions['PFO_pred_ln'] = pfo_model(selected_data['т, мин'], k1)
    predictions['PFO_pred'] = np.exp(predictions['PFO_pred_ln'])

    # Calculate metrics
    mape = mean_absolute_percentage_error(selected_data['А/А0'], predictions['PFO_pred']) * 100
    r2 = r2_score(selected_data['А/А0'], predictions['PFO_pred'])

    return k1, predictions, mape, r2


def fit_pso_model(selected_data: pd.DataFrame) -> Tuple[float, pd.DataFrame, float, float]:
    """
    Fit the Pseudo-Second Order model to the selected data.

    Args:
        selected_data: DataFrame with selected stable points

    Returns:
        Tuple of (k2, predictions_df, mape, r2)
    """
    A0 = selected_data.iloc[0]['А']

    # Create a partial function with A0 fixed
    def pso_model_partial(t, k2):
        return pso_model(t, k2, A0)

    # Fit the PSO model
    pso_params, _ = curve_fit(pso_model_partial, selected_data['т, мин'], selected_data['inv_A'])
    k2 = pso_params[0]

    # Calculate predictions
    predictions = selected_data.copy()
    predictions['PSO_pred_inv'] = pso_model(selected_data['т, мин'], k2, A0)
    predictions['PSO_pred'] = 1 / predictions['PSO_pred_inv']

    # Calculate metrics
    mape = mean_absolute_percentage_error(selected_data['А'], predictions['PSO_pred']) * 100
    r2 = r2_score(selected_data['А'], predictions['PSO_pred'])

    return k2, predictions, mape, r2


def create_results_summary(k1: float, k2: float, mape_pfo: float, mape_pso: float,
                          r2_pfo: float, r2_pso: float) -> pd.DataFrame:
    """
    Create a summary DataFrame of model results.

    Args:
        k1: PFO rate constant
        k2: PSO rate constant
        mape_pfo: PFO MAPE
        mape_pso: PSO MAPE
        r2_pfo: PFO R-squared
        r2_pso: PSO R-squared

    Returns:
        Summary DataFrame
    """
    return pd.DataFrame({
        'Модель': ['PFO', 'PSO'],
        'Параметры': [
            f'k₁ = {abs(k1):.5f} мин⁻¹',
            f'k₂ = {k2:.5f} мин⁻¹'
        ],
        'R²': [r2_pfo, r2_pso],
        'MAPE (%)': [mape_pfo, mape_pso]
    })


def create_detailed_results(pfo_predictions: pd.DataFrame, pso_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Create detailed point-by-point results DataFrame.

    Args:
        pfo_predictions: PFO model predictions
        pso_predictions: PSO model predictions

    Returns:
        Detailed results DataFrame
    """
    return pd.DataFrame({
        'Время (мин)': pfo_predictions['т, мин'],
        'A/A0 фактическое': pfo_predictions['А/А0'],
        'PFO прогноз': pfo_predictions['PFO_pred'],
        'PFO ошибка (%)': np.abs((pfo_predictions['А/А0'] - pfo_predictions['PFO_pred']) / pfo_predictions['А/А0']) * 100,
        'A фактическое': pso_predictions['А'],
        'PSO прогноз': pso_predictions['PSO_pred'],
        'PSO ошибка (%)': np.abs((pso_predictions['А'] - pso_predictions['PSO_pred']) / pso_predictions['А']) * 100,
    })
