"""
Visualization functions for kinetic modeling analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple


def create_matplotlib_plots(df: pd.DataFrame, selected_data: pd.DataFrame,
                           zo_predictions: pd.DataFrame, pfo_predictions: pd.DataFrame, pso_predictions: pd.DataFrame,
                           k0: float, k1: float, k2: float) -> plt.Figure:
    """
    Create matplotlib plots for ZO, PFO and PSO models.

    Args:
        df: Full dataset
        selected_data: Selected stable points
        zo_predictions: ZO model predictions
        pfo_predictions: PFO model predictions
        pso_predictions: PSO model predictions
        k0: ZO rate constant
        k1: PFO rate constant
        k2: PSO rate constant

    Returns:
        Matplotlib figure
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))

    # ZO plot
    ax0.plot(df['т, мин'], df['А'], 'bo-', label='Экспериментальные данные (A)', markersize=4)
    ax0.plot(zo_predictions['т, мин'], zo_predictions['ZO_pred'], 'r--',
             label=f'Модель ZO (k₀={abs(k0):.5f})', linewidth=2)
    ax0.set_title('Модель ZO')
    ax0.set_xlabel('Время, мин')
    ax0.set_ylabel('A')
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # PFO plot
    ax1.plot(df['т, мин'], df['ln_A_A0'], 'go-', label='Экспериментальные данные (ln(A/A₀))', markersize=4)
    ax1.plot(pfo_predictions['т, мин'], pfo_predictions['PFO_pred_ln'], 'r--',
             label=f'Модель PFO (k₁={abs(k1):.5f})', linewidth=2)
    ax1.set_title('Модель PFO')
    ax1.set_xlabel('Время, мин')
    ax1.set_ylabel('ln(A/A₀)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PSO plot
    ax2.plot(df['т, мин'], df['inv_A'], 'mo-', label='Экспериментальные данные (1/A)', markersize=4)
    ax2.plot(pso_predictions['т, мин'], pso_predictions['PSO_pred_inv'], 'r--',
             label=f'Модель PSO (k₂={k2:.5f})', linewidth=2)
    ax2.set_title('Модель PSO')
    ax2.set_xlabel('Время, мин')
    ax2.set_ylabel('1/A')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
