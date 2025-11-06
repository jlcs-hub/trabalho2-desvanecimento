"""
Script para Parte II: Parâmetros de Caracterização do Canal
Calcula PDP, RMS Delay Spread, Mean/Maximum Excess Delay e gera animações

Autor: João Lucas de Castro Santos
Data: 2025-11-05
"""

import sys
import os

# Adicionar path do src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.cir_loader import load_both_datasets
from src.channel_parameters import (
    compute_all_parameters,
    compare_datasets,
    analyze_frequency_selectivity
)
from src.animation import create_cir_animation, create_pdp_comparison_animation

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_pdp(pdp: np.ndarray, delays: np.ndarray,
            title: str, output_path: str):
    """Plota Power Delay Profile"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Escala linear
    ax1.plot(delays, pdp, 'b-o', linewidth=2, markersize=8, alpha=0.7)
    ax1.set_xlabel('Delay (ns)', fontsize=12)
    ax1.set_ylabel('Potência Média Normalizada', fontsize=12)
    ax1.set_title(f'{title} - Escala Linear', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Escala dB
    pdp_db = 10 * np.log10(pdp / np.max(pdp))
    ax2.plot(delays, pdp_db, 'r-o', linewidth=2, markersize=8, alpha=0.7)
    ax2.set_xlabel('Delay (ns)', fontsize=12)
    ax2.set_ylabel('Potência Relativa (dB)', fontsize=12)
    ax2.set_title(f'{title} - Escala dB', fontsize=14, fontweight='bold')
    ax2.axhline(y=-20, color='k', linestyle='--', label='Threshold -20 dB')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def plot_pdp_comparison(pdp_20: np.ndarray, delays_20: np.ndarray,
                       pdp_8: np.ndarray, delays_8: np.ndarray,
                       output_path: str):
    """Plota comparação de PDPs (20 vs 8 taps)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Escala linear
    ax1.plot(delays_20, pdp_20, 'b-o', linewidth=2, markersize=8,
            label='20 taps', alpha=0.7)
    ax1.plot(delays_8, pdp_8, 'r-s', linewidth=2, markersize=8,
            label='8 taps', alpha=0.7)
    ax1.set_xlabel('Delay (ns)', fontsize=12)
    ax1.set_ylabel('Potência Média Normalizada', fontsize=12)
    ax1.set_title('Comparação de PDP: 20 vs 8 taps - Escala Linear',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Escala dB
    pdp_20_db = 10 * np.log10(pdp_20 / np.max(pdp_20))
    pdp_8_db = 10 * np.log10(pdp_8 / np.max(pdp_8))

    ax2.plot(delays_20, pdp_20_db, 'b-o', linewidth=2, markersize=8,
            label='20 taps', alpha=0.7)
    ax2.plot(delays_8, pdp_8_db, 'r-s', linewidth=2, markersize=8,
            label='8 taps', alpha=0.7)
    ax2.set_xlabel('Delay (ns)', fontsize=12)
    ax2.set_ylabel('Potência Relativa (dB)', fontsize=12)
    ax2.set_title('Comparação de PDP: 20 vs 8 taps - Escala dB',
                 fontsize=14, fontweight='bold')
    ax2.axhline(y=-20, color='k', linestyle='--', alpha=0.5, label='Threshold -20 dB')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def plot_parameters_comparison(comparison: dict, output_path: str):
    """Plota comparação de parâmetros entre datasets"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    params = ['mean_excess_delay', 'rms_delay_spread', 'maximum_excess_delay']
    titles = ['Mean Excess Delay', 'RMS Delay Spread', 'Maximum Excess Delay']
    colors = ['steelblue', 'coral']

    for idx, (param, title) in enumerate(zip(params, titles)):
        ax = axes[idx]

        values = [comparison[param]['20_taps'], comparison[param]['8_taps']]
        labels = ['20 taps', '8 taps']

        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')

        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f} ns',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Diferença percentual
        diff_pct = comparison[param]['diff_percent']
        ax.text(0.5, 0.95, f'Δ = {diff_pct:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylabel('Tempo (ns)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Comparação de Parâmetros: 20 vs 8 taps',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def save_results_to_csv(params_20: dict, params_8: dict, comparison: dict):
    """Salva resultados em CSV"""
    # Tabela de parâmetros
    data = {
        'Parametro': [
            'Mean Excess Delay (ns)',
            'RMS Delay Spread (ns)',
            'Maximum Excess Delay (ns)',
            'Number of Taps',
            'Delay Range Min (ns)',
            'Delay Range Max (ns)'
        ],
        '20_taps': [
            params_20['mean_excess_delay_ns'],
            params_20['rms_delay_spread_ns'],
            params_20['maximum_excess_delay_ns'],
            params_20['n_taps'],
            params_20['delay_range_ns'][0],
            params_20['delay_range_ns'][1]
        ],
        '8_taps': [
            params_8['mean_excess_delay_ns'],
            params_8['rms_delay_spread_ns'],
            params_8['maximum_excess_delay_ns'],
            params_8['n_taps'],
            params_8['delay_range_ns'][0],
            params_8['delay_range_ns'][1]
        ]
    }

    df = pd.DataFrame(data)
    output_path = 'results/parte2_parametros.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Parâmetros salvos em: {output_path}")

    # Tabela de comparação
    comp_data = []
    for param in ['mean_excess_delay', 'rms_delay_spread', 'maximum_excess_delay']:
        comp_data.append({
            'Parametro': param,
            '20_taps': comparison[param]['20_taps'],
            '8_taps': comparison[param]['8_taps'],
            'diff_ns': comparison[param]['diff_ns'],
            'diff_percent': comparison[param]['diff_percent']
        })

    df_comp = pd.DataFrame(comp_data)
    output_path_comp = 'results/parte2_comparacao.csv'
    df_comp.to_csv(output_path_comp, index=False)
    print(f"✓ Comparação salva em: {output_path_comp}")


def main():
    """Função principal"""
    print("\n" + "="*70)
    print("PARTE II: PARÂMETROS DE CARACTERIZAÇÃO DO CANAL")
    print("="*70 + "\n")

    # Carregar dados
    print("Carregando datasets CIR...")
    data_20, data_8 = load_both_datasets()
    print(f"  ✓ Dataset 20 taps: {data_20.n_cirs} CIRs, {data_20.n_taps} taps")
    print(f"  ✓ Dataset 8 taps: {data_8.n_cirs} CIRs, {data_8.n_taps} taps")

    # ========================================================================
    # PROCESSAMENTO 20 TAPS
    # ========================================================================
    print("\n" + "="*70)
    print("PROCESSANDO: 20 TAPS")
    print("="*70 + "\n")

    print("1. Calculando parâmetros do canal...")
    params_20 = compute_all_parameters(data_20.cirs, data_20.delays, threshold_db=-20)

    print(f"\n  Resultados (20 taps):")
    print(f"  {'─'*50}")
    print(f"  Mean Excess Delay:     {params_20['mean_excess_delay_ns']:.2f} ns")
    print(f"  RMS Delay Spread:      {params_20['rms_delay_spread_ns']:.2f} ns")
    print(f"  Maximum Excess Delay:  {params_20['maximum_excess_delay_ns']:.2f} ns")
    print(f"                         (threshold = {params_20['threshold_db']} dB)")

    # Coherence bandwidth
    from src.channel_parameters import compute_coherence_bandwidth
    B_c_20_09 = compute_coherence_bandwidth(params_20['rms_delay_spread_ns'], 0.9)
    B_c_20_05 = compute_coherence_bandwidth(params_20['rms_delay_spread_ns'], 0.5)

    print(f"\n  Coherence Bandwidth:")
    print(f"    90% correlation: {B_c_20_09:.2f} MHz")
    print(f"    50% correlation: {B_c_20_05:.2f} MHz")

    print("\n2. Plotando PDP...")
    plot_pdp(params_20['pdp'], data_20.delays,
            'Power Delay Profile (20 taps)',
            'figures/parte2/pdp_20taps.png')

    print("\n3. Criando animação de CIRs...")
    create_cir_animation(
        data_20.cirs,
        data_20.delays,
        n_frames=100,
        output_path='media/cir_animation_20taps.gif',
        fps=10,
        title='Evolução das CIRs (20 taps)'
    )

    # ========================================================================
    # PROCESSAMENTO 8 TAPS
    # ========================================================================
    print("\n" + "="*70)
    print("PROCESSANDO: 8 TAPS")
    print("="*70 + "\n")

    print("1. Calculando parâmetros do canal...")
    params_8 = compute_all_parameters(data_8.cirs, data_8.delays, threshold_db=-20)

    print(f"\n  Resultados (8 taps):")
    print(f"  {'─'*50}")
    print(f"  Mean Excess Delay:     {params_8['mean_excess_delay_ns']:.2f} ns")
    print(f"  RMS Delay Spread:      {params_8['rms_delay_spread_ns']:.2f} ns")
    print(f"  Maximum Excess Delay:  {params_8['maximum_excess_delay_ns']:.2f} ns")
    print(f"                         (threshold = {params_8['threshold_db']} dB)")

    # Coherence bandwidth
    B_c_8_09 = compute_coherence_bandwidth(params_8['rms_delay_spread_ns'], 0.9)
    B_c_8_05 = compute_coherence_bandwidth(params_8['rms_delay_spread_ns'], 0.5)

    print(f"\n  Coherence Bandwidth:")
    print(f"    90% correlation: {B_c_8_09:.2f} MHz")
    print(f"    50% correlation: {B_c_8_05:.2f} MHz")

    print("\n2. Plotando PDP...")
    plot_pdp(params_8['pdp'], data_8.delays,
            'Power Delay Profile (8 taps)',
            'figures/parte2/pdp_8taps.png')

    print("\n3. Criando animação de CIRs...")
    create_cir_animation(
        data_8.cirs,
        data_8.delays,
        n_frames=100,
        output_path='media/cir_animation_8taps.gif',
        fps=10,
        title='Evolução das CIRs (8 taps)'
    )

    # ========================================================================
    # COMPARAÇÃO 20 vs 8 TAPS
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARAÇÃO: 20 vs 8 TAPS")
    print("="*70 + "\n")

    print("1. Comparando parâmetros...")
    comparison = compare_datasets(params_20, params_8)

    print("\n  Análise da Influência da Resolução:")
    print(f"  {'─'*50}")
    print(f"\n  Mean Excess Delay:")
    print(f"    20 taps: {comparison['mean_excess_delay']['20_taps']:.2f} ns")
    print(f"    8 taps:  {comparison['mean_excess_delay']['8_taps']:.2f} ns")
    print(f"    Diferença: {comparison['mean_excess_delay']['diff_ns']:.2f} ns ({comparison['mean_excess_delay']['diff_percent']:.1f}%)")

    print(f"\n  RMS Delay Spread:")
    print(f"    20 taps: {comparison['rms_delay_spread']['20_taps']:.2f} ns")
    print(f"    8 taps:  {comparison['rms_delay_spread']['8_taps']:.2f} ns")
    print(f"    Diferença: {comparison['rms_delay_spread']['diff_ns']:.2f} ns ({comparison['rms_delay_spread']['diff_percent']:.1f}%)")

    print(f"\n  Maximum Excess Delay:")
    print(f"    20 taps: {comparison['maximum_excess_delay']['20_taps']:.2f} ns")
    print(f"    8 taps:  {comparison['maximum_excess_delay']['8_taps']:.2f} ns")
    print(f"    Diferença: {comparison['maximum_excess_delay']['diff_ns']:.2f} ns ({comparison['maximum_excess_delay']['diff_percent']:.1f}%)")

    print(f"\n  {comparison['resolution_effect']['conclusion']}")

    print("\n2. Plotando comparação de PDPs...")
    plot_pdp_comparison(params_20['pdp'], data_20.delays,
                       params_8['pdp'], data_8.delays,
                       'figures/parte2/pdp_comparison.png')

    print("\n3. Plotando comparação de parâmetros...")
    plot_parameters_comparison(comparison,
                              'figures/parte2/parameters_comparison.png')

    print("\n4. Criando animação comparativa...")
    create_pdp_comparison_animation(
        data_20.cirs, data_20.delays,
        data_8.cirs, data_8.delays,
        n_frames=100,
        output_path='media/cir_comparison_20vs8.gif',
        fps=10
    )

    # Salvar resultados
    save_results_to_csv(params_20, params_8, comparison)

    print("\n" + "="*70)
    print("✓ PARTE II CONCLUÍDA COM SUCESSO!")
    print("="*70 + "\n")

    print("Arquivos gerados:")
    print("  Figuras:")
    print("    - figures/parte2/pdp_20taps.png")
    print("    - figures/parte2/pdp_8taps.png")
    print("    - figures/parte2/pdp_comparison.png")
    print("    - figures/parte2/parameters_comparison.png")
    print("  Animações:")
    print("    - media/cir_animation_20taps.gif")
    print("    - media/cir_animation_8taps.gif")
    print("    - media/cir_comparison_20vs8.gif")
    print("  Resultados:")
    print("    - results/parte2_parametros.csv")
    print("    - results/parte2_comparacao.csv")


if __name__ == "__main__":
    main()
