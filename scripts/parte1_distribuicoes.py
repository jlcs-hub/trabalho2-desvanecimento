"""
Script para Parte I: Análise de Distribuições Estatísticas
Processa dados CIR e ajusta distribuições Rayleigh, Rice e Nakagami

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
from scipy import stats

from src.cir_loader import load_both_datasets, CIRData
from src.statistical_distributions import (
    compute_total_power_signal,
    detect_mean_variation,
    filter_to_zero_mean,
    analyze_signal_distributions,
    normalize_envelope
)

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_histogram(data: np.ndarray, title: str, output_path: str, bins: int = 50):
    """Plota histograma dos dados filtrados"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Densidade de Probabilidade')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def plot_cdf_comparison(data: np.ndarray, distributions: dict,
                       title: str, output_path: str):
    """Plota CDF empírica vs distribuições ajustadas"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # CDF empírica
    data_sorted = np.sort(data)
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data_sorted, empirical_cdf, 'k-', linewidth=2,
            label='Empírica (Dados)', alpha=0.8)

    # CDFs teóricas
    x = np.linspace(0, np.max(data), 1000)

    # Rayleigh
    rayleigh_params = distributions['rayleigh']
    rayleigh_cdf = stats.rayleigh.cdf(x, scale=rayleigh_params['sigma'])
    ax.plot(x, rayleigh_cdf, 'r--', linewidth=2,
            label=f"Rayleigh (σ={rayleigh_params['sigma']:.3f})")

    # Rice
    rice_params = distributions['rice']
    b = rice_params['b']
    scale = rice_params['scale']
    rice_cdf = stats.rice.cdf(x, b, scale=scale)
    ax.plot(x, rice_cdf, 'g--', linewidth=2,
            label=f"Rice (K={rice_params['K']:.3f}, Ω={rice_params['omega']:.3f})")

    # Nakagami
    nak_params = distributions['nakagami']
    m = nak_params['m']
    scale_nak = nak_params['scale']
    nak_cdf = stats.nakagami.cdf(x, m, scale=scale_nak)
    ax.plot(x, nak_cdf, 'b--', linewidth=2,
            label=f"Nakagami (m={nak_params['m']:.3f}, Ω={nak_params['omega']:.3f})")

    ax.set_xlabel('Amplitude')
    ax.set_ylabel('CDF')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def plot_normalized_envelope_cdf(data: np.ndarray, distributions: dict,
                                 title: str, output_path: str):
    """Plota CDF da envoltória normalizada vs distribuições"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Normalizar envoltória
    rho = normalize_envelope(np.abs(data))

    # CDF empírica
    rho_sorted = np.sort(rho)
    empirical_cdf = np.arange(1, len(rho) + 1) / len(rho)
    ax.plot(rho_sorted, empirical_cdf, 'k-', linewidth=2,
            label='Empírica (Envoltória Normalizada ρ)', alpha=0.8)

    # CDFs teóricas normalizadas
    x = np.linspace(0, np.max(rho), 1000)

    # Rayleigh normalizado
    rayleigh_params = distributions['rayleigh']
    sigma_norm = rayleigh_params['sigma']
    rayleigh_cdf = stats.rayleigh.cdf(x, scale=sigma_norm)
    ax.plot(x, rayleigh_cdf, 'r--', linewidth=2,
            label=f"Rayleigh (σ={sigma_norm:.3f})")

    # Rice normalizado
    rice_params = distributions['rice']
    K_norm = rice_params['K']
    omega_norm = rice_params['omega']
    # Converter para parâmetros scipy
    b_norm = np.sqrt(2 * K_norm)
    scale_norm = np.sqrt(omega_norm / (1 + b_norm**2))
    rice_cdf = stats.rice.cdf(x, b_norm, scale=scale_norm)
    ax.plot(x, rice_cdf, 'g--', linewidth=2,
            label=f"Rice (K={K_norm:.3f}, Ω={omega_norm:.3f})")

    # Nakagami normalizado
    nak_params = distributions['nakagami']
    m_norm = nak_params['m']
    omega_nak_norm = nak_params['omega']
    scale_nak_norm = np.sqrt(omega_nak_norm)
    nak_cdf = stats.nakagami.cdf(x, m_norm, scale=scale_nak_norm)
    ax.plot(x, nak_cdf, 'b--', linewidth=2,
            label=f"Nakagami (m={m_norm:.3f}, Ω={omega_nak_norm:.3f})")

    ax.set_xlabel('Envoltória Normalizada ρ')
    ax.set_ylabel('CDF')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def process_dataset(cir_data: CIRData, n_taps: int, filter_method: str = 'remove_mean'):
    """
    Processa um dataset CIR completo

    Parameters
    ----------
    cir_data : CIRData
        Dados CIR
    n_taps : int
        Número de taps (para nomear arquivos)
    filter_method : str
        Método de filtragem
    """
    print(f"\n{'='*70}")
    print(f"PROCESSANDO DATASET: {n_taps} TAPS")
    print(f"{'='*70}\n")

    # 1. Computar sinal de potência total
    print("1. Computando sinal de potência total (soma de todos os taps)...")
    signal_power = compute_total_power_signal(cir_data.cirs)
    print(f"  - Sinal: {len(signal_power)} amostras")
    print(f"  - Média: {np.mean(signal_power):.4f}")
    print(f"  - Std: {np.std(signal_power):.4f}")

    # 2. Detectar variação da média
    print("\n2. Detectando variação da média (perda de percurso/sombreamento)...")
    has_variation, mean_std = detect_mean_variation(signal_power)
    print(f"  - Variação detectada: {'SIM' if has_variation else 'NÃO'}")
    print(f"  - Std normalizado da média móvel: {mean_std:.4f}")

    # 3. Filtrar sinal
    print(f"\n3. Filtrando sinal (método: {filter_method})...")
    filtered_signal = filter_to_zero_mean(signal_power, method=filter_method)
    print(f"  - Sinal filtrado - Média: {np.mean(filtered_signal):.6f}")
    print(f"  - Sinal filtrado - Std: {np.std(filtered_signal):.4f}")

    # 4. Plotar histograma
    print("\n4. Gerando histograma das medidas filtradas...")
    envelope = np.abs(filtered_signal)
    plot_histogram(
        envelope,
        f'Histograma de Amplitude Filtrada ({n_taps} taps)',
        f'figures/parte1/histograma_{n_taps}taps.png'
    )

    # 5. Análise completa de distribuições
    print("\n5. Ajustando distribuições...")
    results = analyze_signal_distributions(signal_power, filter_method=filter_method)

    # Resultados - Método scipy
    print("\n  MÉTODO 1: scipy.stats.fit")
    print("  " + "-"*50)
    scipy_results = results['scipy_fit']

    print(f"  Rayleigh:")
    print(f"    σ = {scipy_results['rayleigh']['sigma']:.4f}")
    print(f"    SSE = {scipy_results['rayleigh']['sse']:.6f}")

    print(f"\n  Rice:")
    print(f"    K = {scipy_results['rice']['K']:.4f}")
    print(f"    Ω = {scipy_results['rice']['omega']:.4f}")
    print(f"    SSE = {scipy_results['rice']['sse']:.6f}")

    print(f"\n  Nakagami:")
    print(f"    m = {scipy_results['nakagami']['m']:.4f}")
    print(f"    Ω = {scipy_results['nakagami']['omega']:.4f}")
    print(f"    SSE = {scipy_results['nakagami']['sse']:.6f}")

    # Resultados - Estimadores dissertação
    print("\n  MÉTODO 2: Estimadores da Dissertação (Envoltória Normalizada)")
    print("  " + "-"*50)
    diss_results = results['dissertation_estimators']

    print(f"  Rayleigh:")
    print(f"    σ = {diss_results['rayleigh']['sigma']:.4f}")

    print(f"\n  Rice:")
    print(f"    K = {diss_results['rice']['K']:.4f}")
    print(f"    Ω = {diss_results['rice']['omega']:.4f}")

    print(f"\n  Nakagami:")
    print(f"    m = {diss_results['nakagami']['m']:.4f}")
    print(f"    Ω = {diss_results['nakagami']['omega']:.4f}")

    # 6. Plotar CDFs - Método 1
    print("\n6. Gerando gráficos de CDF...")
    plot_cdf_comparison(
        envelope,
        scipy_results,
        f'CDF: Dados vs Distribuições Ajustadas ({n_taps} taps)',
        f'figures/parte1/cdf_scipy_{n_taps}taps.png'
    )

    # 7. Plotar CDFs - Método 2 (Envoltória normalizada)
    plot_normalized_envelope_cdf(
        filtered_signal,
        diss_results,
        f'CDF: Envoltória Normalizada ρ ({n_taps} taps)',
        f'figures/parte1/cdf_normalizada_{n_taps}taps.png'
    )

    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO CONCLUÍDO: {n_taps} TAPS")
    print(f"{'='*70}\n")

    return results


def save_results_to_csv(results_20: dict, results_8: dict):
    """Salva resultados em CSV para comparação"""
    import pandas as pd

    # Extrair parâmetros
    data = []

    # 20 taps - scipy
    data.append({
        'dataset': '20 taps',
        'metodo': 'scipy.stats',
        'distribuicao': 'Rayleigh',
        'parametro_1': results_20['scipy_fit']['rayleigh']['sigma'],
        'parametro_2': np.nan,
        'sse': results_20['scipy_fit']['rayleigh']['sse']
    })

    data.append({
        'dataset': '20 taps',
        'metodo': 'scipy.stats',
        'distribuicao': 'Rice',
        'parametro_1': results_20['scipy_fit']['rice']['K'],
        'parametro_2': results_20['scipy_fit']['rice']['omega'],
        'sse': results_20['scipy_fit']['rice']['sse']
    })

    data.append({
        'dataset': '20 taps',
        'metodo': 'scipy.stats',
        'distribuicao': 'Nakagami',
        'parametro_1': results_20['scipy_fit']['nakagami']['m'],
        'parametro_2': results_20['scipy_fit']['nakagami']['omega'],
        'sse': results_20['scipy_fit']['nakagami']['sse']
    })

    # 20 taps - dissertação
    data.append({
        'dataset': '20 taps',
        'metodo': 'dissertacao',
        'distribuicao': 'Rayleigh',
        'parametro_1': results_20['dissertation_estimators']['rayleigh']['sigma'],
        'parametro_2': np.nan,
        'sse': np.nan
    })

    data.append({
        'dataset': '20 taps',
        'metodo': 'dissertacao',
        'distribuicao': 'Rice',
        'parametro_1': results_20['dissertation_estimators']['rice']['K'],
        'parametro_2': results_20['dissertation_estimators']['rice']['omega'],
        'sse': np.nan
    })

    data.append({
        'dataset': '20 taps',
        'metodo': 'dissertacao',
        'distribuicao': 'Nakagami',
        'parametro_1': results_20['dissertation_estimators']['nakagami']['m'],
        'parametro_2': results_20['dissertation_estimators']['nakagami']['omega'],
        'sse': np.nan
    })

    # Repetir para 8 taps
    data.append({
        'dataset': '8 taps',
        'metodo': 'scipy.stats',
        'distribuicao': 'Rayleigh',
        'parametro_1': results_8['scipy_fit']['rayleigh']['sigma'],
        'parametro_2': np.nan,
        'sse': results_8['scipy_fit']['rayleigh']['sse']
    })

    data.append({
        'dataset': '8 taps',
        'metodo': 'scipy.stats',
        'distribuicao': 'Rice',
        'parametro_1': results_8['scipy_fit']['rice']['K'],
        'parametro_2': results_8['scipy_fit']['rice']['omega'],
        'sse': results_8['scipy_fit']['rice']['sse']
    })

    data.append({
        'dataset': '8 taps',
        'metodo': 'scipy.stats',
        'distribuicao': 'Nakagami',
        'parametro_1': results_8['scipy_fit']['nakagami']['m'],
        'parametro_2': results_8['scipy_fit']['nakagami']['omega'],
        'sse': results_8['scipy_fit']['nakagami']['sse']
    })

    data.append({
        'dataset': '8 taps',
        'metodo': 'dissertacao',
        'distribuicao': 'Rayleigh',
        'parametro_1': results_8['dissertation_estimators']['rayleigh']['sigma'],
        'parametro_2': np.nan,
        'sse': np.nan
    })

    data.append({
        'dataset': '8 taps',
        'metodo': 'dissertacao',
        'distribuicao': 'Rice',
        'parametro_1': results_8['dissertation_estimators']['rice']['K'],
        'parametro_2': results_8['dissertation_estimators']['rice']['omega'],
        'sse': np.nan
    })

    data.append({
        'dataset': '8 taps',
        'metodo': 'dissertacao',
        'distribuicao': 'Nakagami',
        'parametro_1': results_8['dissertation_estimators']['nakagami']['m'],
        'parametro_2': results_8['dissertation_estimators']['nakagami']['omega'],
        'sse': np.nan
    })

    df = pd.DataFrame(data)
    output_path = 'results/parte1_parametros.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Resultados salvos em: {output_path}")


def main():
    """Função principal"""
    print("\n" + "="*70)
    print("PARTE I: ANÁLISE DE DISTRIBUIÇÕES ESTATÍSTICAS")
    print("="*70 + "\n")

    # Carregar dados
    print("Carregando datasets CIR...")
    data_20, data_8 = load_both_datasets()
    print(f"  ✓ Dataset 20 taps: {data_20.n_cirs} CIRs")
    print(f"  ✓ Dataset 8 taps: {data_8.n_cirs} CIRs")

    # Processar ambos datasets
    results_20 = process_dataset(data_20, n_taps=20)
    results_8 = process_dataset(data_8, n_taps=8)

    # Salvar resultados
    save_results_to_csv(results_20, results_8)

    print("\n" + "="*70)
    print("✓ PARTE I CONCLUÍDA COM SUCESSO!")
    print("="*70 + "\n")

    print("Arquivos gerados:")
    print("  - figures/parte1/histograma_20taps.png")
    print("  - figures/parte1/cdf_scipy_20taps.png")
    print("  - figures/parte1/cdf_normalizada_20taps.png")
    print("  - figures/parte1/histograma_8taps.png")
    print("  - figures/parte1/cdf_scipy_8taps.png")
    print("  - figures/parte1/cdf_normalizada_8taps.png")
    print("  - results/parte1_parametros.csv")


if __name__ == "__main__":
    main()
