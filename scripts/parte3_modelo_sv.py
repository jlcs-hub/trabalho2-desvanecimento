"""
Script para Parte III: Modelo Saleh-Valenzuela
Gera CIRs sintéticas e demonstra efeito de cada parâmetro

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
from typing import List, Dict

from src.saleh_valenzuela import (
    SalehValenzuelaModel,
    vary_parameter_study,
    get_default_parameters,
    compare_nlos_vs_los
)

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def plot_cir_examples(delays: np.ndarray, cirs: np.ndarray,
                     titles: List[str], output_path: str,
                     main_title: str = 'Exemplos de CIRs Geradas'):
    """
    Plota múltiplas CIRs em subplots

    Parameters
    ----------
    delays : np.ndarray
        Array de delays
    cirs : np.ndarray
        Array 2D com CIRs (cada linha = 1 CIR)
    titles : list
        Títulos para cada subplot
    output_path : str
        Caminho para salvar figura
    main_title : str
        Título principal da figura
    """
    n_cirs = cirs.shape[0]
    n_cols = 2
    n_rows = int(np.ceil(n_cirs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_cirs > 1 else [axes]

    for idx in range(n_cirs):
        ax = axes[idx]
        ax.stem(delays, cirs[idx], basefmt=' ', linefmt='b-', markerfmt='bo')
        ax.set_xlabel('Delay (ns)', fontsize=11)
        ax.set_ylabel('Potência Normalizada', fontsize=11)
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Remover axes extras
    for idx in range(n_cirs, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def plot_parameter_variation(results: Dict, output_path: str):
    """
    Plota efeito da variação de um parâmetro

    Parameters
    ----------
    results : dict
        Resultados do vary_parameter_study
    output_path : str
        Caminho para salvar
    """
    param_name = results['param_name']
    param_values = results['param_values']
    delays = results['delays']
    cirs = results['cirs']

    n_values = len(param_values)

    fig, axes = plt.subplots(n_values, 2, figsize=(16, 4*n_values))

    if n_values == 1:
        axes = axes.reshape(1, -1)

    for idx, (value, cir) in enumerate(zip(param_values, cirs)):
        # Subplot 1: CIR (stem)
        ax1 = axes[idx, 0]
        ax1.stem(delays, cir, basefmt=' ', linefmt='b-', markerfmt='bo')
        ax1.set_xlabel('Delay (ns)', fontsize=11)
        ax1.set_ylabel('Potência Normalizada', fontsize=11)
        ax1.set_title(f'{param_name} = {value}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: CIR em escala dB
        ax2 = axes[idx, 1]
        cir_db = 10 * np.log10(cir + 1e-10)  # Evitar log(0)
        ax2.plot(delays, cir_db, 'r-', linewidth=1.5)
        ax2.set_xlabel('Delay (ns)', fontsize=11)
        ax2.set_ylabel('Potência (dB)', fontsize=11)
        ax2.set_title(f'{param_name} = {value} (escala dB)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Efeito da Variação de {param_name}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def plot_all_parameters_comparison(base_params: Dict, output_path: str):
    """
    Plota comparação de todos os parâmetros em uma figura

    Parameters
    ----------
    base_params : dict
        Parâmetros base
    output_path : str
        Caminho para salvar
    """
    # Definir variações para cada parâmetro
    variations = {
        'Gamma': [0.01, 0.02, 0.05],
        'gamma': [0.2, 0.4, 0.8],
        'Lambda': [0.02, 0.04, 0.08],
        'lambda_': [0.25, 0.5, 1.0]
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    param_names = list(variations.keys())
    greek_names = ['Γ (Taxa clusters)', 'γ (Taxa raios)',
                  'Λ (Decaimento clusters)', 'λ (Decaimento raios)']

    for idx, (param_name, greek_name) in enumerate(zip(param_names, greek_names)):
        ax = axes[idx]

        values = variations[param_name]

        for value in values:
            # Criar modelo com parâmetro variado
            params = base_params.copy()
            params[param_name] = value

            model = SalehValenzuelaModel(**params)
            delays, cir = model.generate_cir(t_max=300, dt=2.0, random_seed=42)

            # Plotar
            ax.plot(delays, cir, linewidth=2, label=f'{param_name}={value}', alpha=0.7)

        ax.set_xlabel('Delay (ns)', fontsize=11)
        ax.set_ylabel('Potência Normalizada', fontsize=11)
        ax.set_title(f'Variação de {greek_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Comparação: Efeito de Todos os Parâmetros do Modelo SV',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def plot_nlos_vs_los(results: Dict, output_path: str):
    """
    Plota comparação NLOS vs LOS

    Parameters
    ----------
    results : dict
        Resultados do compare_nlos_vs_los
    output_path : str
        Caminho para salvar
    """
    k_values = results['param_values']
    delays = results['delays']
    cirs = results['cirs']

    fig, axes = plt.subplots(len(k_values), 2, figsize=(16, 4*len(k_values)))

    if len(k_values) == 1:
        axes = axes.reshape(1, -1)

    for idx, (k, cir) in enumerate(zip(k_values, cirs)):
        # Subplot 1: Linear
        ax1 = axes[idx, 0]
        ax1.stem(delays, cir, basefmt=' ', linefmt='b-', markerfmt='bo')
        ax1.set_xlabel('Delay (ns)', fontsize=11)
        ax1.set_ylabel('Potência Normalizada', fontsize=11)

        if k == 0:
            title = f'K = {k} (NLOS - Sem componente LOS)'
        else:
            title = f'K = {k} (LOS - Com componente direta)'

        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: dB
        ax2 = axes[idx, 1]
        cir_db = 10 * np.log10(cir + 1e-10)
        ax2.plot(delays, cir_db, 'r-', linewidth=1.5)
        ax2.set_xlabel('Delay (ns)', fontsize=11)
        ax2.set_ylabel('Potência (dB)', fontsize=11)
        ax2.set_title(f'{title} (escala dB)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    fig.suptitle('Comparação: NLOS vs LOS (Variando Fator K)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: {output_path}")


def save_generated_cirs(delays: np.ndarray, cirs: np.ndarray,
                       labels: List[str], output_path: str):
    """Salva CIRs geradas em CSV"""
    df = pd.DataFrame(cirs.T, columns=labels)
    df.insert(0, 'delay_ns', delays)
    df.to_csv(output_path, index=False)
    print(f"  → Salvo: {output_path}")


def main():
    """Função principal"""
    print("\n" + "="*70)
    print("PARTE III: MODELO SALEH-VALENZUELA")
    print("="*70 + "\n")

    # Parâmetros base
    print("Configurando parâmetros base do modelo...")
    base_params = get_default_parameters()

    print(f"\nParâmetros base:")
    print(f"  {'─'*50}")
    for key, value in base_params.items():
        print(f"  {key:10s} = {value}")

    # ========================================================================
    # 1. EXEMPLOS COM DIFERENTES CONFIGURAÇÕES
    # ========================================================================
    print("\n" + "="*70)
    print("1. GERANDO EXEMPLOS COM DIFERENTES CONFIGURAÇÕES")
    print("="*70 + "\n")

    configs = [
        {'Gamma': 0.02, 'gamma': 0.4, 'Lambda': 0.04, 'lambda_': 0.5, 'k_factor': 0},
        {'Gamma': 0.05, 'gamma': 0.8, 'Lambda': 0.08, 'lambda_': 1.0, 'k_factor': 0},
        {'Gamma': 0.01, 'gamma': 0.2, 'Lambda': 0.02, 'lambda_': 0.25, 'k_factor': 0},
        {'Gamma': 0.02, 'gamma': 0.4, 'Lambda': 0.04, 'lambda_': 0.5, 'k_factor': 3},
    ]

    titles = [
        'Config 1: Padrão (NLOS)',
        'Config 2: Clusters/raios rápidos, decaimento rápido',
        'Config 3: Clusters/raios lentos, decaimento lento',
        'Config 4: Padrão com LOS (K=3)'
    ]

    cirs_examples = []
    for config, title in zip(configs, titles):
        print(f"  Gerando: {title}...")
        model = SalehValenzuelaModel(**{**base_params, **config})
        delays, cir = model.generate_cir(t_max=300, dt=2.0, random_seed=42)
        cirs_examples.append(cir)

    cirs_examples = np.array(cirs_examples)

    print("\n  Plotando exemplos...")
    plot_cir_examples(delays, cirs_examples, titles,
                     'figures/parte3/sv_exemplos_configs.png',
                     'Exemplos de CIRs com Diferentes Configurações')

    # Salvar
    save_generated_cirs(delays, cirs_examples, titles,
                       'results/parte3_exemplos_configs.csv')

    # ========================================================================
    # 2. VARIAÇÃO INDIVIDUAL DE CADA PARÂMETRO
    # ========================================================================
    print("\n" + "="*70)
    print("2. ESTUDANDO EFEITO DE CADA PARÂMETRO INDIVIDUALMENTE")
    print("="*70 + "\n")

    # Gamma
    print("  2.1. Variando Gamma (taxa de chegada de clusters)...")
    results_gamma = vary_parameter_study(base_params, 'Gamma',
                                        [0.01, 0.02, 0.05, 0.1])
    plot_parameter_variation(results_gamma,
                            'figures/parte3/sv_variacao_Gamma_clusters.png')

    # gamma
    print("  2.2. Variando gamma (taxa de chegada de raios)...")
    results_gamma_lower = vary_parameter_study(base_params, 'gamma',
                                              [0.2, 0.4, 0.8, 1.5])
    plot_parameter_variation(results_gamma_lower,
                            'figures/parte3/sv_variacao_gamma_raios.png')

    # Lambda
    print("  2.3. Variando Lambda (decaimento de clusters)...")
    results_lambda_cap = vary_parameter_study(base_params, 'Lambda',
                                             [0.02, 0.04, 0.08, 0.15])
    plot_parameter_variation(results_lambda_cap,
                            'figures/parte3/sv_variacao_Lambda_clusters.png')

    # lambda_
    print("  2.4. Variando lambda (decaimento de raios)...")
    results_lambda = vary_parameter_study(base_params, 'lambda_',
                                         [0.25, 0.5, 1.0, 2.0])
    plot_parameter_variation(results_lambda,
                            'figures/parte3/sv_variacao_lambda_raios.png')

    # ========================================================================
    # 3. COMPARAÇÃO DE TODOS OS PARÂMETROS
    # ========================================================================
    print("\n" + "="*70)
    print("3. COMPARAÇÃO: EFEITO DE TODOS OS PARÂMETROS")
    print("="*70 + "\n")

    print("  Gerando figura comparativa...")
    plot_all_parameters_comparison(base_params,
                                  'figures/parte3/sv_comparacao_todos_parametros.png')

    # ========================================================================
    # 4. NLOS vs LOS
    # ========================================================================
    print("\n" + "="*70)
    print("4. COMPARAÇÃO: NLOS vs LOS (Variando Fator K)")
    print("="*70 + "\n")

    print("  Gerando CIRs para diferentes valores de K...")
    results_k = compare_nlos_vs_los(base_params, k_values=[0, 1, 3, 10])

    print("  Plotando comparação NLOS vs LOS...")
    plot_nlos_vs_los(results_k, 'figures/parte3/sv_nlos_vs_los.png')

    # ========================================================================
    # 5. MÚLTIPLAS REALIZAÇÕES
    # ========================================================================
    print("\n" + "="*70)
    print("5. GERANDO MÚLTIPLAS REALIZAÇÕES")
    print("="*70 + "\n")

    print("  Gerando 100 CIRs com parâmetros padrão...")
    model_default = SalehValenzuelaModel(**base_params)
    delays_multi, cirs_multi = model_default.generate_multiple_cirs(
        n_cirs=100, t_max=300, dt=2.0
    )

    # Calcular PDP médio
    pdp = np.mean(cirs_multi, axis=0)

    print(f"  ✓ {cirs_multi.shape[0]} CIRs geradas")

    # Plotar
    print("  Plotando PDP médio e algumas realizações...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Algumas CIRs + PDP médio
    for i in range(0, 100, 20):
        ax1.plot(delays_multi, cirs_multi[i], 'b-', alpha=0.3, linewidth=0.5)

    ax1.plot(delays_multi, pdp, 'r-', linewidth=3, label='PDP Médio')
    ax1.set_xlabel('Delay (ns)', fontsize=12)
    ax1.set_ylabel('Potência Normalizada', fontsize=12)
    ax1.set_title('100 Realizações do Modelo SV', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: PDP em dB
    pdp_db = 10 * np.log10(pdp + 1e-10)
    ax2.plot(delays_multi, pdp_db, 'r-', linewidth=2)
    ax2.set_xlabel('Delay (ns)', fontsize=12)
    ax2.set_ylabel('Potência (dB)', fontsize=12)
    ax2.set_title('PDP Médio (escala dB)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/parte3/sv_multiplas_realizacoes.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  → Salvo: figures/parte3/sv_multiplas_realizacoes.png")

    # Salvar PDP
    df_pdp = pd.DataFrame({
        'delay_ns': delays_multi,
        'pdp_mean': pdp,
        'pdp_std': np.std(cirs_multi, axis=0)
    })
    df_pdp.to_csv('results/parte3_pdp_medio.csv', index=False)
    print(f"  → Salvo: results/parte3_pdp_medio.csv")

    # ========================================================================
    # RESUMO
    # ========================================================================
    print("\n" + "="*70)
    print("✓ PARTE III CONCLUÍDA COM SUCESSO!")
    print("="*70 + "\n")

    print("Arquivos gerados:")
    print("  Figuras:")
    print("    - figures/parte3/sv_exemplos_configs.png")
    print("    - figures/parte3/sv_variacao_Gamma_clusters.png")
    print("    - figures/parte3/sv_variacao_gamma_raios.png")
    print("    - figures/parte3/sv_variacao_Lambda_clusters.png")
    print("    - figures/parte3/sv_variacao_lambda_raios.png")
    print("    - figures/parte3/sv_comparacao_todos_parametros.png")
    print("    - figures/parte3/sv_nlos_vs_los.png")
    print("    - figures/parte3/sv_multiplas_realizacoes.png")
    print("  Resultados:")
    print("    - results/parte3_exemplos_configs.csv")
    print("    - results/parte3_pdp_medio.csv")


if __name__ == "__main__":
    main()
