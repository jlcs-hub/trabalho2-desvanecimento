"""
Módulo para visualização de dados de propagação.

Este módulo contém funções para criar gráficos e visualizações dos dados
de medição e resultados de modelagem.

Autor: João Lucas de Castro Santos
Data: 2025-11-05
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_received_power_vs_distance(distances: np.ndarray, powers: np.ndarray,
                                    title: str = "Potência Recebida vs Distância",
                                    save_path: Optional[str] = None) -> None:
    """
    Plota potência recebida em função da distância.

    Args:
        distances: Array de distâncias (metros)
        powers: Array de potências (dBm)
        title: Título do gráfico
        save_path: Caminho para salvar figura (opcional)

    Example:
        >>> plot_received_power_vs_distance(dist, pow, save_path='figures/power_vs_dist.png')

    TODO: IMPLEMENTAR A LÓGICA
    - Criar scatter plot com plt.scatter()
    - Adicionar labels, título, grid
    - Salvar se save_path fornecido
    """
    # IMPLEMENTAÇÃO AQUI
    # plt.figure(figsize=(10, 6))
    # plt.scatter(distances, powers, alpha=0.6, s=50)
    # plt.xlabel('Distância (m)')
    # plt.ylabel('Potência Recebida (dBm)')
    # plt.title(title)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def plot_power_vs_log_distance(distances: np.ndarray, powers: np.ndarray,
                               d0: float = 1.0,
                               title: str = "Potência vs log(d)",
                               save_path: Optional[str] = None) -> None:
    """
    Plota potência recebida em função de log10(distância).

    Este é o gráfico fundamental para análise do modelo log-distância,
    onde esperamos uma relação linear.

    Args:
        distances: Array de distâncias (metros)
        powers: Array de potências (dBm)
        d0: Distância de referência
        title: Título do gráfico
        save_path: Caminho para salvar figura

    Example:
        >>> plot_power_vs_log_distance(dist, pow, d0=1.0)

    TODO: IMPLEMENTAR A LÓGICA
    - Calcular log10(d/d0)
    - Plotar potência vs log10(d/d0)
    - Se linearidade boa, indica modelo log-distância apropriado
    """
    # IMPLEMENTAÇÃO AQUI
    # log_dist = np.log10(distances / d0)
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(log_dist, powers, alpha=0.6, s=50, label='Medições')
    # plt.xlabel('log₁₀(d/d₀)')
    # plt.ylabel('Potência Recebida (dBm)')
    # plt.title(title)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def plot_log_distance_fit(distances: np.ndarray, measured_power: np.ndarray,
                          predicted_power: np.ndarray, n: float, r_squared: float,
                          title: str = "Ajuste do Modelo Log-Distância",
                          save_path: Optional[str] = None) -> None:
    """
    Plota dados medidos e ajuste do modelo log-distância.

    Args:
        distances: Distâncias medidas
        measured_power: Potências medidas
        predicted_power: Potências preditas pelo modelo
        n: Índice de decaimento
        r_squared: Coeficiente de determinação
        title: Título
        save_path: Caminho para salvar

    Example:
        >>> plot_log_distance_fit(dist, measured, predicted, n=3.5, r_squared=0.95)

    TODO: IMPLEMENTAR A LÓGICA
    - Plotar pontos medidos (scatter)
    - Plotar linha do modelo ajustado
    - Adicionar legenda com n e R²
    """
    # IMPLEMENTAÇÃO AQUI
    # plt.figure(figsize=(10, 6))
    # plt.scatter(distances, measured_power, alpha=0.6, s=50,
    #            label='Medições', color='blue')
    # plt.plot(distances, predicted_power, 'r-', linewidth=2,
    #         label=f'Modelo (n={n:.2f}, R²={r_squared:.4f})')
    # plt.xlabel('Distância (m)')
    # plt.ylabel('Potência (dBm)')
    # plt.title(title)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def plot_residuals(distances: np.ndarray, residuals: np.ndarray,
                  title: str = "Resíduos do Modelo",
                  save_path: Optional[str] = None) -> None:
    """
    Plota resíduos do modelo vs distância.

    Útil para verificar se o modelo está bem ajustado.
    Resíduos aleatórios em torno de zero indicam bom ajuste.

    Args:
        distances: Distâncias
        residuals: Resíduos (medido - predito)
        title: Título
        save_path: Caminho para salvar

    Example:
        >>> plot_residuals(dist, residuals)

    TODO: IMPLEMENTAR A LÓGICA
    - Plotar resíduos vs distância
    - Adicionar linha horizontal em y=0
    """
    # IMPLEMENTAÇÃO AQUI
    # plt.figure(figsize=(10, 6))
    # plt.scatter(distances, residuals, alpha=0.6, s=50)
    # plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    # plt.xlabel('Distância (m)')
    # plt.ylabel('Resíduo (dB)')
    # plt.title(title)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def plot_shadowing_histogram(shadowing: np.ndarray, sigma: float,
                             title: str = "Histograma do Sombreamento",
                             save_path: Optional[str] = None) -> None:
    """
    Plota histograma do sombreamento com curva normal sobreposta.

    Args:
        shadowing: Array com valores de sombreamento
        sigma: Desvio-padrão
        title: Título
        save_path: Caminho para salvar

    Example:
        >>> plot_shadowing_histogram(shadowing, sigma=6.0)

    TODO: IMPLEMENTAR A LÓGICA
    - Plotar histograma normalizado
    - Sobrepor curva da distribuição normal N(0, σ²)
    - Verificar se dados seguem distribuição normal
    """
    # IMPLEMENTAÇÃO AQUI
    # from scipy import stats
    #
    # plt.figure(figsize=(10, 6))
    #
    # # Histograma
    # counts, bins, patches = plt.hist(shadowing, bins=20, density=True,
    #                                  alpha=0.7, color='skyblue',
    #                                  edgecolor='black', label='Dados')
    #
    # # Curva normal teórica
    # x = np.linspace(shadowing.min(), shadowing.max(), 100)
    # normal_curve = stats.norm.pdf(x, 0, sigma)
    # plt.plot(x, normal_curve, 'r-', linewidth=2,
    #         label=f'Normal(0, {sigma:.2f}²)')
    #
    # plt.xlabel('Sombreamento (dB)')
    # plt.ylabel('Densidade de Probabilidade')
    # plt.title(title)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def plot_shadowing_comparison(distances: np.ndarray,
                              shadowing_uncorr: np.ndarray,
                              shadowing_corr: np.ndarray,
                              title: str = "Comparação: Sombreamento Correlacionado vs Descorrelacionado",
                              save_path: Optional[str] = None) -> None:
    """
    Compara sombreamento correlacionado e descorrelacionado.

    Args:
        distances: Distâncias
        shadowing_uncorr: Sombreamento descorrelacionado
        shadowing_corr: Sombreamento correlacionado
        title: Título
        save_path: Caminho para salvar

    Example:
        >>> plot_shadowing_comparison(dist, uncorr, corr)

    TODO: IMPLEMENTAR A LÓGICA
    - Plotar ambos os sombreamentos vs distância
    - Descorrelacionado deve ter variações mais abruptas
    - Correlacionado deve ter transições mais suaves
    """
    # IMPLEMENTAÇÃO AQUI
    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(distances, shadowing_uncorr, 'o-', alpha=0.6,
    #         label='Descorrelacionado')
    # plt.xlabel('Distância (m)')
    # plt.ylabel('Sombreamento (dB)')
    # plt.title('Descorrelacionado')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(distances, shadowing_corr, 'o-', alpha=0.6,
    #         color='orange', label='Correlacionado')
    # plt.xlabel('Distância (m)')
    # plt.ylabel('Sombreamento (dB)')
    # plt.title('Correlacionado')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    #
    # plt.suptitle(title)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def plot_power_with_shadowing(distances: np.ndarray, base_power: np.ndarray,
                              power_with_uncorr: np.ndarray,
                              power_with_corr: np.ndarray,
                              title: str = "Modelo com Sombreamento",
                              save_path: Optional[str] = None) -> None:
    """
    Plota modelo base e modelos com sombreamento.

    Args:
        distances: Distâncias
        base_power: Potência do modelo base (sem sombreamento)
        power_with_uncorr: Potência com sombreamento descorrelacionado
        power_with_corr: Potência com sombreamento correlacionado
        title: Título
        save_path: Caminho para salvar

    Example:
        >>> plot_power_with_shadowing(dist, base, with_uncorr, with_corr)

    TODO: IMPLEMENTAR A LÓGICA
    - Plotar as três curvas
    - Destacar diferenças entre modelos
    """
    # IMPLEMENTAÇÃO AQUI
    # plt.figure(figsize=(12, 6))
    #
    # plt.plot(distances, base_power, 'k-', linewidth=2,
    #         label='Modelo Base (sem sombreamento)')
    # plt.plot(distances, power_with_uncorr, 'b-', alpha=0.6,
    #         label='Com sombreamento descorrelacionado')
    # plt.plot(distances, power_with_corr, 'r-', alpha=0.6,
    #         label='Com sombreamento correlacionado')
    #
    # plt.xlabel('Distância (m)')
    # plt.ylabel('Potência (dBm)')
    # plt.title(title)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def plot_multiple_realizations(distances: np.ndarray, base_power: np.ndarray,
                               num_realizations: int, sigma: float,
                               correlation: bool = False,
                               decorr_dist: float = 50.0,
                               title: str = "Múltiplas Realizações",
                               save_path: Optional[str] = None) -> None:
    """
    Plota múltiplas realizações de sombreamento.

    Útil para mostrar variabilidade estatística do modelo.

    Args:
        distances: Distâncias
        base_power: Potência base
        num_realizations: Número de realizações
        sigma: Desvio-padrão do sombreamento
        correlation: Se True, usa sombreamento correlacionado
        decorr_dist: Distância de descorrelação (se correlation=True)
        title: Título
        save_path: Caminho para salvar

    Example:
        >>> plot_multiple_realizations(dist, base, num_realizations=10, sigma=6.0)

    TODO: IMPLEMENTAR A LÓGICA
    - Gerar múltiplas realizações de sombreamento
    - Plotar todas com transparência
    - Plotar modelo base em destaque
    """
    # IMPLEMENTAÇÃO AQUI
    # from .shadowing_analysis import (generate_uncorrelated_shadowing,
    #                                  generate_correlated_shadowing)
    #
    # plt.figure(figsize=(12, 6))
    #
    # # Plotar realizações
    # for i in range(num_realizations):
    #     if correlation:
    #         shadowing = generate_correlated_shadowing(sigma, distances,
    #                                                   decorr_dist, seed=i)
    #     else:
    #         shadowing = generate_uncorrelated_shadowing(sigma,
    #                                                     len(distances), seed=i)
    #     power = base_power + shadowing
    #     plt.plot(distances, power, alpha=0.3, linewidth=1)
    #
    # # Plotar modelo base
    # plt.plot(distances, base_power, 'k-', linewidth=3, label='Modelo Base')
    #
    # plt.xlabel('Distância (m)')
    # plt.ylabel('Potência (dBm)')
    # plt.title(title)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def plot_qq_plot(shadowing: np.ndarray,
                title: str = "Q-Q Plot - Teste de Normalidade",
                save_path: Optional[str] = None) -> None:
    """
    Cria Q-Q plot para testar normalidade do sombreamento.

    Args:
        shadowing: Valores de sombreamento
        title: Título
        save_path: Caminho para salvar

    Example:
        >>> plot_qq_plot(shadowing)

    TODO: IMPLEMENTAR A LÓGICA
    - Usar scipy.stats.probplot() para criar Q-Q plot
    - Se pontos próximos da linha, dados são normais
    """
    # IMPLEMENTAÇÃO AQUI
    # from scipy import stats
    #
    # plt.figure(figsize=(8, 8))
    # stats.probplot(shadowing, dist="norm", plot=plt)
    # plt.title(title)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    #
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    pass


def create_comprehensive_report(distances: np.ndarray, measured: np.ndarray,
                                predicted: np.ndarray, n: float, r_squared: float,
                                shadowing: np.ndarray, sigma: float,
                                save_dir: str = 'figures/') -> None:
    """
    Cria relatório visual completo com múltiplos gráficos.

    Args:
        distances: Distâncias
        measured: Potências medidas
        predicted: Potências preditas
        n: Índice de decaimento
        r_squared: R²
        shadowing: Sombreamento
        sigma: Desvio-padrão
        save_dir: Diretório para salvar figuras

    TODO: IMPLEMENTAR A LÓGICA
    - Criar figura com subplots
    - Gerar todos os gráficos principais
    """
    # IMPLEMENTAÇÃO AQUI
    pass


if __name__ == "__main__":
    print("Módulo visualization.py - Teste")
    print("Implemente as funções acima para criar visualizações")
    print("\nFunções disponíveis:")
    print("- plot_received_power_vs_distance()")
    print("- plot_power_vs_log_distance()")
    print("- plot_log_distance_fit()")
    print("- plot_shadowing_histogram()")
    print("- plot_shadowing_comparison()")
    print("- E outras...")
