"""
Módulo para cálculo de parâmetros de caracterização do canal
Parte II do trabalho: PDP, RMS Delay Spread, Mean/Maximum Excess Delay

Autor: João Lucas de Castro Santos
Data: 2025-11-05
"""

import numpy as np
from typing import Dict, Tuple


def compute_pdp(cirs: np.ndarray) -> np.ndarray:
    """
    Calcula o Power Delay Profile (PDP)
    PDP é a média de todas as CIRs

    Parameters
    ----------
    cirs : np.ndarray
        Array 2D com CIRs (linhas = CIRs, colunas = taps/delays)

    Returns
    -------
    np.ndarray
        Array 1D com o PDP (potência média em cada delay)
    """
    return np.mean(cirs, axis=0)


def compute_mean_excess_delay(pdp: np.ndarray, delays: np.ndarray) -> float:
    """
    Calcula o Mean Excess Delay
    τ_mean = Σ(P(τ) * τ) / Σ(P(τ))

    Parameters
    ----------
    pdp : np.ndarray
        Power Delay Profile
    delays : np.ndarray
        Array com os delays/atrasos (em ns)

    Returns
    -------
    float
        Mean Excess Delay (em ns)
    """
    # Normalizar PDP (somar = 1)
    pdp_normalized = pdp / np.sum(pdp)

    # Mean = weighted average
    mean_delay = np.sum(pdp_normalized * delays)

    return mean_delay


def compute_rms_delay_spread(pdp: np.ndarray, delays: np.ndarray) -> float:
    """
    Calcula o RMS Delay Spread
    τ_rms = sqrt(Σ(P(τ) * τ²) / Σ(P(τ)) - τ_mean²)

    Parameters
    ----------
    pdp : np.ndarray
        Power Delay Profile
    delays : np.ndarray
        Array com os delays/atrasos (em ns)

    Returns
    -------
    float
        RMS Delay Spread (em ns)
    """
    # Normalizar PDP
    pdp_normalized = pdp / np.sum(pdp)

    # Mean delay
    mean_delay = np.sum(pdp_normalized * delays)

    # Second moment
    second_moment = np.sum(pdp_normalized * (delays ** 2))

    # RMS
    rms = np.sqrt(second_moment - mean_delay ** 2)

    return rms


def compute_maximum_excess_delay(pdp: np.ndarray, delays: np.ndarray,
                                threshold_db: float = -20.0) -> float:
    """
    Calcula o Maximum Excess Delay
    Último delay onde a potência está acima do threshold relativo ao pico

    Parameters
    ----------
    pdp : np.ndarray
        Power Delay Profile
    delays : np.ndarray
        Array com os delays/atrasos (em ns)
    threshold_db : float
        Threshold em dB relativo ao pico (ex: -20 dB)

    Returns
    -------
    float
        Maximum Excess Delay (em ns)
    """
    # Converter PDP para dB
    pdp_db = 10 * np.log10(pdp / np.max(pdp))

    # Encontrar último delay acima do threshold
    valid_indices = np.where(pdp_db >= threshold_db)[0]

    if len(valid_indices) == 0:
        return delays[-1]  # Se nenhum acima, retornar último delay

    max_index = valid_indices[-1]
    max_delay = delays[max_index]

    return max_delay


def compute_all_parameters(cirs: np.ndarray, delays: np.ndarray,
                          threshold_db: float = -20.0) -> Dict:
    """
    Calcula todos os parâmetros de caracterização do canal

    Parameters
    ----------
    cirs : np.ndarray
        Array 2D com CIRs
    delays : np.ndarray
        Array com os delays/atrasos (em ns)
    threshold_db : float
        Threshold para Maximum Excess Delay

    Returns
    -------
    dict
        Dicionário com todos os parâmetros calculados
    """
    # Calcular PDP
    pdp = compute_pdp(cirs)

    # Calcular parâmetros
    mean_excess_delay = compute_mean_excess_delay(pdp, delays)
    rms_delay_spread = compute_rms_delay_spread(pdp, delays)
    max_excess_delay = compute_maximum_excess_delay(pdp, delays, threshold_db)

    return {
        'pdp': pdp,
        'mean_excess_delay_ns': mean_excess_delay,
        'rms_delay_spread_ns': rms_delay_spread,
        'maximum_excess_delay_ns': max_excess_delay,
        'threshold_db': threshold_db,
        'n_taps': len(delays),
        'delay_range_ns': (delays.min(), delays.max())
    }


def compute_coherence_bandwidth(rms_delay_spread: float,
                               correlation_level: float = 0.9) -> float:
    """
    Estima a Coherence Bandwidth a partir do RMS Delay Spread

    Parameters
    ----------
    rms_delay_spread : float
        RMS Delay Spread em nanosegundos
    correlation_level : float
        Nível de correlação desejado (0.5 ou 0.9)

    Returns
    -------
    float
        Coherence Bandwidth em MHz

    Notes
    -----
    - Para correlação de 0.9: B_c ≈ 1 / (50 * τ_rms)
    - Para correlação de 0.5: B_c ≈ 1 / (5 * τ_rms)
    τ_rms em segundos → B_c em Hz
    """
    # Converter ns para segundos
    rms_s = rms_delay_spread * 1e-9

    if correlation_level >= 0.9:
        # Correlação forte
        B_c_Hz = 1 / (50 * rms_s)
    else:
        # Correlação moderada (0.5)
        B_c_Hz = 1 / (5 * rms_s)

    # Converter Hz para MHz
    B_c_MHz = B_c_Hz / 1e6

    return B_c_MHz


def analyze_frequency_selectivity(rms_delay_spread: float,
                                  signal_bandwidth_mhz: float) -> Dict:
    """
    Analisa se o canal é frequency-selective

    Parameters
    ----------
    rms_delay_spread : float
        RMS Delay Spread em ns
    signal_bandwidth_mhz : float
        Largura de banda do sinal em MHz

    Returns
    -------
    dict
        Análise de seletividade em frequência
    """
    # Calcular coherence bandwidth
    B_c_09 = compute_coherence_bandwidth(rms_delay_spread, 0.9)
    B_c_05 = compute_coherence_bandwidth(rms_delay_spread, 0.5)

    # Determinar tipo de canal
    if signal_bandwidth_mhz < B_c_09:
        channel_type = "Flat Fading"
        description = "Canal com desvanecimento plano (não seletivo em frequência)"
    elif signal_bandwidth_mhz < B_c_05:
        channel_type = "Moderately Selective"
        description = "Canal moderadamente seletivo em frequência"
    else:
        channel_type = "Frequency Selective"
        description = "Canal seletivo em frequência"

    return {
        'coherence_bandwidth_09_MHz': B_c_09,
        'coherence_bandwidth_05_MHz': B_c_05,
        'signal_bandwidth_MHz': signal_bandwidth_mhz,
        'channel_type': channel_type,
        'description': description
    }


def compare_datasets(params_20: Dict, params_8: Dict) -> Dict:
    """
    Compara parâmetros entre datasets de 20 e 8 taps

    Parameters
    ----------
    params_20 : dict
        Parâmetros do dataset de 20 taps
    params_8 : dict
        Parâmetros do dataset de 8 taps

    Returns
    -------
    dict
        Comparação entre os datasets
    """
    comparison = {
        'mean_excess_delay': {
            '20_taps': params_20['mean_excess_delay_ns'],
            '8_taps': params_8['mean_excess_delay_ns'],
            'diff_ns': params_20['mean_excess_delay_ns'] - params_8['mean_excess_delay_ns'],
            'diff_percent': 100 * (params_20['mean_excess_delay_ns'] - params_8['mean_excess_delay_ns']) / params_20['mean_excess_delay_ns']
        },
        'rms_delay_spread': {
            '20_taps': params_20['rms_delay_spread_ns'],
            '8_taps': params_8['rms_delay_spread_ns'],
            'diff_ns': params_20['rms_delay_spread_ns'] - params_8['rms_delay_spread_ns'],
            'diff_percent': 100 * (params_20['rms_delay_spread_ns'] - params_8['rms_delay_spread_ns']) / params_20['rms_delay_spread_ns']
        },
        'maximum_excess_delay': {
            '20_taps': params_20['maximum_excess_delay_ns'],
            '8_taps': params_8['maximum_excess_delay_ns'],
            'diff_ns': params_20['maximum_excess_delay_ns'] - params_8['maximum_excess_delay_ns'],
            'diff_percent': 100 * (params_20['maximum_excess_delay_ns'] - params_8['maximum_excess_delay_ns']) / params_20['maximum_excess_delay_ns']
        },
        'resolution_effect': {
            'n_taps_20': params_20['n_taps'],
            'n_taps_8': params_8['n_taps'],
            'delay_range_20': params_20['delay_range_ns'],
            'delay_range_8': params_8['delay_range_ns'],
            'conclusion': "Maior resolução (20 taps) captura mais detalhes do canal"
        }
    }

    return comparison


if __name__ == "__main__":
    print("Testando módulo channel_parameters.py\n")

    # Gerar dados de teste (PDP exponencial)
    print("Gerando PDP de teste (decaimento exponencial)...")
    delays = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])  # ns
    tau_rms_true = 20.0  # ns

    # PDP exponencial: P(τ) = exp(-τ/τ_rms)
    pdp = np.exp(-delays / tau_rms_true)
    pdp = pdp / np.sum(pdp)  # Normalizar

    print(f"  - Delays: {delays}")
    print(f"  - PDP: {pdp}\n")

    # Criar CIRs de teste (adicionar ruído ao redor do PDP)
    n_cirs = 100
    cirs = np.random.normal(pdp, 0.05 * pdp, (n_cirs, len(delays)))
    cirs = np.maximum(cirs, 0)  # Garantir valores positivos

    # Calcular parâmetros
    print("Calculando parâmetros do canal...")
    params = compute_all_parameters(cirs, delays, threshold_db=-20)

    print(f"\nResultados:")
    print(f"  Mean Excess Delay: {params['mean_excess_delay_ns']:.2f} ns")
    print(f"  RMS Delay Spread: {params['rms_delay_spread_ns']:.2f} ns (true={tau_rms_true} ns)")
    print(f"  Maximum Excess Delay: {params['maximum_excess_delay_ns']:.2f} ns (threshold={params['threshold_db']} dB)")

    # Coherence bandwidth
    B_c_09 = compute_coherence_bandwidth(params['rms_delay_spread_ns'], 0.9)
    B_c_05 = compute_coherence_bandwidth(params['rms_delay_spread_ns'], 0.5)

    print(f"\n  Coherence Bandwidth (90%): {B_c_09:.2f} MHz")
    print(f"  Coherence Bandwidth (50%): {B_c_05:.2f} MHz")

    # Análise de seletividade
    signal_bw = 10.0  # MHz
    selectivity = analyze_frequency_selectivity(params['rms_delay_spread_ns'], signal_bw)

    print(f"\n  Sinal: {signal_bw} MHz")
    print(f"  Tipo de canal: {selectivity['channel_type']}")
    print(f"  Descrição: {selectivity['description']}")

    print("\n✓ Módulo testado com sucesso!")
