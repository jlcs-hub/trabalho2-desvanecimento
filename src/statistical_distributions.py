"""
Módulo para análise de distribuições estatísticas de sinais CIR
Parte I do trabalho: Rayleigh, Rice e Nakagami

Autor: João Lucas de Castro Santos
Data: 2025-11-05
"""

import numpy as np
from scipy import signal, stats
from scipy.optimize import curve_fit
from typing import Tuple, Dict
import warnings


# ============================================================================
# 1. PROCESSAMENTO DE SINAL
# ============================================================================

def compute_total_power_signal(cirs: np.ndarray) -> np.ndarray:
    """
    Monta sinal de potência recebida em função do tempo
    Soma todas as componentes (taps) para cada instante de tempo (CIR)

    Parameters
    ----------
    cirs : np.ndarray
        Array 2D com CIRs (linhas = CIRs/tempo, colunas = taps)

    Returns
    -------
    np.ndarray
        Array 1D com potência total por instante de tempo
    """
    return np.sum(cirs, axis=1)


def detect_mean_variation(signal_power: np.ndarray,
                         window_size: int = 100) -> Tuple[bool, float]:
    """
    Verifica se o sinal tem variação da média (perda de percurso/sombreamento)

    Parameters
    ----------
    signal_power : np.ndarray
        Sinal de potência total
    window_size : int
        Tamanho da janela para média móvel

    Returns
    -------
    tuple
        (has_variation, mean_std) - se há variação e desvio padrão da média
    """
    # Calcular média móvel
    moving_avg = np.convolve(signal_power,
                            np.ones(window_size)/window_size,
                            mode='valid')

    # Desvio padrão da média móvel normalizado pela média global
    mean_std = np.std(moving_avg) / np.mean(signal_power)

    # Considerar que há variação se std > 5%
    has_variation = mean_std > 0.05

    return has_variation, mean_std


def apply_highpass_filter(signal_power: np.ndarray,
                         cutoff_freq: float = 0.01,
                         filter_order: int = 5) -> np.ndarray:
    """
    Aplica filtro passa-alta para remover componentes de baixa frequência
    (perda de percurso e sombreamento)

    Parameters
    ----------
    signal_power : np.ndarray
        Sinal de potência total
    cutoff_freq : float
        Frequência de corte normalizada (0 a 1)
    filter_order : int
        Ordem do filtro Butterworth

    Returns
    -------
    np.ndarray
        Sinal filtrado com média nula
    """
    # Filtro Butterworth passa-alta
    sos = signal.butter(filter_order, cutoff_freq,
                       btype='highpass', output='sos')
    filtered_signal = signal.sosfilt(sos, signal_power)

    return filtered_signal


def filter_to_zero_mean(signal_power: np.ndarray, method: str = 'remove_mean') -> np.ndarray:
    """
    Filtra componentes de perda de percurso e sombreamento,
    deixando sinal com média nula

    Parameters
    ----------
    signal_power : np.ndarray
        Sinal de potência total
    method : str
        Método: 'remove_mean' (simples) ou 'highpass' (filtro)

    Returns
    -------
    np.ndarray
        Sinal filtrado com média nula
    """
    if method == 'remove_mean':
        # Método simples: subtrai média
        return signal_power - np.mean(signal_power)

    elif method == 'highpass':
        # Método com filtro passa-alta
        return apply_highpass_filter(signal_power)

    elif method == 'detrend':
        # Remove tendência linear
        return signal.detrend(signal_power)

    else:
        raise ValueError(f"Método '{method}' desconhecido")


# ============================================================================
# 2. AJUSTE DE DISTRIBUIÇÕES - Método 1 (scipy.stats)
# ============================================================================

def fit_rayleigh(data: np.ndarray) -> Tuple[float, Dict]:
    """
    Ajusta distribuição Rayleigh aos dados usando MLE

    Parameters
    ----------
    data : np.ndarray
        Dados (devem ser positivos)

    Returns
    -------
    tuple
        (sigma, {'sigma': sigma, 'sse': sse})
    """
    data_positive = np.abs(data)  # Garantir valores positivos

    # MLE: scipy usa parametrização (loc, scale)
    # Rayleigh: scale = sigma
    loc, scale = stats.rayleigh.fit(data_positive, floc=0)

    sigma = scale

    # Calcular erro (SSE - Sum of Squared Errors)
    theoretical_cdf = stats.rayleigh.cdf(np.sort(data_positive), scale=sigma)
    empirical_cdf = np.arange(1, len(data_positive) + 1) / len(data_positive)
    sse = np.sum((theoretical_cdf - empirical_cdf) ** 2)

    return sigma, {'sigma': sigma, 'sse': sse, 'loc': loc}


def fit_rice(data: np.ndarray) -> Tuple[Tuple[float, float], Dict]:
    """
    Ajusta distribuição Rice aos dados usando MLE

    Parameters
    ----------
    data : np.ndarray
        Dados (devem ser positivos)

    Returns
    -------
    tuple
        ((K, omega), {'b': b, 'scale': scale, 'K': K, 'omega': omega, 'sse': sse})

    Notes
    -----
    scipy.stats.rice usa parametrização (b, scale):
    - b = nu/sigma (razão sinal/ruído)
    - scale = sigma

    Convertemos para (K, omega):
    - K = b^2 / 2 (fator Rice)
    - omega = scale^2 * (1 + b^2)
    """
    data_positive = np.abs(data)

    # MLE com scipy
    b, loc, scale = stats.rice.fit(data_positive, floc=0)

    # Converter para parâmetros K e omega
    K = (b ** 2) / 2
    omega = scale ** 2 * (1 + b ** 2)

    # Calcular SSE
    theoretical_cdf = stats.rice.cdf(np.sort(data_positive), b, scale=scale)
    empirical_cdf = np.arange(1, len(data_positive) + 1) / len(data_positive)
    sse = np.sum((theoretical_cdf - empirical_cdf) ** 2)

    return (K, omega), {'b': b, 'scale': scale, 'K': K, 'omega': omega,
                        'sse': sse, 'loc': loc}


def fit_nakagami(data: np.ndarray) -> Tuple[Tuple[float, float], Dict]:
    """
    Ajusta distribuição Nakagami-m aos dados usando MLE

    Parameters
    ----------
    data : np.ndarray
        Dados (devem ser positivos)

    Returns
    -------
    tuple
        ((m, omega), {'m': m, 'omega': omega, 'sse': sse})
    """
    data_positive = np.abs(data)

    # MLE com scipy
    # scipy.stats.nakagami usa parametrização (nu, scale)
    # onde nu = m, omega = scale^2
    nu, loc, scale = stats.nakagami.fit(data_positive, floc=0)

    m = nu
    omega = scale ** 2

    # Calcular SSE
    theoretical_cdf = stats.nakagami.cdf(np.sort(data_positive), nu, scale=scale)
    empirical_cdf = np.arange(1, len(data_positive) + 1) / len(data_positive)
    sse = np.sum((theoretical_cdf - empirical_cdf) ** 2)

    return (m, omega), {'m': m, 'omega': omega, 'sse': sse, 'loc': loc, 'nu': nu, 'scale': scale}


# ============================================================================
# 3. ESTIMADORES DA DISSERTAÇÃO (Envoltória Normalizada ρ)
# ============================================================================

def normalize_envelope(envelope: np.ndarray) -> np.ndarray:
    """
    Normaliza envoltória conforme dissertação
    ρ = r / sqrt(E[r^2])

    Parameters
    ----------
    envelope : np.ndarray
        Envoltória do sinal

    Returns
    -------
    np.ndarray
        Envoltória normalizada
    """
    rms = np.sqrt(np.mean(envelope ** 2))
    return envelope / rms


def estimate_rayleigh_from_envelope(envelope: np.ndarray) -> float:
    """
    Estima parâmetro sigma da distribuição Rayleigh
    usando envoltória normalizada (Seção 4.2 da dissertação)

    Parameters
    ----------
    envelope : np.ndarray
        Envoltória do sinal (positiva)

    Returns
    -------
    float
        Parâmetro sigma estimado
    """
    # Para Rayleigh: E[r^2] = 2*sigma^2
    # Após normalização: sigma_normalizado = 1/sqrt(2)
    rho = normalize_envelope(envelope)

    # Estimar sigma a partir da envoltória normalizada
    sigma = np.sqrt(np.mean(rho ** 2) / 2)

    return sigma


def estimate_rice_from_envelope(envelope: np.ndarray) -> Tuple[float, float]:
    """
    Estima parâmetros K e omega da distribuição Rice
    usando estimadores da Seção 4.2 da dissertação

    Parameters
    ----------
    envelope : np.ndarray
        Envoltória do sinal (positiva)

    Returns
    -------
    tuple
        (K, omega) - fator Rice e potência média

    Notes
    -----
    Estimadores baseados em momentos:
    - E[r^2] = omega
    - K relacionado com razão momento de 1ª e 2ª ordem
    """
    # Normalizar envoltória
    rho = normalize_envelope(envelope)

    # Momentos
    m1 = np.mean(rho)  # Primeiro momento
    m2 = np.mean(rho ** 2)  # Segundo momento

    # Estimador de K baseado em momentos
    # Aproximação: K ≈ (m1^2) / (m2 - m1^2) para Rice
    # Limitando para evitar divisão por zero ou valores negativos
    denominator = max(m2 - m1**2, 1e-10)
    K_normalized = (m1 ** 2) / denominator

    # Para envoltória normalizada: m2 = 1
    # Convertendo de volta para parâmetros originais
    omega_original = np.mean(envelope ** 2)
    K_original = K_normalized  # K é adimensional

    return K_original, omega_original


def estimate_nakagami_from_envelope(envelope: np.ndarray) -> Tuple[float, float]:
    """
    Estima parâmetros m e omega da distribuição Nakagami
    usando estimadores da Seção 4.2 da dissertação

    Parameters
    ----------
    envelope : np.ndarray
        Envoltória do sinal (positiva)

    Returns
    -------
    tuple
        (m, omega) - parâmetro de forma e potência média

    Notes
    -----
    Estimador de m baseado em momentos:
    m = E[r^2]^2 / E[(r^2 - E[r^2])^2]
    omega = E[r^2]
    """
    # Potência instantânea
    power = envelope ** 2

    # Estimadores
    omega = np.mean(power)
    var_power = np.var(power)

    # m = omega^2 / var(power)
    m = (omega ** 2) / max(var_power, 1e-10)

    # Limitar m entre 0.5 e 100 (valores físicos)
    m = np.clip(m, 0.5, 100.0)

    return m, omega


# ============================================================================
# 4. GERAÇÃO DE AMOSTRAS
# ============================================================================

def generate_rayleigh_samples(sigma: float, n_samples: int = 1000,
                             random_seed: int = None) -> np.ndarray:
    """
    Gera amostras de desvanecimento Rayleigh

    Parameters
    ----------
    sigma : float
        Parâmetro sigma
    n_samples : int
        Número de amostras
    random_seed : int, optional
        Seed para reprodutibilidade

    Returns
    -------
    np.ndarray
        Amostras geradas
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    return stats.rayleigh.rvs(scale=sigma, size=n_samples)


def generate_rice_samples(K: float, omega: float, n_samples: int = 1000,
                         random_seed: int = None) -> np.ndarray:
    """
    Gera amostras de desvanecimento Rice

    Parameters
    ----------
    K : float
        Fator Rice (K = A^2 / 2σ^2)
    omega : float
        Potência média total
    n_samples : int
        Número de amostras
    random_seed : int, optional
        Seed para reprodutibilidade

    Returns
    -------
    np.ndarray
        Amostras geradas
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Converter K e omega para parâmetros scipy (b, scale)
    # K = b^2 / 2  =>  b = sqrt(2*K)
    # omega = scale^2 * (1 + b^2)  =>  scale = sqrt(omega / (1 + b^2))

    b = np.sqrt(2 * K)
    scale = np.sqrt(omega / (1 + b ** 2))

    return stats.rice.rvs(b, scale=scale, size=n_samples)


def generate_nakagami_samples(m: float, omega: float, n_samples: int = 1000,
                             random_seed: int = None) -> np.ndarray:
    """
    Gera amostras de desvanecimento Nakagami-m

    Parameters
    ----------
    m : float
        Parâmetro de forma (m >= 0.5)
    omega : float
        Potência média
    n_samples : int
        Número de amostras
    random_seed : int, optional
        Seed para reprodutibilidade

    Returns
    -------
    np.ndarray
        Amostras geradas
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # scipy.stats.nakagami usa nu=m, scale=sqrt(omega)
    scale = np.sqrt(omega)

    return stats.nakagami.rvs(m, scale=scale, size=n_samples)


# ============================================================================
# 5. ANÁLISE COMPLETA
# ============================================================================

def analyze_signal_distributions(signal_power: np.ndarray,
                                filter_method: str = 'remove_mean') -> Dict:
    """
    Realiza análise completa de distribuições no sinal

    Parameters
    ----------
    signal_power : np.ndarray
        Sinal de potência total
    filter_method : str
        Método de filtragem ('remove_mean', 'highpass', 'detrend')

    Returns
    -------
    dict
        Resultados completos da análise
    """
    results = {}

    # 1. Verificar variação da média
    has_variation, mean_std = detect_mean_variation(signal_power)
    results['mean_variation'] = {
        'has_variation': has_variation,
        'mean_std': mean_std
    }

    # 2. Filtrar sinal
    filtered_signal = filter_to_zero_mean(signal_power, method=filter_method)
    results['filtered_signal'] = filtered_signal

    # 3. Converter para envoltória (valores positivos)
    envelope = np.abs(filtered_signal)

    # 4. Método 1: Ajuste com scipy.stats
    sigma_rayleigh, rayleigh_info = fit_rayleigh(envelope)
    (K_rice, omega_rice), rice_info = fit_rice(envelope)
    (m_nakagami, omega_nakagami), nakagami_info = fit_nakagami(envelope)

    results['scipy_fit'] = {
        'rayleigh': rayleigh_info,
        'rice': rice_info,
        'nakagami': nakagami_info
    }

    # 5. Método 2: Estimadores da dissertação
    sigma_rayleigh_diss = estimate_rayleigh_from_envelope(envelope)
    K_rice_diss, omega_rice_diss = estimate_rice_from_envelope(envelope)
    m_nakagami_diss, omega_nakagami_diss = estimate_nakagami_from_envelope(envelope)

    results['dissertation_estimators'] = {
        'rayleigh': {'sigma': sigma_rayleigh_diss},
        'rice': {'K': K_rice_diss, 'omega': omega_rice_diss},
        'nakagami': {'m': m_nakagami_diss, 'omega': omega_nakagami_diss}
    }

    # 6. Envoltória normalizada
    rho = normalize_envelope(envelope)
    results['normalized_envelope'] = rho

    return results


if __name__ == "__main__":
    print("Testando módulo statistical_distributions.py\n")

    # Gerar dados de teste (distribuição Rice conhecida)
    print("Gerando dados de teste (Rice com K=3, omega=1)...")
    true_K = 3.0
    true_omega = 1.0
    n_samples = 10000

    test_data = generate_rice_samples(true_K, true_omega, n_samples, random_seed=42)

    print(f"  - Geradas {n_samples} amostras")
    print(f"  - Média: {np.mean(test_data):.4f}")
    print(f"  - Std: {np.std(test_data):.4f}\n")

    # Testar ajuste
    print("Testando ajuste de distribuições...")

    # Rayleigh
    sigma, rayleigh_info = fit_rayleigh(test_data)
    print(f"  Rayleigh: sigma = {sigma:.4f}, SSE = {rayleigh_info['sse']:.6f}")

    # Rice
    (K, omega), rice_info = fit_rice(test_data)
    print(f"  Rice: K = {K:.4f} (true={true_K}), omega = {omega:.4f} (true={true_omega})")
    print(f"        SSE = {rice_info['sse']:.6f}")

    # Nakagami
    (m, omega_nak), nak_info = fit_nakagami(test_data)
    print(f"  Nakagami: m = {m:.4f}, omega = {omega_nak:.4f}, SSE = {nak_info['sse']:.6f}")

    print("\n✓ Módulo testado com sucesso!")
