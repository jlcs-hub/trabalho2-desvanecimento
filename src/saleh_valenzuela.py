"""
Módulo para geração de CIRs usando o Modelo Saleh-Valenzuela
Parte III do trabalho

Referência:
Saleh, A. A., & Valenzuela, R. (1987). A statistical model for indoor multipath
propagation. IEEE Journal on selected areas in communications, 5(2), 128-137.

Autor: João Lucas de Castro Santos
Data: 2025-11-05
"""

import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class SalehValenzuelaModel:
    """
    Modelo Saleh-Valenzuela para geração de CIRs

    Parâmetros do modelo:
    ---------------------
    Gamma (Γ): Taxa de chegada de clusters (1/ns)
    gamma (γ): Taxa de chegada de raios dentro dos clusters (1/ns)
    Lambda (Λ): Taxa de decaimento de potência dos clusters (1/ns)
    lambda_ (λ): Taxa de decaimento de potência dos raios (1/ns)
    omega_0 (Ω₀): Potência média do primeiro raio do primeiro cluster
    k_factor (K): Fator Rice para componente LOS (opcional, default=0 para NLOS)
    """

    def __init__(self,
                 Gamma: float = 0.02,
                 gamma: float = 0.4,
                 Lambda: float = 0.04,
                 lambda_: float = 0.5,
                 omega_0: float = 1.0,
                 k_factor: float = 0.0):
        """
        Inicializa modelo SV

        Parameters
        ----------
        Gamma : float
            Taxa de chegada de clusters (1/ns)
        gamma : float
            Taxa de chegada de raios dentro dos clusters (1/ns)
        Lambda : float
            Taxa de decaimento de potência dos clusters (1/ns)
        lambda_ : float
            Taxa de decaimento de potência dos raios (1/ns)
        omega_0 : float
            Potência média do primeiro raio do primeiro cluster
        k_factor : float
            Fator Rice (K=0 para NLOS, K>0 para LOS)
        """
        self.Gamma = Gamma
        self.gamma = gamma
        self.Lambda = Lambda
        self.lambda_ = lambda_
        self.omega_0 = omega_0
        self.k_factor = k_factor

    def generate_cir(self,
                    t_max: float = 500.0,
                    dt: float = 1.0,
                    n_clusters: int = 5,
                    n_rays_per_cluster: int = 10,
                    random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma CIR usando o modelo SV

        Parameters
        ----------
        t_max : float
            Tempo máximo (delay máximo) em ns
        dt : float
            Resolução temporal em ns
        n_clusters : int
            Número de clusters a gerar
        n_rays_per_cluster : int
            Número de raios por cluster
        random_seed : int, optional
            Seed para reprodutibilidade

        Returns
        -------
        tuple
            (delays, cir) - Arrays com delays e potências da CIR
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Grid de tempo
        delays = np.arange(0, t_max, dt)
        cir = np.zeros_like(delays)

        # Gerar tempos de chegada dos clusters
        # Processo de Poisson com taxa Gamma
        cluster_delays = np.cumsum(np.random.exponential(1/self.Gamma, n_clusters))

        # Para cada cluster
        for l, T_l in enumerate(cluster_delays):
            if T_l >= t_max:
                break

            # Potência do cluster l
            # Ω_l = Ω_0 * exp(-T_l * Λ)
            omega_l = self.omega_0 * np.exp(-T_l * self.Lambda)

            # Gerar raios dentro do cluster
            # Tempos de chegada relativos ao cluster
            ray_delays = np.cumsum(np.random.exponential(1/self.gamma, n_rays_per_cluster))

            for k, tau_kl in enumerate(ray_delays):
                # Tempo absoluto do raio
                t_kl = T_l + tau_kl

                if t_kl >= t_max:
                    break

                # Potência do raio (k,l)
                # β_kl² = Ω_l * exp(-τ_kl * λ)
                beta_kl_squared = omega_l * np.exp(-tau_kl * self.lambda_)

                # Amplitude (Rayleigh distribuída)
                beta_kl = np.sqrt(beta_kl_squared) * np.random.rayleigh(1.0)

                # Fase uniformemente distribuída
                phi_kl = np.random.uniform(0, 2*np.pi)

                # Contribuição complexa do raio
                # h_kl = β_kl * exp(j*φ_kl)
                # Potência = |h_kl|² = β_kl²
                power_kl = beta_kl ** 2

                # Adicionar ao bin de tempo mais próximo
                idx = int(t_kl / dt)
                if idx < len(cir):
                    cir[idx] += power_kl

        # Adicionar componente LOS se K > 0
        if self.k_factor > 0:
            # Componente LOS no tempo zero com potência determinada por K
            # K = potência_LOS / potência_scatter
            power_scatter = np.sum(cir)
            power_los = self.k_factor * power_scatter
            cir[0] += power_los

        # Normalizar pela potência total
        total_power = np.sum(cir)
        if total_power > 0:
            cir = cir / total_power

        return delays, cir

    def generate_multiple_cirs(self,
                              n_cirs: int = 100,
                              t_max: float = 500.0,
                              dt: float = 1.0,
                              n_clusters: int = 5,
                              n_rays_per_cluster: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera múltiplas CIRs

        Parameters
        ----------
        n_cirs : int
            Número de CIRs a gerar
        t_max : float
            Delay máximo em ns
        dt : float
            Resolução temporal em ns
        n_clusters : int
            Número de clusters por CIR
        n_rays_per_cluster : int
            Número de raios por cluster

        Returns
        -------
        tuple
            (delays, cirs) - Arrays com delays e matriz de CIRs
        """
        # Gerar primeira CIR para obter tamanho
        delays, cir0 = self.generate_cir(t_max, dt, n_clusters, n_rays_per_cluster)

        # Matriz para armazenar CIRs
        cirs = np.zeros((n_cirs, len(delays)))
        cirs[0] = cir0

        # Gerar demais CIRs
        for i in range(1, n_cirs):
            _, cirs[i] = self.generate_cir(t_max, dt, n_clusters, n_rays_per_cluster)

        return delays, cirs

    def get_parameters_dict(self) -> Dict:
        """Retorna dicionário com parâmetros do modelo"""
        return {
            'Gamma': self.Gamma,
            'gamma': self.gamma,
            'Lambda': self.Lambda,
            'lambda': self.lambda_,
            'omega_0': self.omega_0,
            'k_factor': self.k_factor
        }

    def __repr__(self):
        return (f"SalehValenzuelaModel(Γ={self.Gamma}, γ={self.gamma}, "
                f"Λ={self.Lambda}, λ={self.lambda_}, "
                f"Ω₀={self.omega_0}, K={self.k_factor})")


def vary_parameter_study(base_params: Dict,
                        param_name: str,
                        param_values: List[float],
                        t_max: float = 500.0,
                        dt: float = 1.0) -> Dict:
    """
    Estuda efeito da variação de um parâmetro

    Parameters
    ----------
    base_params : dict
        Parâmetros base do modelo
    param_name : str
        Nome do parâmetro a variar (ex: 'Gamma', 'gamma', 'Lambda', etc)
    param_values : list
        Lista de valores para o parâmetro
    t_max : float
        Delay máximo
    dt : float
        Resolução temporal

    Returns
    -------
    dict
        Dicionário com CIRs geradas para cada valor do parâmetro
    """
    results = {
        'param_name': param_name,
        'param_values': param_values,
        'cirs': [],
        'delays': None
    }

    for value in param_values:
        # Atualizar parâmetro
        params = base_params.copy()
        params[param_name] = value

        # Criar modelo
        model = SalehValenzuelaModel(**params)

        # Gerar CIR
        delays, cir = model.generate_cir(t_max, dt, random_seed=42)

        # Armazenar
        results['cirs'].append(cir)
        if results['delays'] is None:
            results['delays'] = delays

    results['cirs'] = np.array(results['cirs'])

    return results


def get_default_parameters() -> Dict:
    """
    Retorna parâmetros padrão do modelo SV
    Baseados em valores típicos para ambientes indoor

    Returns
    -------
    dict
        Parâmetros padrão
    """
    return {
        'Gamma': 0.02,      # Taxa de chegada de clusters (1/ns)
        'gamma': 0.4,       # Taxa de chegada de raios (1/ns)
        'Lambda': 0.04,     # Decaimento de clusters (1/ns)
        'lambda_': 0.5,     # Decaimento de raios (1/ns)
        'omega_0': 1.0,     # Potência inicial
        'k_factor': 0.0     # NLOS (sem LOS)
    }


def compare_nlos_vs_los(base_params: Dict,
                       k_values: List[float] = [0, 1, 3, 10]) -> Dict:
    """
    Compara cenários NLOS vs LOS (variando fator K)

    Parameters
    ----------
    base_params : dict
        Parâmetros base
    k_values : list
        Valores de K a testar

    Returns
    -------
    dict
        Resultados da comparação
    """
    return vary_parameter_study(base_params, 'k_factor', k_values)


if __name__ == "__main__":
    print("Testando módulo saleh_valenzuela.py\n")

    # Parâmetros padrão
    print("1. Testando com parâmetros padrão...")
    params = get_default_parameters()
    print(f"   Parâmetros: {params}\n")

    # Criar modelo
    model = SalehValenzuelaModel(**params)
    print(f"   {model}\n")

    # Gerar CIR
    print("2. Gerando CIR sintética...")
    delays, cir = model.generate_cir(t_max=500, dt=2.0, n_clusters=5,
                                    n_rays_per_cluster=10, random_seed=42)

    print(f"   - Delays: {len(delays)} pontos")
    print(f"   - CIR gerada com sucesso")
    print(f"   - Potência total: {np.sum(cir):.6f} (deve ser ≈1.0)")
    print(f"   - Delay médio: {np.sum(delays * cir):.2f} ns")

    # Gerar múltiplas CIRs
    print("\n3. Gerando 50 CIRs...")
    delays_multi, cirs_multi = model.generate_multiple_cirs(n_cirs=50, t_max=500, dt=2.0)

    print(f"   - {cirs_multi.shape[0]} CIRs geradas")
    print(f"   - Shape: {cirs_multi.shape}")

    # PDP médio
    pdp = np.mean(cirs_multi, axis=0)
    print(f"   - PDP médio calculado")
    print(f"   - Delay médio do PDP: {np.sum(delays_multi * pdp):.2f} ns")

    # Testar variação de parâmetro
    print("\n4. Testando variação do parâmetro Gamma...")
    results = vary_parameter_study(
        params,
        'Gamma',
        [0.01, 0.02, 0.05, 0.1],
        t_max=500,
        dt=2.0
    )

    print(f"   - Parâmetro variado: {results['param_name']}")
    print(f"   - Valores testados: {results['param_values']}")
    print(f"   - CIRs geradas: {results['cirs'].shape}")

    print("\n✓ Módulo testado com sucesso!")
