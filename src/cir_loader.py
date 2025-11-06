"""
Módulo para carregamento e processamento de dados CIR
(Channel Impulse Response)

Autor: João Lucas de Castro Santos
Data: 2025-11-05
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class CIRData:
    """Classe para armazenar e processar dados de CIR"""

    def __init__(self, delays: np.ndarray, cirs: np.ndarray, filename: str = ""):
        """
        Inicializa objeto CIRData

        Parameters
        ----------
        delays : np.ndarray
            Array 1D com os atrasos/delays em nanosegundos
        cirs : np.ndarray
            Array 2D com as CIRs (linhas = CIRs, colunas = taps)
        filename : str, optional
            Nome do arquivo original
        """
        self.delays = delays
        self.cirs = cirs
        self.filename = filename
        self.n_cirs = cirs.shape[0]
        self.n_taps = cirs.shape[1]

    def __repr__(self):
        return (f"CIRData(n_cirs={self.n_cirs}, n_taps={self.n_taps}, "
                f"filename='{self.filename}')")

    def get_total_power_per_cir(self) -> np.ndarray:
        """
        Calcula a potência total de cada CIR (soma de todos os taps)

        Returns
        -------
        np.ndarray
            Array 1D com a potência total de cada CIR
        """
        return np.sum(self.cirs, axis=1)

    def get_average_pdp(self) -> np.ndarray:
        """
        Calcula o Power Delay Profile (PDP) médio
        (média de todas as CIRs)

        Returns
        -------
        np.ndarray
            Array 1D com o PDP médio
        """
        return np.mean(self.cirs, axis=0)

    def get_cir_subset(self, n_samples: int = 100, random_seed: int = 42) -> np.ndarray:
        """
        Retorna um subconjunto aleatório de CIRs

        Parameters
        ----------
        n_samples : int
            Número de CIRs a retornar
        random_seed : int
            Seed para reprodutibilidade

        Returns
        -------
        np.ndarray
            Array 2D com subset de CIRs
        """
        np.random.seed(random_seed)
        indices = np.random.choice(self.n_cirs,
                                   size=min(n_samples, self.n_cirs),
                                   replace=False)
        return self.cirs[indices]

    def summary(self) -> Dict:
        """
        Retorna resumo estatístico dos dados

        Returns
        -------
        dict
            Dicionário com estatísticas
        """
        total_powers = self.get_total_power_per_cir()
        return {
            'filename': self.filename,
            'n_cirs': self.n_cirs,
            'n_taps': self.n_taps,
            'delay_range_ns': (self.delays.min(), self.delays.max()),
            'power_mean': total_powers.mean(),
            'power_std': total_powers.std(),
            'power_range': (total_powers.min(), total_powers.max())
        }


def load_cir_data(filepath: str) -> CIRData:
    """
    Carrega dados de CIR de um arquivo CSV

    Formato esperado:
    - Linha 1: atrasos/delays em nanosegundos
    - Linhas seguintes: potências normalizadas (cada linha = uma CIR)

    Parameters
    ----------
    filepath : str
        Caminho para o arquivo CSV

    Returns
    -------
    CIRData
        Objeto contendo os dados carregados

    Examples
    --------
    >>> data = load_cir_data('data/raw/cirNTaps20SteamPlant.csv')
    >>> print(data)
    CIRData(n_cirs=1000, n_taps=20, filename='cirNTaps20SteamPlant.csv')
    """
    # Carregar CSV
    df = pd.read_csv(filepath, header=None)

    # Primeira linha = delays
    delays = df.iloc[0].values.astype(float)

    # Demais linhas = CIRs
    cirs = df.iloc[1:].values.astype(float)

    # Extrair nome do arquivo
    filename = filepath.split('/')[-1]

    # Validar dados
    if np.any(np.isnan(delays)):
        raise ValueError("Delays contêm valores NaN")

    if np.any(np.isnan(cirs)):
        print(f"Aviso: CIRs contêm {np.sum(np.isnan(cirs))} valores NaN")
        # Substituir NaN por 0
        cirs = np.nan_to_num(cirs, nan=0.0)

    return CIRData(delays, cirs, filename)


def load_both_datasets() -> Tuple[CIRData, CIRData]:
    """
    Carrega ambos os datasets (20 e 8 taps)

    Returns
    -------
    tuple
        (data_20taps, data_8taps)
    """
    data_20 = load_cir_data('data/raw/cirNTaps20SteamPlant.csv')
    data_8 = load_cir_data('data/raw/cirNTaps8SteamPlant.csv')

    return data_20, data_8


def validate_cir_data(cir_data: CIRData, verbose: bool = True) -> bool:
    """
    Valida dados de CIR

    Parameters
    ----------
    cir_data : CIRData
        Dados a validar
    verbose : bool
        Se True, imprime informações de validação

    Returns
    -------
    bool
        True se dados são válidos
    """
    valid = True

    # Verificar dimensões
    if cir_data.n_taps != len(cir_data.delays):
        if verbose:
            print("ERRO: Número de taps não corresponde ao tamanho do array de delays")
        valid = False

    # Verificar valores negativos
    if np.any(cir_data.cirs < 0):
        if verbose:
            print("AVISO: Dados contêm valores negativos")
        valid = False

    # Verificar normalização (pelo menos um valor próximo de 1)
    max_val = np.max(cir_data.cirs)
    if not (0.95 <= max_val <= 1.05):
        if verbose:
            print(f"AVISO: Valor máximo {max_val:.3f} não está próximo de 1.0")

    if verbose and valid:
        print("✓ Dados validados com sucesso")
        print(f"  - {cir_data.n_cirs} CIRs")
        print(f"  - {cir_data.n_taps} taps")
        print(f"  - Delays: {cir_data.delays[0]:.1f} a {cir_data.delays[-1]:.1f} ns")

    return valid


if __name__ == "__main__":
    # Teste básico
    print("Testando carregamento de dados CIR...\n")

    # Carregar dados
    data_20, data_8 = load_both_datasets()

    # Mostrar informações
    print("Dataset 20 taps:")
    print(data_20)
    validate_cir_data(data_20)
    print("\nResumo:")
    for key, value in data_20.summary().items():
        print(f"  {key}: {value}")

    print("\n" + "="*60 + "\n")

    print("Dataset 8 taps:")
    print(data_8)
    validate_cir_data(data_8)
    print("\nResumo:")
    for key, value in data_8.summary().items():
        print(f"  {key}: {value}")
