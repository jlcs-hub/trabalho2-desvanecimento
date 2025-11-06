"""
Pacote para caracterização de canal sem fio via CIRs.

Este pacote contém módulos para:
- Carregamento de CIRs (cir_loader)
- Análise de distribuições estatísticas (statistical_distributions)
- Cálculo de parâmetros do canal (channel_parameters)
- Criação de animações (animation)
- Modelo Saleh-Valenzuela (saleh_valenzuela)
- Visualizações (visualization)
"""

__version__ = "2.0.0"
__author__ = "João Lucas de Castro Santos"

# Importações facilitadas para uso direto
from . import cir_loader
from . import statistical_distributions
from . import channel_parameters
from . import animation
from . import saleh_valenzuela
from . import visualization

__all__ = [
    'cir_loader',
    'statistical_distributions',
    'channel_parameters',
    'animation',
    'saleh_valenzuela',
    'visualization'
]
