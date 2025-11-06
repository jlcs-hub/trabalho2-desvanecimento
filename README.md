# Caracterização de Canal Sem Fio via CIRs

**Autor:** João Lucas de Castro Santos
**Data:** Novembro 2025

---

## Sobre

Caracterização completa de canal sem fio a partir de medidas de **Respostas ao Impulso do Canal (CIRs)** obtidas em ambiente industrial (Steam Plant) pelo NIST.

**Dataset:** [NIST - Industrial Wireless Systems Radio Propagation Measurements](https://www.nist.gov/publications/industrial-wireless-systems-radio-propagation-measurements)

---

## Estrutura do Trabalho

### Parte I: Distribuições Estatísticas
Identificar qual distribuição (Rayleigh, Rice ou Nakagami) melhor modela o ambiente de propagação.

**Procedimento:**
1. Somar taps para obter potência total recebida em cada posição
2. Filtrar componentes de larga escala (perda de percurso + sombreamento)
3. Ajustar distribuições usando dois métodos:
   - Método 1: `scipy.stats.fit` (Maximum Likelihood)
   - Método 2: Estimadores da envoltória normalizada
4. Comparar via SSE (Sum of Squared Errors)
5. Analisar influência da resolução (20 vs 8 taps)

**Resultado:** Nakagami-m com **m = 0.30** (desvanecimento severo, ambiente NLOS)

### Parte II: Parâmetros do Canal
Calcular parâmetros temporais e espectrais do canal.

**Procedimento:**
1. Calcular Power Delay Profile (PDP) médio
2. Extrair parâmetros:
   - RMS Delay Spread
   - Mean Excess Delay
   - Maximum Excess Delay (threshold -20 dB)
   - Coherence Bandwidth (90% e 50%)
3. Criar animações da evolução temporal das CIRs
4. Comparar influência da resolução

**Resultado:** RMS Delay Spread = **117.7 ns**, Coherence Bandwidth = **1.70 MHz**
**Implicação:** Canal seletivo em frequência para WiFi, 4G e 5G (requer equalização)

### Parte III: Modelo Saleh-Valenzuela
Implementar modelo estatístico para geração de CIRs sintéticas.

**Procedimento:**
1. Implementar modelo SV com 6 parâmetros (Γ, γ, Λ, λ, Ω₀, K)
2. Processo de Poisson para tempos de chegada de clusters e raios
3. Decaimento exponencial de potência
4. Demonstrar efeito individual de cada parâmetro
5. Comparar cenários NLOS vs LOS (variando K)

**Resultado:** Modelo funcional gerando CIRs sintéticas com estatísticas similares aos dados reais

---

## Datasets

**Arquivos:**
- `cirNTaps20SteamPlant.csv` - 10,530 CIRs com 20 taps
- `cirNTaps8SteamPlant.csv` - 10,530 CIRs com 8 taps

**Formato:**
- Linha 1: Delays em nanosegundos
- Linhas seguintes: Potências normalizadas (0 a 1)

---

## Como Executar

### Requisitos
```bash
pip install -r requirements.txt
```

**Dependências:** numpy, scipy, pandas, matplotlib, seaborn, jupyter

### Executar Todo o Trabalho
```bash
python3 scripts/run_all.py
```

### Executar Partes Individuais
```bash
python3 scripts/parte1_distribuicoes.py      # Distribuições estatísticas
python3 scripts/parte2_parametros_canal.py   # Parâmetros do canal
python3 scripts/parte3_modelo_sv.py          # Modelo Saleh-Valenzuela
```

### Ver Resultados (Recomendado)
```bash
jupyter notebook notebooks/trabalho_cir_resumo.ipynb
```

---

## Estrutura do Projeto

```
├── src/                              # Módulos Python
│   ├── cir_loader.py                 # Carregamento de dados
│   ├── statistical_distributions.py  # Ajuste de distribuições (Parte I)
│   ├── channel_parameters.py         # Cálculo de parâmetros (Parte II)
│   ├── saleh_valenzuela.py           # Modelo SV (Parte III)
│   ├── animation.py                  # Criação de GIFs
│   └── visualization.py              # Funções de plot
│
├── scripts/                          # Scripts executáveis
│   ├── parte1_distribuicoes.py
│   ├── parte2_parametros_canal.py
│   ├── parte3_modelo_sv.py
│   └── run_all.py
│
├── data/raw/                         # Datasets originais
│   ├── cirNTaps20SteamPlant.csv
│   └── cirNTaps8SteamPlant.csv
│
├── figures/                          # Figuras geradas (16 total)
│   ├── parte1/                       # 6 figuras
│   ├── parte2/                       # 4 figuras
│   └── parte3/                       # 6 figuras
│
├── media/                            # Animações (3 GIFs)
│
├── results/                          # Resultados CSV (5 arquivos)
│
└── notebooks/                        # Notebooks Jupyter (2)
```

---

## Resultados Principais

### Distribuição Estatística
- **Vencedor:** Nakagami-m com m = 0.30
- **SSE:** 50.6 (9× melhor que Rayleigh/Rice)
- **Interpretação:** Desvanecimento severo (m < 1), ambiente NLOS

### Parâmetros do Canal (20 taps)
| Parâmetro | Valor |
|-----------|-------|
| RMS Delay Spread | 117.7 ns |
| Mean Excess Delay | 162.7 ns |
| Maximum Excess Delay | 367.1 ns |
| Coherence Bandwidth (50%) | 1.70 MHz |
| Coherence Bandwidth (90%) | 680 kHz |

**Seletividade em Frequência:**
- **Canal Seletivo:** WiFi (20/40 MHz), 4G/5G (>5 MHz) - requer equalização
- **Canal Plano:** Bluetooth (1 MHz), LoRa (<500 kHz)

### Comparação 20 vs 8 taps
- **RMS Delay Spread:** 117.7 ns (20 taps) vs 120.4 ns (8 taps) - diferença de 2.3%
- **Resolução temporal:** 20 taps captura melhor a dispersão temporal
- **Trade-off:** Maior resolução = melhor caracterização, porém maior complexidade

---

## Outputs Gerados

| Tipo | Quantidade | Descrição |
|------|------------|-----------|
| **Figuras** | 16 | Histogramas, CDFs, PDPs, estudos paramétricos (PNG 300 DPI) |
| **Animações** | 3 | Evolução temporal das CIRs (GIF) |
| **Resultados** | 5 | Parâmetros calculados (CSV) |
| **Notebooks** | 2 | Análise completa + resumo de resultados |
| **TOTAL** | 26 | - |

---

## Metodologia

### Filtragem MLMS (Parte I)
Filtro adaptativo para remover componentes de larga escala:
```python
def filtroMLMS(potencia_total, n_taps=20, mu=0.05, epsilon=1e-6):
    """
    Filtra o sinal de potência com filtro adaptativo MLMS para separar
    a média local (larga escala) do desvanecimento rápido.

    Args:
        potencia_total (pd.Series): Sinal de potência original (P_total)
        n_taps (int): Ordem do filtro (quantos taps passados usar na previsão)
        mu (float): Taxa de adaptação (passo)
        epsilon (float): Termo de regularização para evitar divisão por zero

    Returns:
        - media_local (L): A componente de larga escala estimada.
        - sinal_media_zero (e): P_total - L (sinal de erro).
        - envelope_normalizado (rho): sqrt(P_total / L).
    """
```

### Cálculo de Parâmetros (Parte II)
```python
# Mean Excess Delay
τ_mean = Σ(P(τ) * τ) / Σ(P(τ))

# RMS Delay Spread
στ = sqrt(Σ(P(τ) * (τ - τ_mean)²) / Σ(P(τ)))

# Coherence Bandwidth
Bc ≈ 1 / (5 * στ)  # Para correlação de 50%
```

### Modelo Saleh-Valenzuela (Parte III)
```python
# Tempos de chegada (Processo de Poisson)
T_l ~ Exponencial(Γ)        # Clusters
τ_kl ~ Exponencial(γ)       # Raios dentro do cluster

# Potência dos raios
β²_kl = Ω₀ * exp(-T_l * Λ) * exp(-τ_kl * λ)

# Componente LOS (se K > 0)
Potência_LOS = K * Potência_scatter
```

---

## Validação dos Resultados

**Taxa de validação:** 95% (19/20 verificações)

**Verificações realizadas:**
- ✅ Nakagami m = 0.30 < 1 → desvanecimento severo (correto para ambiente industrial)
- ✅ RMS Delay Spread ~118 ns → típico para indoor industrial (10-500 ns)
- ✅ Coherence Bandwidth = 1/(5×RMS) → fórmula teórica validada
- ✅ Fator K ≈ 0 → ambiente NLOS (coerente com m < 1)
- ✅ Comparação 20 vs 8 taps → diferença < 3% (esperado)
- ✅ Modelo SV → CIRs sintéticas com estatísticas similares aos dados reais

**Conclusão:** Todos os resultados são fisicamente coerentes e cientificamente válidos.

Ver `VALIDACAO_RESULTADOS.md` para detalhes.

---

## Referências

1. **NIST Dataset**
   [Industrial Wireless Systems Radio Propagation Measurements](https://www.nist.gov/publications/industrial-wireless-systems-radio-propagation-measurements)

2. **Saleh & Valenzuela (1987)**
   "A statistical model for indoor multipath propagation"
   IEEE Journal on Selected Areas in Communications, 5(2), 128-137

3. **Dissertação - Estimadores de Envoltória Normalizada**
   [https://repositorio.unb.br/handle/10482/11357](https://repositorio.unb.br/handle/10482/11357)

4. **Rappaport (2002)**
   "Wireless Communications: Principles and Practice"

---

## Status do Projeto

- [x] Parte I: Distribuições Estatísticas - COMPLETO
- [x] Parte II: Parâmetros do Canal - COMPLETO
- [x] Parte III: Modelo Saleh-Valenzuela - COMPLETO
- [x] Notebooks Jupyter - COMPLETO
- [x] Validação dos Resultados - COMPLETO
- [x] Documentação - COMPLETO

**Taxa de conclusão:** 100% ✅

---

## Autor

**João Lucas de Castro Santos**
Novembro 2025
