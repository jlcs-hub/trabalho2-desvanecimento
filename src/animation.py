"""
Módulo para criação de animações de CIR
Parte II do trabalho

Autor: João Lucas de Castro Santos
Data: 2025-11-05
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
import warnings


def create_cir_animation(cirs: np.ndarray,
                        delays: np.ndarray,
                        n_frames: int = 100,
                        output_path: str = 'media/cir_animation.gif',
                        fps: int = 10,
                        title: str = 'CIR Evolution',
                        random_seed: int = 42) -> None:
    """
    Cria animação mostrando variação das CIRs

    Parameters
    ----------
    cirs : np.ndarray
        Array 2D com CIRs (linhas = CIRs, colunas = taps)
    delays : np.ndarray
        Array com os delays/atrasos (em ns)
    n_frames : int
        Número de frames na animação
    output_path : str
        Caminho para salvar animação (.gif ou .mp4)
    fps : int
        Frames por segundo
    title : str
        Título do gráfico
    random_seed : int
        Seed para seleção aleatória de CIRs
    """
    # Selecionar subset de CIRs aleatoriamente
    np.random.seed(random_seed)
    n_cirs_total = cirs.shape[0]
    n_frames_actual = min(n_frames, n_cirs_total)

    indices = np.random.choice(n_cirs_total, size=n_frames_actual, replace=False)
    cirs_subset = cirs[indices]

    # Calcular PDP médio para referência
    pdp_mean = np.mean(cirs, axis=0)

    # Configurar figura
    fig, ax = plt.subplots(figsize=(12, 6))

    # Limites do eixo Y (baseado no max de todas CIRs)
    y_max = np.max(cirs_subset) * 1.1
    y_min = 0

    # Plot inicial
    line_cir, = ax.plot([], [], 'b-o', linewidth=2, markersize=6,
                        label='CIR atual', alpha=0.7)
    line_pdp, = ax.plot(delays, pdp_mean, 'r--', linewidth=2,
                        label='PDP médio', alpha=0.8)

    ax.set_xlim(delays.min() - 10, delays.max() + 10)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Delay (ns)', fontsize=12)
    ax.set_ylabel('Potência Normalizada', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Texto para número do frame
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                  fontsize=12, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        """Inicialização da animação"""
        line_cir.set_data([], [])
        text.set_text('')
        return line_cir, text

    def update(frame):
        """Atualiza frame da animação"""
        # Atualizar CIR
        cir_current = cirs_subset[frame]
        line_cir.set_data(delays, cir_current)

        # Atualizar texto
        text.set_text(f'CIR #{indices[frame]} / Frame {frame+1}/{n_frames_actual}')

        return line_cir, text

    # Criar animação
    print(f"Criando animação com {n_frames_actual} frames...")
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=n_frames_actual,
                                  interval=1000/fps,  # ms
                                  blit=True,
                                  repeat=True)

    # Salvar animação
    print(f"Salvando animação em: {output_path}")

    try:
        if output_path.endswith('.gif'):
            # Salvar como GIF
            anim.save(output_path, writer='pillow', fps=fps)
        elif output_path.endswith('.mp4'):
            # Salvar como MP4 (requer ffmpeg)
            anim.save(output_path, writer='ffmpeg', fps=fps, bitrate=1800)
        else:
            warnings.warn(f"Formato desconhecido, salvando como GIF")
            output_path_gif = output_path.replace(output_path.split('.')[-1], 'gif')
            anim.save(output_path_gif, writer='pillow', fps=fps)

        print(f"✓ Animação salva com sucesso!")

    except Exception as e:
        print(f"ERRO ao salvar animação: {e}")
        print("Tentando salvar como GIF...")
        try:
            output_path_gif = output_path.replace('.mp4', '.gif')
            anim.save(output_path_gif, writer='pillow', fps=fps)
            print(f"✓ Animação salva como GIF: {output_path_gif}")
        except Exception as e2:
            print(f"ERRO ao salvar como GIF: {e2}")

    plt.close(fig)


def create_pdp_comparison_animation(cirs_20: np.ndarray,
                                   delays_20: np.ndarray,
                                   cirs_8: np.ndarray,
                                   delays_8: np.ndarray,
                                   n_frames: int = 100,
                                   output_path: str = 'media/pdp_comparison.gif',
                                   fps: int = 10) -> None:
    """
    Cria animação comparando CIRs de 20 e 8 taps lado a lado

    Parameters
    ----------
    cirs_20 : np.ndarray
        CIRs de 20 taps
    delays_20 : np.ndarray
        Delays de 20 taps
    cirs_8 : np.ndarray
        CIRs de 8 taps
    delays_8 : np.ndarray
        Delays de 8 taps
    n_frames : int
        Número de frames
    output_path : str
        Caminho para salvar
    fps : int
        Frames por segundo
    """
    # Selecionar subset
    np.random.seed(42)
    n_frames_actual = min(n_frames, cirs_20.shape[0], cirs_8.shape[0])
    indices = np.random.choice(min(cirs_20.shape[0], cirs_8.shape[0]),
                              size=n_frames_actual, replace=False)

    cirs_20_subset = cirs_20[indices]
    cirs_8_subset = cirs_8[indices]

    # PDP médios
    pdp_20 = np.mean(cirs_20, axis=0)
    pdp_8 = np.mean(cirs_8, axis=0)

    # Configurar figura (2 subplots lado a lado)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Limites Y
    y_max = max(np.max(cirs_20_subset), np.max(cirs_8_subset)) * 1.1

    # Subplot 1: 20 taps
    line_20, = ax1.plot([], [], 'b-o', linewidth=2, markersize=6,
                       label='CIR atual', alpha=0.7)
    ax1.plot(delays_20, pdp_20, 'r--', linewidth=2, label='PDP médio', alpha=0.8)
    ax1.set_xlim(delays_20.min() - 10, delays_20.max() + 10)
    ax1.set_ylim(0, y_max)
    ax1.set_xlabel('Delay (ns)', fontsize=12)
    ax1.set_ylabel('Potência Normalizada', fontsize=12)
    ax1.set_title('20 Taps', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: 8 taps
    line_8, = ax2.plot([], [], 'g-o', linewidth=2, markersize=6,
                      label='CIR atual', alpha=0.7)
    ax2.plot(delays_8, pdp_8, 'r--', linewidth=2, label='PDP médio', alpha=0.8)
    ax2.set_xlim(delays_8.min() - 10, delays_8.max() + 10)
    ax2.set_ylim(0, y_max)
    ax2.set_xlabel('Delay (ns)', fontsize=12)
    ax2.set_ylabel('Potência Normalizada', fontsize=12)
    ax2.set_title('8 Taps', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Título geral
    fig.suptitle('Comparação: Resolução de CIR (20 vs 8 taps)',
                fontsize=16, fontweight='bold')

    # Texto
    text = fig.text(0.5, 0.02, '', ha='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        line_20.set_data([], [])
        line_8.set_data([], [])
        text.set_text('')
        return line_20, line_8, text

    def update(frame):
        # Atualizar CIRs
        line_20.set_data(delays_20, cirs_20_subset[frame])
        line_8.set_data(delays_8, cirs_8_subset[frame])

        # Atualizar texto
        text.set_text(f'Frame {frame+1}/{n_frames_actual}')

        return line_20, line_8, text

    # Criar e salvar animação
    print(f"Criando animação comparativa com {n_frames_actual} frames...")
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=n_frames_actual,
                                  interval=1000/fps,
                                  blit=True,
                                  repeat=True)

    print(f"Salvando em: {output_path}")

    try:
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            output_path_gif = output_path.replace('.mp4', '.gif')
            anim.save(output_path_gif, writer='pillow', fps=fps)

        print(f"✓ Animação salva com sucesso!")

    except Exception as e:
        print(f"ERRO: {e}")

    plt.close(fig)


if __name__ == "__main__":
    print("Testando módulo animation.py\n")

    # Gerar dados de teste
    print("Gerando dados de teste...")
    n_cirs = 200
    n_taps = 10
    delays = np.linspace(0, 100, n_taps)

    # CIRs com PDP exponencial + ruído
    pdp_base = np.exp(-delays / 30)
    cirs_test = np.random.normal(pdp_base, 0.1 * pdp_base, (n_cirs, n_taps))
    cirs_test = np.maximum(cirs_test, 0)

    print(f"  - {n_cirs} CIRs geradas")
    print(f"  - {n_taps} taps\n")

    # Criar animação de teste
    print("Criando animação de teste...")
    create_cir_animation(
        cirs_test,
        delays,
        n_frames=50,
        output_path='media/test_animation.gif',
        fps=10,
        title='Teste de Animação CIR'
    )

    print("\n✓ Módulo testado com sucesso!")
    print("  → Verifique: media/test_animation.gif")
