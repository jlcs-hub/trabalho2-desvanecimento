"""
Script mestre para executar todas as partes do trabalho
Parte I + Parte II + Parte III

Autor: João Lucas de Castro Santos
Data: 2025-11-05
"""

import subprocess
import sys
import time
from pathlib import Path


def print_header(text: str):
    """Imprime cabeçalho formatado"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def run_script(script_path: str, description: str) -> bool:
    """
    Executa um script Python

    Parameters
    ----------
    script_path : str
        Caminho para o script
    description : str
        Descrição do script

    Returns
    -------
    bool
        True se executou com sucesso, False caso contrário
    """
    print(f"▶ Executando: {description}")
    print(f"  Script: {script_path}")
    print(f"  Início: {time.strftime('%H:%M:%S')}\n")

    start_time = time.time()

    try:
        # Executar script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos de timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {description} - SUCESSO")
            print(f"  Tempo de execução: {elapsed:.1f}s")
            return True
        else:
            print(f"\n✗ {description} - FALHOU")
            print(f"  Código de retorno: {result.returncode}")
            print(f"\nErro:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} - TIMEOUT")
        print(f"  Execução excedeu 10 minutos")
        return False

    except Exception as e:
        print(f"\n✗ {description} - ERRO")
        print(f"  Exceção: {e}")
        return False


def check_requirements():
    """Verifica se os arquivos de dados existem"""
    print_header("VERIFICANDO REQUISITOS")

    required_files = [
        'data/raw/cirNTaps20SteamPlant.csv',
        'data/raw/cirNTaps8SteamPlant.csv'
    ]

    all_ok = True

    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NÃO ENCONTRADO")
            all_ok = False

    if not all_ok:
        print("\n⚠ Alguns arquivos de dados não foram encontrados!")
        return False

    print("\n✓ Todos os requisitos atendidos")
    return True


def main():
    """Função principal"""
    print_header("TRABALHO CIR - EXECUÇÃO COMPLETA")

    print("Este script executará todas as 3 partes do trabalho:")
    print("  1. Parte I: Distribuições Estatísticas")
    print("  2. Parte II: Parâmetros do Canal")
    print("  3. Parte III: Modelo Saleh-Valenzuela")

    # Verificar requisitos
    if not check_requirements():
        print("\n⚠ Não é possível continuar sem os arquivos necessários.")
        sys.exit(1)

    # Scripts a executar
    scripts = [
        ('scripts/parte1_distribuicoes.py', 'Parte I - Distribuições Estatísticas'),
        ('scripts/parte2_parametros_canal.py', 'Parte II - Parâmetros do Canal'),
        ('scripts/parte3_modelo_sv.py', 'Parte III - Modelo Saleh-Valenzuela')
    ]

    results = []
    start_total = time.time()

    # Executar cada script
    for script_path, description in scripts:
        print_header(description)

        success = run_script(script_path, description)
        results.append((description, success))

        if not success:
            print("\n⚠ Erro ao executar script. Deseja continuar? (s/n): ")
            # response = input().lower()
            # if response != 's':
            #     print("\n⚠ Execução interrompida pelo usuário.")
            #     break

    total_time = time.time() - start_total

    # Resumo
    print_header("RESUMO DA EXECUÇÃO")

    print("Resultados:")
    print("─" * 80)

    for description, success in results:
        status = "✓ SUCESSO" if success else "✗ FALHOU"
        print(f"  {description:50s} {status}")

    print("─" * 80)

    # Estatísticas
    n_success = sum(1 for _, success in results if success)
    n_total = len(results)

    print(f"\nEstatísticas:")
    print(f"  Total de scripts: {n_total}")
    print(f"  Sucessos: {n_success}")
    print(f"  Falhas: {n_total - n_success}")
    print(f"  Tempo total: {total_time/60:.1f} minutos")

    # Status final
    if n_success == n_total:
        print("\n" + "="*80)
        print("✓ TODAS AS PARTES FORAM EXECUTADAS COM SUCESSO!".center(80))
        print("="*80 + "\n")

        print("Arquivos gerados:")
        print("\n  Figuras:")
        print("    Parte I: figures/parte1/ (6 gráficos)")
        print("    Parte II: figures/parte2/ (4 gráficos)")
        print("    Parte III: figures/parte3/ (8 gráficos)")
        print("\n  Animações:")
        print("    media/cir_animation_20taps.gif")
        print("    media/cir_animation_8taps.gif")
        print("    media/cir_comparison_20vs8.gif")
        print("\n  Resultados:")
        print("    results/parte1_parametros.csv")
        print("    results/parte2_parametros.csv")
        print("    results/parte2_comparacao.csv")
        print("    results/parte3_exemplos_configs.csv")
        print("    results/parte3_pdp_medio.csv")

        print("\n✓ Próximos passos:")
        print("  1. Revisar figuras e resultados")
        print("  2. Preencher notebook Jupyter")
        print("  3. Preparar apresentação")
        print("  4. Gravar vídeo de 15 minutos")

        return 0
    else:
        print("\n" + "="*80)
        print("⚠ ALGUMAS PARTES FALHARAM".center(80))
        print("="*80 + "\n")
        print("Verifique os erros acima e execute os scripts individualmente.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
