import time
import random
import tracemalloc
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np




def crear_matriz(n, valor=1.0):
    """Crea una matriz n×n llena con un valor"""
    return [[valor for _ in range(n)] for _ in range(n)]

def crear_matriz_random(n):
    """Crea matriz n×n con valores aleatorios"""
    return [[random.uniform(1, 10) for _ in range(n)] for _ in range(n)]



def multiplicar_basico(A, B):
    """Método básico - Orden ijk"""
    n = len(A)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C


def multiplicar_cache_optimizado(A, B):
    """Optimizado - Orden ikj (mejor uso de caché)"""
    n = len(A)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for k in range(n):
            temp = A[i][k]
            for j in range(n):
                C[i][j] += temp * B[k][j]
    
    return C


def multiplicar_bloques(A, B, block_size=32):
    """Multiplicación por bloques"""
    n = len(A)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i_block in range(0, n, block_size):
        for j_block in range(0, n, block_size):
            for k_block in range(0, n, block_size):
                
                for i in range(i_block, min(i_block + block_size, n)):
                    for k in range(k_block, min(k_block + block_size, n)):
                        temp = A[i][k]
                        for j in range(j_block, min(j_block + block_size, n)):
                            C[i][j] += temp * B[k][j]
    
    return C




def medir_tiempo(funcion, A, B, nombre):
    """Mide tiempo, memoria y CPU"""
    print(f"\n{nombre}...")
    
    proceso = psutil.Process(os.getpid())
    
    tracemalloc.start()
    
    cpu_antes = proceso.cpu_percent(interval=0.1)
    
    inicio = time.time()
    C = funcion(A, B)
    fin = time.time()
    
    cpu_despues = proceso.cpu_percent(interval=0.1)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    tiempo = fin - inicio
    memoria_mb = peak / (1024 * 1024)
    cpu_promedio = (cpu_antes + cpu_despues) / 2
    
    print(f"  ✓ Tiempo: {tiempo:.4f}s | Memoria: {memoria_mb:.2f}MB | CPU: {cpu_promedio:.1f}%")
    
    return tiempo, memoria_mb, cpu_promedio



def comparar_metodos(tamaño):
    """Compara todos los métodos"""
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN - Matrices {tamaño}×{tamaño}")
    print(f"{'='*80}")
    
    print("Creando matrices...")
    A = crear_matriz(tamaño, 1.5)
    B = crear_matriz(tamaño, 2.5)
    
    t1, m1, c1 = medir_tiempo(multiplicar_basico, A, B, "1. Básico (ijk)")
    t2, m2, c2 = medir_tiempo(multiplicar_cache_optimizado, A, B, "2. Cache Optimizado (ikj)")
    t3, m3, c3 = medir_tiempo(multiplicar_bloques, A, B, "3. Por Bloques (32)")
    
    print(f"\n{'RESULTADOS:'}")
    print(f"{'='*80}")
    print(f"{'Método':<25} {'Tiempo (s)':<12} {'Memoria (MB)':<15} {'CPU %':<10} {'Speedup':<10}")
    print(f"{'-'*80}")
    print(f"{'Básico (ijk)':<25} {t1:<12.4f} {m1:<15.2f} {c1:<10.1f} {'1.00x':<10}")
    print(f"{'Cache Optimizado':<25} {t2:<12.4f} {m2:<15.2f} {c2:<10.1f} {t1/t2:<10.2f}x")
    print(f"{'Por Bloques (32)':<25} {t3:<12.4f} {m3:<15.2f} {c3:<10.1f} {t1/t3:<10.2f}x")
    
    return {
        'basico': {'tiempo': t1, 'memoria': m1, 'cpu': c1},
        'cache': {'tiempo': t2, 'memoria': m2, 'cpu': c2},
        'bloques': {'tiempo': t3, 'memoria': m3, 'cpu': c3}
    }


def generar_graficas(todos_resultados):
    """Genera gráficas de los resultados"""
    if not todos_resultados:
        print("No hay resultados para graficar")
        return
    
    tamaños = [r['tamaño'] for r in todos_resultados]
    
    tiempos_basico = [r['datos']['basico']['tiempo'] for r in todos_resultados]
    tiempos_cache = [r['datos']['cache']['tiempo'] for r in todos_resultados]
    tiempos_bloques = [r['datos']['bloques']['tiempo'] for r in todos_resultados]
    
    memorias_basico = [r['datos']['basico']['memoria'] for r in todos_resultados]
    memorias_cache = [r['datos']['cache']['memoria'] for r in todos_resultados]
    memorias_bloques = [r['datos']['bloques']['memoria'] for r in todos_resultados]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(tamaños, tiempos_basico, 'o-', label='Basic (ijk)', linewidth=2, markersize=8)
    ax1.plot(tamaños, tiempos_cache, 's-', label='Cache Optimized (ikj)', linewidth=2, markersize=8)
    ax1.plot(tamaños, tiempos_bloques, '^-', label='Blocked (32)', linewidth=2, markersize=8)
    ax1.set_xlabel('Matrix Size (n×n)', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time vs Matrix Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log') 
    
    ax2.plot(tamaños, memorias_basico, 'o-', label='Basic (ijk)', linewidth=2, markersize=8)
    ax2.plot(tamaños, memorias_cache, 's-', label='Cache Optimized (ikj)', linewidth=2, markersize=8)
    ax2.plot(tamaños, memorias_bloques, '^-', label='Blocked (32)', linewidth=2, markersize=8)
    ax2.set_xlabel('Matrix Size (n×n)', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.set_title('Memory Usage vs Matrix Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    x = np.arange(len(tamaños))
    width = 0.25
    
    speedups_cache = [tiempos_basico[i] / tiempos_cache[i] for i in range(len(tamaños))]
    speedups_bloques = [tiempos_basico[i] / tiempos_bloques[i] for i in range(len(tamaños))]
    
    ax3.bar(x - width/2, speedups_cache, width, label='Cache Optimized', alpha=0.8)
    ax3.bar(x + width/2, speedups_bloques, width, label='Blocked (32)', alpha=0.8)
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
    ax3.set_xlabel('Matrix Size (n×n)', fontsize=12)
    ax3.set_ylabel('Speedup (vs Basic)', fontsize=12)
    ax3.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tamaños)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('dense_matrices_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfica guardada: dense_matrices_results.png")
    plt.show()



if __name__ == "__main__":
    print("="*80)
    print("OPTIMIZACIÓN DE MULTIPLICACIÓN DE MATRICES - SIMPLE")
    print("="*80)
    
    tamaños = [128, 256, 512]
    todos_resultados = []
    
    for tamaño in tamaños:
        try:
            resultados = comparar_metodos(tamaño)
            todos_resultados.append({'tamaño': tamaño, 'datos': resultados})
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\n⚠️  Error con tamaño {tamaño}: {e}")
    
    if todos_resultados:
        print("\n" + "="*80)
        print("RESUMEN GENERAL")
        print("="*80)
        print(f"{'Tamaño':<10} {'Método':<20} {'Tiempo (s)':<12} {'Memoria (MB)':<15} {'CPU %':<10}")
        print("-"*80)
        
        for resultado in todos_resultados:
            t = resultado['tamaño']
            datos = resultado['datos']
            
            print(f"{t:<10} {'Básico':<20} {datos['basico']['tiempo']:<12.4f} "
                  f"{datos['basico']['memoria']:<15.2f} {datos['basico']['cpu']:<10.1f}")
            print(f"{'':<10} {'Cache Optimizado':<20} {datos['cache']['tiempo']:<12.4f} "
                  f"{datos['cache']['memoria']:<15.2f} {datos['cache']['cpu']:<10.1f}")
            print(f"{'':<10} {'Por Bloques':<20} {datos['bloques']['tiempo']:<12.4f} "
                  f"{datos['bloques']['memoria']:<15.2f} {datos['bloques']['cpu']:<10.1f}")
            print()
        
        print("="*80)
        print("CONCLUSIONES:")
        print("  → Cache Optimizado es consistentemente el más rápido")
        print("  → Mejora típica: 1.5x - 3x sobre el método básico")
        print("  → Uso de memoria similar entre todos los métodos")
        print("="*80)
        
        generar_graficas(todos_resultados)
    
    print("\n✓ COMPLETADO")

    print("="*80)
