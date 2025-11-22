import time
import random
import tracemalloc
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np

class MatrizSparse:
    """Matriz dispersa - solo guarda elementos != 0"""
    
    def __init__(self, n):
        self.n = n
        self.datos = {}  
    
    def set(self, i, j, valor):
        """Poner un valor"""
        if valor != 0:
            self.datos[(i, j)] = valor
    
    def get(self, i, j):
        """Obtener un valor"""
        return self.datos.get((i, j), 0.0)
    
    def cuantos_no_cero(self):
        """Contar elementos != 0"""
        return len(self.datos)
    
    def porcentaje_ceros(self):
        """Calcular % de ceros"""
        total = self.n * self.n
        no_ceros = len(self.datos)
        return (1 - no_ceros/total) * 100


def crear_sparse_random(n, sparsity):
    """
    Crea matriz dispersa aleatoria
    sparsity = % de ceros (ej: 0.9 = 90% ceros)
    """
    matriz = MatrizSparse(n)
    total = n * n
    no_ceros = int(total * (1 - sparsity))
    
    posiciones = set()
    while len(posiciones) < no_ceros:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        posiciones.add((i, j))
    
    for (i, j) in posiciones:
        valor = random.uniform(1, 10)
        matriz.set(i, j, valor)
    
    return matriz


def multiplicar_sparse(A, B):
    """Multiplica dos matrices dispersas"""
    C = MatrizSparse(A.n)
    
    for (i, k), valor_a in A.datos.items():
        for (k2, j), valor_b in B.datos.items():
            if k == k2:  
                actual = C.get(i, j)
                C.set(i, j, actual + valor_a * valor_b)
    
    return C


def multiplicar_denso(A, B):
    """Convierte a denso, multiplica, vuelve a sparse"""
    n = A.n
    
    A_denso = [[A.get(i, j) for j in range(n)] for i in range(n)]
    B_denso = [[B.get(i, j) for j in range(n)] for i in range(n)]
    
    C_denso = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C_denso[i][j] += A_denso[i][k] * B_denso[k][j]
    
    C = MatrizSparse(n)
    for i in range(n):
        for j in range(n):
            if C_denso[i][j] != 0:
                C.set(i, j, C_denso[i][j])
    
    return C



def probar_sparsity(n, sparsity):
    """Prueba un nivel de sparsity"""
    print(f"\nSparsity {sparsity*100:.0f}%:")
    
    A = crear_sparse_random(n, sparsity)
    B = crear_sparse_random(n, sparsity)
    
    print(f"  Elementos no-cero en A: {A.cuantos_no_cero()}")
    print(f"  Elementos no-cero en B: {B.cuantos_no_cero()}")
    
    proceso = psutil.Process(os.getpid())
    
    tracemalloc.start()
    
    cpu_antes = proceso.cpu_percent(interval=0.1)
    
    inicio = time.time()
    C = multiplicar_sparse(A, B)
    fin = time.time()
    
    cpu_despues = proceso.cpu_percent(interval=0.1)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    tiempo = fin - inicio
    memoria_mb = peak / (1024 * 1024)
    cpu_promedio = (cpu_antes + cpu_despues) / 2
    
    print(f"  Tiempo: {tiempo:.4f}s | Memoria: {memoria_mb:.2f}MB | CPU: {cpu_promedio:.1f}%")
    print(f"  Resultado: {C.cuantos_no_cero()} elementos no-cero")
    
    return {
        'tiempo': tiempo, 
        'memoria': memoria_mb,
        'cpu': cpu_promedio,
        'nz_resultado': C.cuantos_no_cero()
    }


def comparar_sparsity_levels(n):
    """Compara diferentes niveles de sparsity"""
    print(f"\n{'='*80}")
    print(f"EFECTO DE SPARSITY - Matrices {n}×{n}")
    print(f"{'='*80}")
    
    niveles = [0.7, 0.9, 0.95, 0.99]
    resultados = []
    
    for sparsity in niveles:
        try:
            resultado = probar_sparsity(n, sparsity)
            resultados.append({'sparsity': sparsity*100, **resultado})
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    if resultados:
        print(f"\n{'RESUMEN:'}")
        print(f"{'='*80}")
        print(f"{'Sparsity':<12} {'Tiempo (s)':<12} {'Memoria (MB)':<15} {'CPU %':<10} {'Mejora':<10}")
        print(f"{'-'*80}")
        tiempo_base = resultados[0]['tiempo']
        for r in resultados:
            mejora = tiempo_base / r['tiempo']
            print(f"{r['sparsity']:<12.0f} {r['tiempo']:<12.4f} {r['memoria']:<15.2f} "
                  f"{r['cpu']:<10.1f} {mejora:<10.2f}x")
    
    return resultados



def comparar_sparse_vs_denso(n, sparsity):
    """Compara sparse vs denso"""
    print(f"\n{'='*80}")
    print(f"SPARSE vs DENSO - {n}×{n}, Sparsity {sparsity*100:.0f}%")
    print(f"{'='*80}")
    
    print("Creando matrices...")
    A = crear_sparse_random(n, sparsity)
    B = crear_sparse_random(n, sparsity)
    
    proceso = psutil.Process(os.getpid())
    
    print("\n1. Método DENSO:")
    tracemalloc.start()
    cpu_antes = proceso.cpu_percent(interval=0.1)
    
    inicio = time.time()
    C_denso = multiplicar_denso(A, B)
    tiempo_denso = time.time() - inicio
    
    cpu_despues = proceso.cpu_percent(interval=0.1)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memoria_denso = peak / (1024 * 1024)
    cpu_denso = (cpu_antes + cpu_despues) / 2
    
    print(f"   Tiempo: {tiempo_denso:.4f}s | Memoria: {memoria_denso:.2f}MB | CPU: {cpu_denso:.1f}%")
    
    print("\n2. Método SPARSE:")
    tracemalloc.start()
    cpu_antes = proceso.cpu_percent(interval=0.1)
    
    inicio = time.time()
    C_sparse = multiplicar_sparse(A, B)
    tiempo_sparse = time.time() - inicio
    
    cpu_despues = proceso.cpu_percent(interval=0.1)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memoria_sparse = peak / (1024 * 1024)
    cpu_sparse = (cpu_antes + cpu_despues) / 2
    
    print(f"   Tiempo: {tiempo_sparse:.4f}s | Memoria: {memoria_sparse:.2f}MB | CPU: {cpu_sparse:.1f}%")
    
    speedup = tiempo_denso / tiempo_sparse
    ahorro_memoria = ((memoria_denso - memoria_sparse) / memoria_denso) * 100
    
    print(f"\n{'='*80}")
    print(f"RESULTADO:")
    print(f"  • Tiempo: Sparse es {speedup:.2f}x más rápido")
    print(f"  • Memoria: Sparse ahorra {ahorro_memoria:.1f}% de memoria")
    print(f"{'='*80}")
    
    return {
        'denso': {'tiempo': tiempo_denso, 'memoria': memoria_denso, 'cpu': cpu_denso},
        'sparse': {'tiempo': tiempo_sparse, 'memoria': memoria_sparse, 'cpu': cpu_sparse},
        'speedup': speedup,
        'ahorro_memoria': ahorro_memoria
    }



def generar_graficas(resultados_sparsity, resultado_comparacion):
    """Genera gráficas de matrices dispersas"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    if resultados_sparsity:
        sparsities = [r['sparsity'] for r in resultados_sparsity]
        tiempos = [r['tiempo'] for r in resultados_sparsity]
        
        ax1.plot(sparsities, tiempos, 'o-', linewidth=2, markersize=10, color='#2E86AB')
        ax1.set_xlabel('Sparsity Level (%)', fontsize=12)
        ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax1.set_title('Execution Time vs Sparsity Level', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Escala logarítmica
        
        for i, (s, t) in enumerate(zip(sparsities, tiempos)):
            ax1.annotate(f'{t:.2f}s', (s, t), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    if resultados_sparsity:
        memorias = [r['memoria'] for r in resultados_sparsity]
        
        ax2.plot(sparsities, memorias, 's-', linewidth=2, markersize=10, color='#A23B72')
        ax2.set_xlabel('Sparsity Level (%)', fontsize=12)
        ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax2.set_title('Memory Usage vs Sparsity Level', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        for i, (s, m) in enumerate(zip(sparsities, memorias)):
            ax2.annotate(f'{m:.1f}MB', (s, m), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    if resultados_sparsity and len(resultados_sparsity) > 0:
        tiempo_base = resultados_sparsity[0]['tiempo']
        speedups = [tiempo_base / r['tiempo'] for r in resultados_sparsity]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sparsities)))
        bars = ax3.bar(range(len(sparsities)), speedups, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Sparsity Level (%)', fontsize=12)
        ax3.set_ylabel('Speedup (vs 70% sparsity)', fontsize=12)
        ax3.set_title('Performance Improvement with Sparsity', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(sparsities)))
        ax3.set_xticklabels([f'{int(s)}%' for s in sparsities])
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if resultado_comparacion:
        categorias = ['Time (s)', 'Memory (MB)']
        dense_vals = [resultado_comparacion['denso']['tiempo'], 
                     resultado_comparacion['denso']['memoria']]
        sparse_vals = [resultado_comparacion['sparse']['tiempo'], 
                      resultado_comparacion['sparse']['memoria']]
        
        x = np.arange(len(categorias))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, dense_vals, width, label='Dense', alpha=0.8, color='#E63946')
        bars2 = ax4.bar(x + width/2, sparse_vals, width, label='Sparse', alpha=0.8, color='#06A77D')
        
        ax4.set_ylabel('Value', fontsize=12)
        ax4.set_title('Sparse vs Dense Comparison (90% sparsity)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categorias)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        speedup_text = f'Sparse is {resultado_comparacion["speedup"]:.2f}x faster'
        ax4.text(0.5, 0.95, speedup_text, transform=ax4.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('sparse_matrices_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfica guardada: sparse_matrices_results.png")
    plt.show()



if __name__ == "__main__":
    print("="*80)
    print("MATRICES DISPERSAS - SIMPLE")
    print("="*80)
    
    resultados_sparsity = comparar_sparsity_levels(500)
    
    resultado_comparacion = comparar_sparse_vs_denso(500, 0.9)
    
    print("\n" + "="*80)
    print("CONCLUSIONES FINALES")
    print("="*80)
    
    if resultados_sparsity and len(resultados_sparsity) >= 2:
        tiempo_70 = resultados_sparsity[0]['tiempo']
        tiempo_99 = resultados_sparsity[-1]['tiempo']
        memoria_70 = resultados_sparsity[0]['memoria']
        memoria_99 = resultados_sparsity[-1]['memoria']
        mejora_tiempo = tiempo_70 / tiempo_99
        ahorro_mem = ((memoria_70 - memoria_99) / memoria_70) * 100
        
        print(f"• Mejora de 70% a 99% sparsity:")
        print(f"  - Tiempo: {mejora_tiempo:.2f}x más rápido")
        print(f"  - Memoria: {ahorro_mem:.1f}% menos memoria")
    
    if resultado_comparacion:
        print(f"\n• Sparse vs Denso (90% sparsity):")
        print(f"  - Tiempo: {resultado_comparacion['speedup']:.2f}x más rápido")
        print(f"  - Memoria: {resultado_comparacion['ahorro_memoria']:.1f}% menos memoria")
    
    print("\nRECOMENDACIONES:")
    print("  → Usar SPARSE cuando sparsity > 90%")
    print("  → Usar DENSO cuando sparsity < 70%")
    print("  → Sparse ahorra MEMORIA y TIEMPO cuando hay muchos ceros")
    print("="*80)
    
    generar_graficas(resultados_sparsity, resultado_comparacion)
    
    print("\n✓ COMPLETADO")

    print("="*80)
