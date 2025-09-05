#!/usr/bin/env python3
"""
VISUALIZACIÓN DE RESULTADOS DE ENTRENAMIENTO
Guarda gráficas y reporte completo de métricas
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

def visualize_training_results(history, test_accuracy=0.9403, model_name="MobileNetV2"):
    """
    CREA GRÁFICAS Y REPORTE COMPLETO DEL ENTRENAMIENTO
    """
    print("📊 Generando visualización de resultados...")
    
    # Crear directorio de resultados
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. GRÁFICA DE ACCURACY
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title(f'Accuracy - {model_name}\nFinal Test: {test_accuracy:.2%}', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. GRÁFICA DE LOSS
    plt.subplot(2, 2, 2)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='red')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
    plt.title('Loss durante Entrenamiento', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. GRÁFICA DE TOP-5 ACCURACY (si existe)
    plt.subplot(2, 2, 3)
    if 'top_k_categorical_accuracy' in history.history:
        plt.plot(history.history['top_k_categorical_accuracy'], 
                label='Training Top-5', linewidth=2, color='green')
    if 'val_top_k_categorical_accuracy' in history.history:
        plt.plot(history.history['val_top_k_categorical_accuracy'], 
                label='Validation Top-5', linewidth=2, color='blue')
    plt.title('Top-5 Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Top-5 Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. GRÁFICA COMPARATIVA FINAL
    plt.subplot(2, 2, 4)
    metrics = ['Training', 'Validation', 'Test']
    
    # Obtener últimos valores disponibles
    train_acc = history.history['accuracy'][-1] if 'accuracy' in history.history else 0
    val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
    
    values = [train_acc, val_acc, test_accuracy]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = plt.bar(metrics, values, color=colors, alpha=0.8)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Comparación Final de Accuracy', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/training_results_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{results_dir}/training_results_{timestamp}.pdf')
    plt.close()
    
    # 5. REPORTE DETALLADO EN TEXTO
    create_text_report(history, test_accuracy, model_name, results_dir, timestamp)
    
    print(f"✅ Resultados guardados en: {results_dir}/")
    print(f"   📈 Gráficas: training_results_{timestamp}.png")
    print(f"   📝 Reporte: training_report_{timestamp}.txt")

def create_text_report(history, test_accuracy, model_name, results_dir, timestamp):
    """Crea un reporte detallado en texto"""
    
    report = f"""
{'='*60}
📊 REPORTE DE ENTRENAMIENTO - {model_name}
{'='*60}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Modelo: {model_name}
Total Epochs: {len(history.history['accuracy']) if 'accuracy' in history.history else 0}

{'='*60}
🎯 MÉTRICAS FINALES:
{'='*60}
• Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})
• Training Accuracy: {history.history['accuracy'][-1] if 'accuracy' in history.history else 'N/A':.4f}
• Validation Accuracy: {history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 'N/A':.4f}
• Training Loss: {history.history['loss'][-1] if 'loss' in history.history else 'N/A':.4f}
• Validation Loss: {history.history['val_loss'][-1] if 'val_loss' in history.history else 'N/A':.4f}

{'='*60}
📈 EVOLUCIÓN:
{'='*60}
"""

    # Añadir métricas por epoch
    if 'accuracy' in history.history:
        for i in range(len(history.history['accuracy'])):
            report += f"Epoch {i+1}:\n"
            report += f"  Accuracy: {history.history['accuracy'][i]:.4f} "
            if 'val_accuracy' in history.history:
                report += f"-> Val: {history.history['val_accuracy'][i]:.4f}\n"
            else:
                report += "\n"
                
            if 'loss' in history.history:
                report += f"  Loss:     {history.history['loss'][i]:.4f} "
                if 'val_loss' in history.history:
                    report += f"-> Val: {history.history['val_loss'][i]:.4f}\n"
                else:
                    report += "\n"
            
            if 'top_k_categorical_accuracy' in history.history and i < len(history.history['top_k_categorical_accuracy']):
                report += f"  Top-5:    {history.history['top_k_categorical_accuracy'][i]:.4f} "
                if 'val_top_k_categorical_accuracy' in history.history and i < len(history.history['val_top_k_categorical_accuracy']):
                    report += f"-> Val: {history.history['val_top_k_categorical_accuracy'][i]:.4f}\n"
                else:
                    report += "\n"
            
            report += "\n"

    report += f"""
{'='*60}
📋 ANÁLISIS:
{'='*60}
"""
    
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        overfitting = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
        report += f"""• Overfitting: {'SÍ' if overfitting > 0.05 else 'NO'} 
  (Diferencia: {overfitting:.4f})

• Mejor Epoch Validation: Epoch {np.argmax(history.history['val_accuracy']) + 1}
  con Accuracy: {np.max(history.history['val_accuracy']):.4f}

• Mejor Epoch Training: Epoch {np.argmax(history.history['accuracy']) + 1}
  con Accuracy: {np.max(history.history['accuracy']):.4f}
"""
    
    # Guardar reporte
    with open(f'{results_dir}/training_report_{timestamp}.txt', 'w') as f:
        f.write(report)

def save_training_history(history, test_accuracy, model_name):
    """Guarda el historial completo en JSON"""
    
    history_dict = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'test_accuracy': float(test_accuracy),
        'training_history': {},
        'final_metrics': {
            'train_accuracy': float(history.history['accuracy'][-1]) if 'accuracy' in history.history else 0,
            'val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else 0,
            'train_loss': float(history.history['loss'][-1]) if 'loss' in history.history else 0,
            'val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else 0
        }
    }
    
    # Copiar todas las métricas del history
    for key, values in history.history.items():
        history_dict['training_history'][key] = [float(v) if not np.isnan(v) else None for v in values]
    
    # Guardar JSON
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print("💾 Historial guardado en: training_history.json")

# ==============================================================================
# EJECUCIÓN DIRECTA (para testing)
# ==============================================================================
if __name__ == "__main__":
    print("📊 Módulo de visualización de entrenamiento")
    print("💡 Importa este módulo en tu script de entrenamiento")