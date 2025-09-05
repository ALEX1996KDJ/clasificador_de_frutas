#!/usr/bin/env python3
"""
RECUPERAR_HISTORIAL.py - Visualiza tu entrenamiento anterior SIN reentrenar
"""

import numpy as np
from visualize_training import visualize_training_results

# ==============================================================================
# DATOS DE TU ENTRENAMIENTO ANTERIOR (de los logs que me compartiste)
# ==============================================================================
history_data = {
    'accuracy': [0.6554, 0.7960, 0.8207, 0.8821, 0.8997, 0.9061, 0.9104, 0.9156],
    'val_accuracy': [0.8238, 0.8577, 0.8700, 0.9025, 0.9135, 0.9086, 0.9128, 0.9134],
    'loss': [1.2536, 0.6594, 0.5864, 0.3665, 0.3069, 0.2858, 0.2722, 0.2542],
    'val_loss': [0.5615, 0.4482, 0.4219, 0.3047, 0.2757, 0.2872, 0.2707, 0.2663],
    'top_k_categorical_accuracy': [0.8873, 0.9723, 0.9788] + [np.nan]*5,
    'val_top_k_categorical_accuracy': [0.9758, 0.9843, 0.9869] + [np.nan]*5
}

# Crear objeto history compatible
class MockHistory:
    def __init__(self, history_dict):
        self.history = history_dict

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("📊 RECUPERANDO HISTORIAL DE ENTRENAMIENTO ANTERIOR")
    print("=" * 60)
    
    # Crear objeto history
    history = MockHistory(history_data)
    
    # Visualizar resultados (test_accuracy = 0.9403 de tu entrenamiento)
    try:
        visualize_training_results(history, test_accuracy=0.9403, model_name="MobileNetV2")
        print("✅ Visualización completada!")
        print("📁 Revisa la carpeta 'training_results/'")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de tener el archivo visualize_training.py")