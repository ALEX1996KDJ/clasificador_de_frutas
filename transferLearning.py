#!/usr/bin/env python3
"""
TRANSFER LEARNING PARA FRUIT360
Usa una red pre-entrenada para entrenar en minutos instead de horas
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess_data import preprocess_fruit360_data
import numpy as np

def create_transfer_learning_model(base_model_name='EfficientNetB0', num_classes=208):
    """
    CREA MODELO DE TRANSFER LEARNING
    """
    print(f"🧠 Creando modelo de transfer learning con {base_model_name}")
    
    # Seleccionar modelo base pre-entrenado
    if base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(100, 100, 3)
        )
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(100, 100, 3)
        )
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(100, 100, 3)
        )
    else:
        raise ValueError("Modelo no soportado")
    
    # Congelar capas del modelo base
    base_model.trainable = False
    
    # Añadir capas personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Modelo final
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

def train_transfer_learning(epochs=10, batch_size=64, base_model='EfficientNetB0'):
    """
    ENTRENAMIENTO CON TRANSFER LEARNING
    """
    print("🍎 TRANSFER LEARNING - FRUIT360")
    print("=" * 50)
    
    # 1. Cargar datos
    print("📥 Cargando datos...")
    train_gen, val_gen, test_gen, classes, num_classes = preprocess_fruit360_data(
        batch_size=batch_size
    )
    
    # 2. Crear modelo
    model = create_transfer_learning_model(base_model, num_classes)
    model.summary()
    
    # 3. Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('transfer_learning_best.h5', save_best_only=True)
    ]
    
    # 4. Entrenar solo las capas nuevas (rápido)
    print("🚀 Entrenando capas nuevas...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=2  # Métricas por época
    )
    
    # 5. Fine-tuning (opcional)
    print("🔧 Fine-tuning (opcional)...")
    # Descongelar algunas capas
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar un poco más
    history_ft = model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        verbose=2
    )
    
    # 6. Evaluar
    print("📊 Evaluando modelo...")
    results = model.evaluate(test_gen, verbose=0)
    print(f"Test accuracy: {results[1]:.4f}")
    print(f"Top-5 accuracy: {results[2]:.4f}")
    
    # 7. Guardar
    model.save('fruit360_transfer_learning.h5')
    print("💾 Modelo guardado: fruit360_transfer_learning.h5")
    
    return model, history

# ==============================================================================
# EJECUCIÓN RÁPIDA
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='EfficientNetB0')
    
    args = parser.parse_args()
    
    print(f"⚡ Transfer Learning - {args.model}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    
    # Entrenar
    model, history = train_transfer_learning(
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_model=args.model
    )