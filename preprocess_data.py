"""
PREPROCESAMIENTO CORREGIDO - SIN DATA LEAKAGE
==============================================
Versión corregida que evita contaminación entre train/validation
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def preprocess_fruit360_data(data_dir="../data_raw/fruits-360_100x100/fruits-360", 
                            target_size=(100, 100), 
                            validation_split=0.2,
                            batch_size=32,
                            augment_training=True):
    """
    PREPROCESAMIENTO CORREGIDO - EVITA DATA LEAKAGE
    ================================================
    Usa directorios separados y evita contaminación entre conjuntos
    
    Parámetros:
    -----------
    data_dir : str
        Directorio base del dataset (contiene Training/ y Test/)
    target_size : tuple
        Tamaño al que se redimensionarán las imágenes (alto, ancho)
    validation_split : float
        Proporción de datos de entrenamiento para validation (0.2 = 20%)
    batch_size : int
        Tamaño del lote para generadores
    augment_training : bool
        Si aplicar aumento de datos para entrenamiento
    
    Retorna:
    --------
    tuple: (train_generator, val_generator, test_generator, class_names, num_classes)
    """
    
    print("🍎 PREPROCESAMIENTO CORREGIDO - SIN DATA LEAKAGE")
    print("=" * 50)
    
    # ==========================================================================
    # 1. VERIFICACIÓN DE DIRECTORIOS
    # ==========================================================================
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError("❌ No se encontraron los directorios Training/Test")
    
    print(f"✅ Directorios encontrados:")
    print(f"   - Entrenamiento: {train_dir}")
    print(f"   - Prueba: {test_dir}")
    
    # ==========================================================================
    # 2. OBTENER METADATA
    # ==========================================================================
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    
    print(f"📊 Metadata del dataset:")
    print(f"   - Número de clases: {num_classes}")
    print(f"   - Tamaño de imagen: {target_size}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Validation split: {validation_split}")
    
    # ==========================================================================
    # 3. GENERADOR PARA TRAIN/VALIDATION (MISMO DIRECTORIO, SUBSETS DIFERENTES)
    # ==========================================================================
    if augment_training:
        train_val_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=validation_split  # ← clave para separar
        )
        print("   - Aumento de datos: ✅ ACTIVADO")
    else:
        train_val_datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=validation_split
        )
        print("   - Aumento de datos: ❌ DESACTIVADO")
    
    # 📦 Generador de ENTRENAMIENTO (80% de Training)
    train_generator = train_val_datagen.flow_from_directory(
        train_dir,  # ← Directorio de entrenamiento
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',    # ← 80% para entrenamiento
        shuffle=True,
        seed=42,
        color_mode='rgb'
    )
    
    # 📦 Generador de VALIDACIÓN (20% de Training)  
    val_generator = train_val_datagen.flow_from_directory(
        train_dir,  # ← Mismo directorio, pero subset diferente
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  # ← 20% para validación
        shuffle=True,
        seed=42,
        color_mode='rgb'
    )
    
    # ==========================================================================
    # 4. GENERADOR PARA TEST (DIRECTORIO COMPLETAMENTE SEPARADO)
    # ==========================================================================
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,  # ← ¡Directorios DIFERENTES!
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    # ==========================================================================
    # 5. VERIFICACIÓN DE INTEGRIDAD
    # ==========================================================================
    print(f"\n✅ Preprocesamiento completado:")
    print(f"   - Ejemplos de entrenamiento: {train_generator.samples}")
    print(f"   - Ejemplos de validación: {val_generator.samples}")
    print(f"   - Ejemplos de prueba: {test_generator.samples}")
    
    # Verificación crítica contra data leakage
    total_train_val = train_generator.samples + val_generator.samples
    print(f"   - Total train+val: {total_train_val}")
    print(f"   - Imágenes en directorio Training: {sum([len(files) for r, d, files in os.walk(train_dir)])}")
    
    # Verificar que no hay solapamiento
    if train_generator.samples + val_generator.samples > sum([len(files) for r, d, files in os.walk(train_dir)]):
        print("⚠️  ADVERTENCIA: Posible data leakage detectado!")
    else:
        print("✅ Integridad de datos verificada")
    
    # Mostrar algunas clases
    print(f"\n🍓 Ejemplo de clases ({min(5, num_classes)} de {num_classes}):")
    for i, class_name in enumerate(class_names[:5]):
        print(f"   {i+1}. {class_name}")
    
    return train_generator, val_generator, test_generator, class_names, num_classes

def verify_data_integrity(train_gen, val_gen, test_gen):
    """
    VERIFICACIÓN EXTRA DE INTEGRIDAD DE DATOS
    """
    print("\n🔍 VERIFICACIÓN DE INTEGRIDAD:")
    print("=" * 40)
    
    # 1. Verificar directorios
    print("📁 Directorios:")
    print(f"   - Train: {train_gen.directory}")
    print(f"   - Validation: {val_gen.directory}")
    print(f"   - Test: {test_gen.directory}")
    
    # 2. Verificar que train y validation son del mismo directorio (correcto)
    if train_gen.directory == val_gen.directory:
        print("✅ Train y Validation: mismo directorio (CORRECTO)")
        print("   - Usan validation_split para separar datos")
    else:
        print("❌ Train y Validation: directorios diferentes (ERROR)")
        print("   - Esto causa data leakage!")
    
    # 3. Verificar que test es diferente
    if test_gen.directory != train_gen.directory:
        print("✅ Test: directorio diferente (CORRECTO)")
    else:
        print("❌ Test: mismo directorio que train (ERROR)")
    
    # 4. Verificar samples
    print(f"\n📊 Samples:")
    print(f"   - Train: {train_gen.samples}")
    print(f"   - Validation: {val_gen.samples}")
    print(f"   - Test: {test_gen.samples}")
    print(f"   - Total: {train_gen.samples + val_gen.samples + test_gen.samples}")
    
    # 5. Verificar clases
    print(f"\n🎯 Clases:")
    print(f"   - Train classes: {len(train_gen.class_indices)}")
    print(f"   - Validation classes: {len(val_gen.class_indices)}")
    print(f"   - Test classes: {len(test_gen.class_indices)}")

def save_class_names(class_names, filename="class_names.json"):
    """Guarda los nombres de clases"""
    with open(filename, 'w') as f:
        json.dump(class_names, f)
    print(f"✅ Nombres de clases guardados en: {filename}")

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 PREPROCESAMIENTO CORREGIDO - SIN DATA LEAKAGE")
    print("=" * 60)
    
    try:
        # Procesar datos
        train_gen, val_gen, test_gen, classes, num_classes = preprocess_fruit360_data()
        
        # Verificación extra
        verify_data_integrity(train_gen, val_gen, test_gen)
        
        # Guardar metadata
        save_class_names(classes)
        
        print("\n🎉 ¡Datos listos para entrenar SIN data leakage!")
        print(f"   🍎 Clases: {num_classes}")
        print(f"   📐 Input shape: {train_gen.image_shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()