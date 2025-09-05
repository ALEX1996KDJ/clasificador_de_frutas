import os

def check_dataset_structure(dataset_dir="../data_raw"):
    """
    Verifica la estructura real del dataset Fruits360 (versión actualizada)
    """
    # ===== RUTAS POSIBLES ACTUALIZADAS =====
    possible_paths = [
        dataset_dir,
        os.path.join(dataset_dir, "fruits-360"),
        os.path.join(dataset_dir, "fruits"),
        os.path.join(dataset_dir, "Fruits-360"),
        os.path.join(dataset_dir, "Fruits360"),
        # NUEVAS RUTAS BASADAS EN LA DESCARGA REAL
        os.path.join(dataset_dir, "fruits-360_100x100"),
        os.path.join(dataset_dir, "fruits-360_original-size"),
        os.path.join(dataset_dir, "fruits-360_multi"),
        os.path.join(dataset_dir, "fruits-360_3-body-problem"),
    ]
    
    train_path = None
    test_path = None
    
    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue
            
        print(f"🔍 Examinando: {base_path}")
        
        # ===== BUSQUEDA EN PRIMER NIVEL =====
        try:
            items = os.listdir(base_path)
            dirs = [d for d in items if os.path.isdir(os.path.join(base_path, d))]
            
            # Verificar estructura en primer nivel
            if "Training" in dirs and "Test" in dirs:
                train_path = os.path.join(base_path, "Training")
                test_path = os.path.join(base_path, "Test")
                print(f"   ✅ Encontrado en primer nivel")
                break
                
            if "train" in dirs and "test" in dirs:
                train_path = os.path.join(base_path, "train")
                test_path = os.path.join(base_path, "test")
                print(f"   ✅ Encontrado en primer nivel (minúsculas)")
                break
                
        except PermissionError:
            print(f"   ⚠️  Sin permisos para listar: {base_path}")
            continue
        
        # ===== BUSQUEDA RECURSIVA =====
        print(f"   🔎 Buscando recursivamente...")
        found_in_recursion = False
        
        for root, dirs, files in os.walk(base_path):
            if "Training" in dirs and "Test" in dirs:
                train_path = os.path.join(root, "Training")
                test_path = os.path.join(root, "Test")
                print(f"   ✅ Encontrado en: {root}")
                found_in_recursion = True
                break
                
            if "train" in dirs and "test" in dirs:
                train_path = os.path.join(root, "train")
                test_path = os.path.join(root, "test")
                print(f"   ✅ Encontrado en: {root} (minúsculas)")
                found_in_recursion = True
                break
        
        # Si encontró en la búsqueda recursiva, salir del bucle principal
        if found_in_recursion:
            break
    
    # ===== MANEJAR RESULTADO =====
    if train_path and test_path:
        # Verificar que las rutas existen realmente
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print("❌ Rutas encontradas pero no existen físicamente")
            return None, None
        
        print(f"✅ Estructura encontrada:")
        print(f"   - Entrenamiento: {train_path}")
        print(f"   - Prueba: {test_path}")
        
        # Contar clases
        try:
            train_classes = len([d for d in os.listdir(train_path) 
                               if os.path.isdir(os.path.join(train_path, d))])
            test_classes = len([d for d in os.listdir(test_path) 
                              if os.path.isdir(os.path.join(test_path, d))])
            
            # Contar imágenes (muestra progreso para datasets grandes)
            print("   📊 Contando imágenes...")
            train_samples = 0
            for root, dirs, files in os.walk(train_path):
                train_samples += len(files)
            
            test_samples = 0
            for root, dirs, files in os.walk(test_path):
                test_samples += len(files)
            
            print(f"📊 Estadísticas del dataset:")
            print(f"   - Clases en entrenamiento: {train_classes}")
            print(f"   - Clases en prueba: {test_classes}")
            print(f"   - Imágenes de entrenamiento: {train_samples}")
            print(f"   - Imágenes de prueba: {test_samples}")
            
            return train_path, test_path
            
        except Exception as e:
            print(f"❌ Error al contar archivos: {e}")
            return None, None
        
    else:
        print("❌ No se pudo encontrar la estructura esperada (Training/Test)")
        print("📁 Directorios examinados:")
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                print(f"   - {base_path}:")
                try:
                    items = os.listdir(base_path)
                    for item in items[:10]:  # Mostrar solo primeros 10
                        item_path = os.path.join(base_path, item)
                        if os.path.isdir(item_path):
                            print(f"     📂 {item}/")
                        else:
                            print(f"     📄 {item}")
                    if len(items) > 10:
                        print(f"     ... y {len(items) - 10} más")
                except PermissionError:
                    print(f"     ⚠️  Sin permisos para listar")
        
        return None, None

# ===== CÓDIGO PARA EJECUTAR LA FUNCIÓN =====
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 VALIDADOR DE ESTRUCTURA DE DATASET FRUIT360")
    print("=" * 60)
    
    # Llamar a la función principal
    train_path, test_path = check_dataset_structure()
    
    print("\n" + "=" * 60)
    if train_path and test_path:
        print("🎯 VALIDACIÓN EXITOSA")
        print(f"📍 Ruta de entrenamiento: {train_path}")
        print(f"📍 Ruta de prueba: {test_path}")
    else:
        print("💥 VALIDACIÓN FALLIDA")
        print("No se encontró la estructura Training/Test")
    print("=" * 60)