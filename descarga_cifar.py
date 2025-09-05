import os
import subprocess
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def download_fruit360_dataset(dataset_dir="../data_raw", verbose=True):
    """
    DESCARGAR DATASET FRUIT360 DESDE KAGGLE
    =======================================
    Función compatible con versiones antiguas de la API de Kaggle
    """
    
    if verbose:
        print("🔍 Iniciando descarga de Fruit360 dataset...")
        print(f"📂 Directorio destino: {os.path.abspath(dataset_dir)}")
    
    try:
        # ======================================================================
        # PASO 1: VERIFICAR INSTALACIÓN DE KAGGLE
        # ======================================================================
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            error_msg = "Librería 'kaggle' no instalada. Ejecuta: pip install kaggle"
            if verbose:
                print(f"❌ {error_msg}")
            return False, error_msg, None
        
        # ======================================================================
        # PASO 2: CONFIGURAR DIRECTORIO
        # ======================================================================
        os.makedirs(dataset_dir, exist_ok=True)
        
        if not os.access(dataset_dir, os.W_OK):
            error_msg = f"Sin permisos de escritura en: {dataset_dir}"
            if verbose:
                print(f"❌ {error_msg}")
            return False, error_msg, None
        
        # ======================================================================
        # PASO 3: AUTENTICACIÓN (MÉTODO COMPATIBLE)
        # ======================================================================
        if verbose:
            print("🔐 Autenticando con Kaggle API...")
        
        try:
            api = KaggleApi()
            api.authenticate()
            if verbose:
                print("✅ Autenticación exitosa")
        except Exception as auth_error:
            error_msg = f"Error de autenticación: {auth_error}"
            if verbose:
                print(f"❌ {error_msg}")
                print("💡 Verifica tu archivo ~/.kaggle/kaggle.json")
            return False, error_msg, None
        
        # ======================================================================
        # PASO 4: VERIFICAR DATASET (MÉTODO ALTERNATIVO)
        # ======================================================================
        if verbose:
            print("📊 Verificando disponibilidad del dataset...")
        
        # Método alternativo para verificar el dataset
        try:
            # Usar el método de listado que es más compatible
            datasets = api.datasets_list(search="fruits")
            fruit_dataset = None
            
            for dataset in datasets:
                if dataset.ref == "moltean/fruits":
                    fruit_dataset = dataset
                    break
            
            if fruit_dataset:
                if verbose:
                    print(f"✅ Dataset encontrado: '{fruit_dataset.title}'")
                    print(f"   👤 Autor: {fruit_dataset.ownerName}")
            else:
                error_msg = "Dataset 'moltean/fruits' no encontrado"
                if verbose:
                    print(f"❌ {error_msg}")
                return False, error_msg, None
                
        except Exception as info_error:
            # Si falla el listado, intentar igualmente la descarga
            if verbose:
                print(f"⚠️  No se pudo verificar metadata: {info_error}")
                print("   Intentando descarga directamente...")
        
        # ======================================================================
        # PASO 5: DESCARGAR USANDO MÉTODO MÁS COMPATIBLE
        # ======================================================================
        if verbose:
            print("⬇️  Descargando dataset...")
            print("   Esto puede tomar varios minutos (3.5 GB aprox)")
            print("   Dataset: https://www.kaggle.com/datasets/moltean/fruits")
        
        try:
            # Método más compatible para descargar
            api.dataset_download_files(
                "moltean/fruits",      # Identificador del dataset
                path=dataset_dir,      # Directorio de destino
                unzip=True,            # Descomprimir automáticamente
                quiet=not verbose      # Mostrar progreso si verbose=True
            )
            
            if verbose:
                print("✅ Descarga y descompresión completadas!")
                
        except Exception as download_error:
            # Intentar con método alternativo usando subprocess
            if verbose:
                print(f"⚠️  Error con API: {download_error}")
                print("🔄 Intentando con comando directo...")
            
            try:
                cmd = [
                    "kaggle", "datasets", "download", 
                    "moltean/fruits",
                    "--unzip",
                    "--path", dataset_dir
                ]
                
                if not verbose:
                    cmd.append("--quiet")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode != 0:
                    raise Exception(f"Comando falló: {result.stderr}")
                
                if verbose:
                    print("✅ Descarga completada via comando directo!")
                    
            except Exception as cmd_error:
                error_msg = f"Error en la descarga: {cmd_error}"
                if verbose:
                    print(f"❌ {error_msg}")
                return False, error_msg, None
        
        # ======================================================================
        # PASO 6: VERIFICAR INTEGRIDAD DE LA DESCARGA
        # ======================================================================
        if verbose:
            print("🔍 Verificando integridad de los datos...")
        
        # Buscar el directorio principal del dataset
        possible_paths = [
            dataset_dir,
            os.path.join(dataset_dir, "fruits-360"),
            os.path.join(dataset_dir, "fruits"),
            os.path.join(dataset_dir, "Fruits-360"),
            os.path.join(dataset_dir, "Fruits360"),
            os.path.join(dataset_dir, "fruits-360_dataset", "fruits-360"),
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                if verbose:
                    print(f"📁 Dataset encontrado en: {path}")
                break
        
        if not dataset_path:
            error_msg = "Descarga completada pero no se encontró la estructura esperada"
            if verbose:
                print(f"❌ {error_msg}")
                print("💡 Revisa el contenido del directorio:")
                for item in os.listdir(dataset_dir):
                    item_path = os.path.join(dataset_dir, item)
                    if os.path.isdir(item_path):
                        print(f"   📂 {item}/")
                    else:
                        print(f"   📄 {item}")
            return False, error_msg, None
        
        # Verificar que hay contenido
        try:
            content = os.listdir(dataset_path)
            if not content:
                error_msg = "Directorio descargado está vacío"
                if verbose:
                    print(f"❌ {error_msg}")
                return False, error_msg, None
                
            if verbose:
                print(f"✅ Contenido encontrado: {len(content)} items")
                # Mostrar primeros 5 elementos
                for item in content[:5]:
                    item_path = os.path.join(dataset_path, item)
                    if os.path.isdir(item_path):
                        print(f"   📂 {item}/")
                    else:
                        print(f"   📄 {item}")
                if len(content) > 5:
                    print(f"   ... y {len(content) - 5} más")
                
        except OSError as list_error:
            error_msg = f"Error accediendo al directorio: {list_error}"
            if verbose:
                print(f"❌ {error_msg}")
            return False, error_msg, None
        
        # ======================================================================
        # PASO 7: ÉXITO - RETORNAR RESULTADO
        # ======================================================================
        success_msg = f"Dataset descargado exitosamente en: {dataset_path}"
        if verbose:
            print(f"🎉 {success_msg}")
            print(f"📊 Tamaño total: {get_folder_size(dataset_path)}")
        
        return True, success_msg, dataset_path
        
    except Exception as global_error:
        error_msg = f"Error inesperado: {global_error}"
        if verbose:
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
        return False, error_msg, None


def get_folder_size(path):
    """
    FUNCIÓN AUXILIAR: Calcular tamaño de directorio
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                continue
    
    # Convertir a formato legible
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return f"{total_size:.2f} {unit}"
        total_size /= 1024.0
    return f"{total_size:.2f} TB"


if __name__ == "__main__":
    print("=" * 60)
    print("🌐 DESCARGADOR DE DATASET FRUIT360 (COMPATIBLE)")
    print("=" * 60)
    
    success, message, dataset_path = download_fruit360_dataset(
        dataset_dir="../data_raw",
        verbose=True
    )
    
    print("\n" + "=" * 60)
    if success:
        print("✅ RESULTADO: DESCARGA EXITOSA")
        print(f"📁 Datos en: {dataset_path}")
    else:
        print("❌ RESULTADO: DESCARGA FALLIDA")
        print(f"📋 Error: {message}")
    print("=" * 60)