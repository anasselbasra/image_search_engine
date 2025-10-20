############################################################################
# Fichier: computer_vision/vision_models.py
# Auteur: Anass El Basraoui
# Date: 2025-10-20
# Description: Fonctions et modèles de vision par ordinateur avec gestion des téléchargements d'images.
############################################################################


from urllib.parse import urlparse # Pour extraire le nom de fichier à partir d’une URL.
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, time, os
from pathlib import Path  # Gère proprement les chemins de fichiers (OS-independent)
import pandas as pd
import numpy as np 
import hdbscan
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import torch # Bibliothèque principale pour exécuter les modèles sur GPU (PyTorch)
from PIL import Image # Librairie pour ouvrir, convertir et manipuler des images locales
from io import BytesIO 
import logging # Pour contrôler le niveau d'affichage des logs (supprimer les warnings inutiles)
from transformers import (
    AutoImageProcessor,  # Gère automatiquement les pré-traitements (resize, normalisation...) selon le modèle
    AutoModel,           # Permet de charger les modèles de vision (ex : DINOv2, SigLIP)
    AutoProcessor,
    CLIPProcessor,       # Processor spécifique pour les modèles CLIP (image + texte)
    CLIPModel           # Modèle CLIP complet, utilisé ici uniquement pour la partie image
)
logging.getLogger("transformer").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)


device = "cuda" if torch.cuda.is_available() else "cpu"

def show_image(path):
    from  matplotlib.pyplot import  imread, imshow, axis, show
    # Chemin vers votre image
    chemin_image = path
    # Lire l'image
    img = imread(chemin_image)

    # Afficher l'image
    imshow(img)
    axis('off')  # Masquer les axes pour une vue propre
    show()





def download_with_retry(url, path, folder_name, retries=2, delay=1, timeout=8):
    """
    Télécharge une image avec plusieurs tentatives.
    Retourne : (url, local_path ou None, message d’erreur ou None)
    """
    save_dir = os.path.join(path, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(urlparse(url).path) or f"img_{int(time.time() * 1000)}.jpg"
    local_path = os.path.join(save_dir, filename)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()

            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(local_path)
            return (url, local_path, None)

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(delay * attempt)
            else:
                return (url, None, err)




# Exécution parallèle et collecte des résultats
def parallel_download(urls, path, folder_name, return_failed_csv=False, csv_name="failed_urls.csv", timeout=10):
    """
    Télécharge une liste d'images en parallèle.
    
    Args:
        urls (list[str]): Liste des URLs à télécharger.
        path (str | Path): Répertoire de base où le dossier sera créé.
        folder_name (str): Nom du dossier à créer à l'intérieur de base_path.
        return_failed_csv (bool): Si True, enregistre un CSV des téléchargements échoués.
        csv_name (str): Nom du fichier CSV à créer si return_failed_csv=True.

    Returns:
        results (list[tuple]): (url, path, error) pour chaque image téléchargée.
        failed (list[tuple]): (url, error) pour chaque échec.
    """
    results = []
    failed = []
    max_workers = os.cpu_count() or 8
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_with_retry, url, path, folder_name, timeout=timeout): url for url in urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Téléchargement"):
            url = futures[fut]
            try:
                url, local_path, error = fut.result()
                results.append((url, local_path, error))
                if error:
                    failed.append((url, error))
            except Exception as e:
                failed.append((url, f"FutureError: {e}"))
    if return_failed_csv and failed:
        failed_csv = os.path.join(path, csv_name)
        with open(failed_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "error"])
            writer.writerows(failed)
        print(f"Fichier des échecs enregistré: {failed_csv}")

    print(f"Total URLs: {len(urls)}")
    print(f"Succès: {len([r for r in results if r[1]])}")
    print(f"Échecs: {len(failed)}" + (f"(voir {failed_csv})" if return_failed_csv else ""))
    return results, failed



##################################################################################################
## Application d'embedding avec un modèle de vision
##################################################################################################


################################################ DINO ###############################################
def encode_with_dino(image_dir, processor, model, device="cuda", batch_size=16):
    """
    Encode toutes les images d'un dossier avec un modèle visuel (ex : DINOv2).
    Retourne un DataFrame (filename, embedding).
    """
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]

    embeddings, names = [], []

    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding batches"):
        batch_paths = paths[i:i+batch_size]

        # Chargement parallèle des images
        with ThreadPoolExecutor(max_workers=8) as ex:
            images = list(ex.map(lambda p: Image.open(p).convert("RGB"), batch_paths))

        # Prétraitement des images + conversion en float16
        inputs = processor(images=images, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"]  # garantit compatibilité FP16

        # Encodage
        with torch.no_grad():
            out = model(**inputs)
            if hasattr(out, "pooler_output"):
                feats = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state.mean(dim=1)
            else:
                raise ValueError("Modèle non compatible : pas de pooler_output ni last_hidden_state")

            feats = torch.nn.functional.normalize(feats, dim=-1)

        embeddings.extend(feats.cpu().float().numpy())
        names.extend([p.name for p in batch_paths])

    return pd.DataFrame({"filename": names, "embedding": embeddings})

def upload_dino(model_name="facebook/dinov2-large", device="cuda"):
    """
    Charge automatiquement un modèle visuel (DINOv2, v3, etc.)
    et renvoie (processor, model) sur le bon device.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()

    print(f"Modèle: {model_name} chargé sur {device}")

    return processor, model

########################################################################################################
################################################ CLIP ###############################################

## Econdage d'une image avec un modèle de vision
def encode_with_clip(image_dir=None, texts=None, processor=None, model=None, device=device, batch_size=16):
    """
    Encode des images et/ou des textes avec un modèle de la famille CLIP/OpenCLIP.
    'image_dir' peut être un chemin vers un dossier, un fichier, ou une liste de fichiers.
    Retourne un dictionnaire contenant des DataFrames pour les embeddings.
    """
    if image_dir is None and texts is None:
        raise ValueError("Vous devez fournir soit 'image_dir', soit 'texts'.")

    # On initialise toujours la structure du résultat
    results = {
        "image_embeddings": pd.DataFrame(),
        "text_embeddings": pd.DataFrame()
    }

    if image_dir:
        print("Préparation des chemins d'images...")
        paths = []
        exts = {".jpg", ".jpeg", ".png", ".webp"}

        # --- NOUVELLE LOGIQUE INTELLIGENTE ---

        if isinstance(image_dir, list):
            # Cas 1 : On a directement une liste de chemins
            paths = [Path(p) for p in image_dir if Path(p).suffix.lower() in exts]
        else:
            # Cas 2 : On a un seul chemin (dossier ou fichier)
            p = Path(image_dir)
            if p.is_dir():
                paths = [img_p for img_p in p.iterdir() if img_p.suffix.lower() in exts]
            elif p.is_file():
                if p.suffix.lower() in exts:
                    paths = [p]
        
        if not paths:
            print("Avertissement : Aucune image valide trouvée à traiter.")
        else:
            img_embeddings, img_names = [], []
            for i in tqdm(range(0, len(paths), batch_size), desc="Image batches"):
                batch_paths = paths[i:i+batch_size]
                with ThreadPoolExecutor(max_workers=18) as ex:
                    images = list(ex.map(lambda p: Image.open(p).convert("RGB"), batch_paths))

                inputs = processor(images=images, return_tensors="pt").to(device)


                with torch.no_grad():
                    feats = model.get_image_features(**inputs)
                    feats = torch.nn.functional.normalize(feats, dim=-1)

                img_embeddings.extend(feats.cpu().float().numpy())
                img_names.extend([p.name for p in batch_paths])
            results["image_embeddings"] = pd.DataFrame({"filename": img_names, "embedding": img_embeddings})


    # --- PARTIE 2 : ENCODAGE DU TEXTE (si demandé) ---
    if texts:
        print("Encodage du texte...")
        # Assurer que 'texts' est une liste
        if isinstance(texts, str):
            texts = [texts]

        # Pas besoin de batching pour le texte si la liste est petite
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)

        text_embeddings = feats.cpu().float().numpy()
        results["text_embeddings"] = pd.DataFrame({"text": texts, "embedding": list(text_embeddings)})

        # ajoute vstack ou autre

    return results




def upload_clip(model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", device="cuda"):
    """
    Charge automatiquement un modèle visuel (CLIP)
    et renvoie (processor, model) sur le bon device.
    """
    # CLIP / OpenCLIP
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    model = CLIPModel.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()

    print(f" Modèle {model_name} chargé sur {device}")
    return processor, model

### "openai/clip-vit-large-patch14" <- 768 <- 3
### "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" <- 1024 <- 1 (le meilleur) 
### "openai/clip-vit-large-patch14-336"<- 768 <-  2

########################################################################################################
########################################################################################################

######   Outil 1: SEARCH ENGINE

########################################################################################################
########################################################################################################

def search_engine(image_dir, query, df, model, processor, k=10, device="cuda"):
    """
    Cherche les images les plus similaires à une requête dans un index pré-calculé.
    La requête peut être un texte, un chemin vers une image, ou un vecteur d'embedding.
    """
    
    # --- 1. Obtenir l'embedding de la requête ---
    query_embedding = None
    
    # Cas 1 : La requête est déjà un embedding
    if isinstance(query, np.ndarray):
        query_embedding = query
        
    # Cas 2 : La requête est un texte (str)
    elif isinstance(query, str):
        # On vérifie si c'est un chemin d'image valide
        is_image_path = os.path.exists(query) and query.split('.')[-1].lower() in ["jpg", "jpeg", "png", "webp"]
        
        if is_image_path:
            print("Recherche par image...")
            results = encode_with_clip(processor=processor, model=model, image_dir=query)
            query_embedding = results["image_embeddings"]["embedding"].iloc[0]
        else:
            print("Recherche par texte...")
            results = encode_with_clip(processor=processor, model=model, texts=query)
            query_embedding = results["text_embeddings"]["embedding"].iloc[0]
            
    if query_embedding is None:
        raise ValueError("Format de la requête non reconnu.")

    # --- 2. Calculer les similarités ---    
    image_matrix = np.vstack(df["embedding"].values)
    
    # Le produit scalaire mesure la similarité cosinus pour des vecteurs normalisés
    similarities = image_matrix @ query_embedding.T
    
    # --- 3. Trouver et afficher les top k résultats ---
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    top_files = df["filename"].iloc[top_k_indices].tolist()
    top_scores = similarities[top_k_indices]

    # Affichage des images
    cols = 5
    rows = (k + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 5 * rows))
    
    for i, (fname, score) in enumerate(zip(top_files, top_scores)):
        img_path = Path(image_dir) / fname
        img = Image.open(img_path).convert("RGB")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{fname}\n Similarité: {score:.3f}", fontsize=12)

    plt.tight_layout()
    plt.show()