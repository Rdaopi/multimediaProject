import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "embedding_db.pkl")
OUTPUT_CSV = os.path.join(DATA_DIR, "statistical_results.csv")

def load_data():
    """Loads the embedding database."""
    if not os.path.exists(DB_PATH):
        print(f"Data path not found: {DB_PATH}")
        print("Assure you executed the data extraction step first.")
        return None
    with open(DB_PATH, 'rb') as f:
        return pickle.load(f)

def parse_metadata(df):
    """
    Extract spearker id and type
    """
    def get_type(filename):
        if 'real' in filename.lower(): return 'Real'
        if 'fake' in filename.lower(): return 'Fake'
        return 'Unknown'
        
    def get_speaker(filename):
        return filename.split('_')[0]

    df['speaker_id'] = df['filename'].apply(get_speaker)
    df['voice_type'] = df['filename'].apply(get_type)
    
    df = df[df['voice_type'] != 'Unknown']
    return df

def calculate_dispersion(embeddings):
    """
    Calculate the overall dispersion (variance) of a set of embeddings.
    """
    if len(embeddings) < 2:
        return 0.0
    
    matrix = np.stack(embeddings)
    centroid = np.mean(matrix, axis=0)
    
    distances = [euclidean(vec, centroid)**2 for vec in matrix]
    return np.mean(distances)

def run_statistical_analysis(df):
    print("Analysis statistic of latent space")
    print("="*60)

    models = df['model'].unique()
    results = []

    for model in models:
        model_df = df[df['model'] == model]
        
        real_df = model_df[model_df['voice_type'] == 'Real']
        fake_df = model_df[model_df['voice_type'] == 'Fake']
        
        if real_df.empty or fake_df.empty:
            print(f"Model {model}: Missing 'real' or 'fake' files for comparison, skip.")
            continue

        # 1. Variance/Dispersion Analysis
        real_dispersion = calculate_dispersion(real_df['embedding'].values)
        fake_dispersion = calculate_dispersion(fake_df['embedding'].values)
        entropy_ratio = fake_dispersion / real_dispersion if real_dispersion > 0 else 0

        # 2. Cosine Similarity
        similarities = []
        speakers = set(real_df['speaker_id']).intersection(set(fake_df['speaker_id']))
        
        for speaker in speakers:
            emb_real = real_df[real_df['speaker_id'] == speaker]['embedding'].values[0]
            emb_fake = fake_df[fake_df['speaker_id'] == speaker]['embedding'].values[0]
            
            sim = 1 - cosine(emb_real, emb_fake)
            similarities.append(sim)
            
        avg_similarity = np.mean(similarities) if similarities else 0.0

        results.append({
            'Model': model,
            'Real_Variance': round(real_dispersion, 4),
            'Fake_Variance': round(fake_dispersion, 4),
            'Entropy_Ratio (Fake/Real)': round(entropy_ratio, 4),
            'Avg_Real_Fake_Similarity': round(avg_similarity, 4),
            'Speakers_Analyzed': len(speakers)
        })

    if not results:
        print("No result available to display. Check the audio file names.")
        return

    # Creazione del Report
    results_df = pd.DataFrame(results)
    print("\n RESULTS:\n")
    print(results_df.to_string(index=False))
    
    print("\n" + "-"*60)
    print("CONCLUSIONS:")
    for res in results:
        model = res['Model']
        ratio = res['Entropy_Ratio (Fake/Real)']
        sim = res['Avg_Real_Fake_Similarity']
        
        ent_msg = "Reduced Entropy spotted" if ratio < 0.9 else ("No" if ratio > 1.1 else "Undefined")
            
        print(f"\n🔹 {model.upper()}:")
        print(f"   - The fakes lost details compares to real ones? {ent_msg} (Ratio: {ratio})")
        print(f"   - Cloning Fidelty: {sim} su 1.0")
    
    print("-" * 60)
    
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results exported to: {OUTPUT_CSV}")

def main():
    df = load_data()
    if df is not None:
        df = parse_metadata(df)
        run_statistical_analysis(df)

if __name__ == "__main__":
    main()