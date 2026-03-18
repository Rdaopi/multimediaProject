import os
import sys

def main():
    print("--- Voice Identity Analysis Tool ---")
    
    # ========== STEP 1: SETUP ==========
    print("\n🔧 Esecuzione setup iniziale...\n")
    
    import setup_project
    if not setup_project.main():
        print("\n❌ Setup fallito. Impossibile avviare la pipeline.")
        sys.exit(1)
    
    # ========== STEP 2: ESECUZIONE PIPELINE ==========
    print("\n🎬 Avvio pipeline principale...\n")

    # Avviamo il total_wrapper
    import total_wrapper
    total_wrapper.main()

    # ========== STEP 3: ANALISI STATISTICA ==========
    print("\n📈 Avvio Analisi Statistica e calcolo Entropia...\n")
    
    try:
        import statistical_analysis
        statistical_analysis.main()
    except ImportError:
        print("\n⚠️ Impossibile trovare 'statistical_analysis.py'. Assicurati che sia nella stessa cartella di main.py.")
    except Exception as e:
        print(f"\n❌ Errore durante l'analisi statistica: {e}")

    print("\n✅ PIPELINE COMPLETATA CON SUCCESSO! 🎉")

if __name__ == "__main__":
    main()