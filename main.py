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

if __name__ == "__main__":
    main()