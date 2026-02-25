import os
import sys
import subprocess

# --- SETUP PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")

def run_script(script_path):
    """Executes a script as separate process"""
    print(f"\\n Execution: {os.path.basename(script_path)}...")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR during the execution of {script_path}")
        return False
    return True

def main():
    print("\t STARTING PIPELINE \t")
    
    # 1. Embeddings extraction
    super_wrapper = os.path.join(SRC_DIR, "super_wrapper.py")
    if not run_script(super_wrapper):
        sys.exit(1)
        
    # 2. Analysis and Plots
    total_embed = os.path.join(CURRENT_DIR, "total_embed.py")
    if os.path.exists(total_embed):
        if not run_script(total_embed):
            sys.exit(1)
    else:
        print(f"Analysis script {total_embed} not found.")

    print("\nALL OPERATIONS COMPLETED")

if __name__ == "__main__":
    main()