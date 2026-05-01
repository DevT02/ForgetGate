import os
import shutil
import glob

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting Phase 2 Refactor...")

    # ---------------------------
    # IDEA C: Analysis Segregation
    # ---------------------------
    res_dir = os.path.join(root, 'results', 'analysis')
    if os.path.exists(res_dir):
        metrics_dir = os.path.join(res_dir, 'metrics')
        reports_dir = os.path.join(res_dir, 'reports')
        figures_dir = os.path.join(res_dir, 'figures')
        
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # Move JSONs
        moved_jsons = 0
        for f in glob.glob(os.path.join(res_dir, '*.json')):
            shutil.move(f, os.path.join(metrics_dir, os.path.basename(f)))
            moved_jsons += 1
            
        # Move MDs and TXTs
        moved_docs = 0
        for f in glob.glob(os.path.join(res_dir, '*.md')):
            shutil.move(f, os.path.join(reports_dir, os.path.basename(f)))
            moved_docs += 1
        for f in glob.glob(os.path.join(res_dir, '*.txt')):
            shutil.move(f, os.path.join(reports_dir, os.path.basename(f)))
            moved_docs += 1
            
        # Move TeX and PNGs
        moved_figs = 0
        for f in glob.glob(os.path.join(res_dir, '*.tex')):
            shutil.move(f, os.path.join(figures_dir, os.path.basename(f)))
            moved_figs += 1
        for f in glob.glob(os.path.join(res_dir, '*.png')):
            shutil.move(f, os.path.join(figures_dir, os.path.basename(f)))
            moved_figs += 1
            
        print(f"[{moved_jsons} JSONs -> metrics/] [{moved_docs} DOCs -> reports/] [{moved_figs} FIGs -> figures/]")

    # ---------------------------
    # IDEA A: Script Modularization
    # ---------------------------
    scripts_dir = os.path.join(root, 'scripts')
    
    categories = {
        "train": ["train", "unlearn", "vpt", "benign", "run_controls.py"],
        "attacks": ["attack", "patch", "probe", "evaluate"],
        "audits": ["audit", "probe", "evaluate", "leakage"],
        "analysis": ["analyze", "summarize", "build", "visualize"],
        "pipelines": ["run_"]
    }
    
    for cat in categories.keys():
        os.makedirs(os.path.join(scripts_dir, cat), exist_ok=True)
        
    moved_scripts = 0
    import_patches = 0
    
    for f in os.listdir(scripts_dir):
        fp = os.path.join(scripts_dir, f)
        if not os.path.isfile(fp):
            continue
        if f == "refactor_project.py" or f == "run_awp_audits.ps1":
            continue
            
        name = f.lower()
        matched_cat = None
        
        # Explicit priority: PS scripts go to pipelines
        if name.endswith('.ps1'):
            matched_cat = "pipelines"
        else:
            # Check others
            for cat, keywords in categories.items():
                if any(kw in name for kw in keywords):
                    matched_cat = cat
                    break
                    
        if matched_cat:
            dest = os.path.join(scripts_dir, matched_cat, f)
            shutil.move(fp, dest)
            moved_scripts += 1
            
            # If python, fix relative imports safely
            if f.endswith('.py'):
                with open(dest, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                old_str_1 = "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
                new_str_1 = "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))"
                
                if old_str_1 in content:
                    content = content.replace(old_str_1, new_str_1)
                    with open(dest, 'w', encoding='utf-8') as file:
                        file.write(content)
                    import_patches += 1
                    
    print(f"Modularized {moved_scripts} scripts. Patched imports in {import_patches} scripts.")

    print("Refactor complete! The repository is now logically structured.")

if __name__ == "__main__":
    main()
