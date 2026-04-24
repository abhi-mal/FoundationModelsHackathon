import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as roc_auc, roc_auc_score

# ── 1. Configuration ──────────────────────────────────────────────────────────
files = {
    "Pretrained": "./outputs/outputs_pretrain_s_top_0.npz",
    "Fine-tuned": "./outputs/outputs_my_finetune_top_0.npz",
    "Frozen-Backbone": "./outputs/outputs_my_finetune_frozen_top_0.npz"
}

# ── 2. Top Indices Printing ───────────────────────────────────────────────────
loaded_data = {}

for name, path in files.items():
    print(f"\n" + "="*60)
    print(f"ANALYZING: {name}")
    print(f"\n" + "="*60)
    
    try:
        data = np.load(path)
        labels = data["pid"]
        preds = data["prediction"]
        loaded_data[name] = {"labels": labels, "preds": preds}
        
        # Signal Top 5
        signal_means = preds[labels == 1].mean(axis=0)
        top_5_indices = np.argsort(signal_means)[-5:][::-1]
        print("Top 5 most active model indices for Signal (Top Quark) samples:")
        for idx in top_5_indices:
            print(f"  Index {idx:3d}: Mean Score = {signal_means[idx]:.4f}")
            
        # Background Top 5
        bkg_means = preds[labels == 0].mean(axis=0)
        qcd_5_indices = np.argsort(bkg_means)[-5:][::-1]
        print("\nTop 5 most active model indices for Bkg (QCD) samples:")
        for idx in qcd_5_indices:
            print(f"  Index {idx:3d}: Mean Score = {bkg_means[idx]:.4f}")
            
    except FileNotFoundError:
        print(f"ERROR: Could not find {path}")
    except Exception as e:
        print(f"ERROR: {e}")

# ── 3. Plotting ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = {"Pretrained": "black", "Fine-tuned": "crimson", "Frozen-Backbone": "steelblue"}
linestyles = {"Pretrained": "--", "Fine-tuned": "-", "Frozen-Backbone": "-."}

# ── Left Plot: ROC Comparison (Top vs QCD) ──────────────────────────────────
# Comparing the best signal discriminator across the 3 models
ax1.set_title('ROC Comparison (Top vs QCD)', fontsize=12, fontweight='bold')
print("\nPlotting Left Panel: Top vs QCD Comparison...")

for name, data in loaded_data.items():
    y_true = data["labels"]
    num_classes = data["preds"].shape[1]
    
    # Best signal node: Index 10 for 200-class, Index 1 for binary
    idx = 10 if num_classes > 2 else 1
    y_score = data["preds"][:, idx]
    
    # Calculate AUC
    auc_val = roc_auc_score(y_true, y_score)
    if auc_val < 0.5:
        auc_val = 1 - auc_val
        y_score = 1 - y_score
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax1.plot(fpr, tpr, color=colors[name], ls=linestyles[name], lw=2,
             label=f"{name} (AUC={auc_val:.4f})")
    print(f"  {name:15}: idx {idx:2d} | AUC = {auc_val:.4f}")

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(alpha=0.3)

# ── Right Plot: Comparing AUC for QCD node vs Top node ───────────────────────
# Specifically compares nodes 2 and 10 within the Pretrained model
ax2.set_title('Comparing AUC for QCD node vs Top node for discrimination', 
              fontsize=11, fontweight='bold')

if "Pretrained" in loaded_data:
    p_data = loaded_data["Pretrained"]
    y_true = p_data["labels"]
    
    nodes = [
        {"idx": 2,  "label": "QCD node as discriminator", "color": "orange"},
        {"idx": 10, "label": "Top node as discriminator", "color": "green"}
    ]
    
    for node in nodes:
        idx = node["idx"]
        y_score = p_data["preds"][:, idx]
        
        auc_val = roc_auc_score(y_true, y_score)
        if auc_val < 0.5:
            auc_val = 1 - auc_val
            y_score = 1 - y_score
            
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax2.plot(fpr, tpr, color=node["color"], lw=2,
                 label=f"{node['label']} (AUC={auc_val:.4f})")
        print(f"  Pretrained node {idx:2d}: {node['label']:25} | AUC = {auc_val:.4f}")

ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_roc.png', dpi=150)