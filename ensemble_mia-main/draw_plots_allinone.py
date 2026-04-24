import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import numpy as np
import argparse

def plot_combined_attack_rocs(
    methods_root,
    synth_size,
    attack_size,
    true_labels,
    cf_methods=None,
    out_dir=None,
):
    """Plot combined ROC curves per attack across multiple methods.

    methods_root: directory that contains one subfolder per method, each with
    synth_size_<size>/attack_size_<size>/all_scores.csv
    """
    methods_root = Path(methods_root)

    if cf_methods is None:
        cf_methods = [p.name for p in methods_root.iterdir() if p.is_dir()]

    if out_dir is None:
        out_dir = methods_root / "combined_rocs" / f"synth_size_{synth_size}" / f"attack_size_{attack_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}  # attack -> list of (method, fpr, tpr, auc)

    for method in cf_methods:
        csv_path = methods_root / method / f"synth_size_{synth_size}" / f"attack_size_{attack_size}" / "all_scores.csv"
        if not csv_path.exists():
            print(f"Missing: {csv_path}")
            continue

        scores_df = pd.read_csv(csv_path)

        # for col in scores_df.columns:
        #     scores = np.array(pd.to_numeric(scores_df[col], errors='coerce'))


        if len(true_labels) != len(scores_df):
            print(f"Label length mismatch for {csv_path}: labels={len(true_labels)}, rows={len(scores_df)}")
            continue

        for attack in scores_df.columns:  # attack name from column header
            fpr, tpr, _ = roc_curve(true_labels, scores_df[attack].values)
            roc_auc = auc(fpr, tpr)
            curves.setdefault(attack, []).append((method, fpr, tpr, roc_auc))

    # for attack, entries in curves.items():
    #     plt.figure()
    #     for method, fpr, tpr, roc_auc in entries:
    #         plt.plot(fpr, tpr, lw=2, label=f"{method} (AUC={roc_auc:.3f})")
    #     plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title(f"{methods_root.name} • {attack}")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(out_dir / f"{attack}_combined_roc.png")
    #     plt.close()

    for attack, entries in curves.items():
        fig, axs = plt.subplots(1, 1, figsize=(10, 4))  # unified shape
        ax = axs if not isinstance(axs, np.ndarray) else axs[0]

        for method, fpr, tpr, roc_auc in entries:
            color_map = {"dice_kdtree": "green", "dice_gradient": "blue", "NICE": "orange", "scfe": "red"}
            color = color_map.get(method, None)
            ax.loglog(fpr, tpr, lw=2, color=color, label=f"{method} – auc:{roc_auc:.2f}")
            # ax.loglog(fpr, tpr, lw=1, label=f"{method} – auc:{roc_auc:.2f}")
        plt.rcParams["font.family"] = "serif"    
        ax.plot([0, 1], [0, 1], linestyle="dotted", color="black", label="Random Baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim([0.001, 1.01])
        ax.set_ylim([0.001, 1.01])
        # ax.set_title(f" {attack}")
        ax.legend(framealpha=0.25,fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"{attack}_combined_roc.png", dpi=200)
        plt.close(fig)
        





    print(f"Saved combined ROC plots to: {out_dir}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset_name', type=str, default='heloc', help='hospital,adult,informs,synth_adult,synth_informs,synth_hospital,compas,default_credit')
    parser.add_argument('--rseed', type=int, default=3, help='random seed: choose between 0 - 5')
    parser.add_argument('--attack_size', type=int, default=500, help='random seed: choose between 0 - 5')
    # parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 5')
    # seed = 0
    # attack_size = 300
    rseed = parser.parse_args().rseed
    attack_size = parser.parse_args().attack_size
    dataset_name = parser.parse_args().dataset_name

    true_labels = np.concatenate([np.ones(attack_size),np.zeros(attack_size)])
    
    
    plot_combined_attack_rocs(
        methods_root="outputs/{}/{}/".format(dataset_name,rseed),  # directory that contains method folders (e.g., NICE, dice_kdtree, ...)
        synth_size=10000,
        attack_size=attack_size,
        true_labels=true_labels,
        # cf_methods=["NICE", "dice_kdtree", "dice_gradient"],  # optional; omit to auto-discover
    )







