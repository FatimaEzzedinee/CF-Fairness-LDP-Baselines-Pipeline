import os
import pandas as pd

import matplotlib.pyplot as plt





def plot_comparison_plot(df, metric,x_label , plot_title, plot_dir):
    """
    Plots line plots comparing specified metrics for each row in the dataframe.

    Args:
        final_df (pd.DataFrame): DataFrame containing grouped results.
        metrics (list): List of metric column names to plot.
        plot_title (str): Title for the plot.
        save_path (str): Path to save the plot image.
    """

    plt.figure(figsize=(8, 5))
    for attack in df['attack_method'].unique():
        subset = df[df['attack_method'] == attack]
        plt.plot(subset[x_label], subset[metric], marker='o', label=attack)
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.title(plot_title) # 'Attack Performance vs Attack Set Size')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plot_path = '{}/{}_vs_{}.png'.format(plot_dir, metric, x_label)
    plt.savefig(plot_path)
    plt.close()
    # plt.savefig('{}/attack_performance_vs_attack_size.png'.format(plot_dir))



# Path to the mia_results file (assumed CSV format)
# /Users/maryam/PhD/My projects/Ensemble MIA joshward/Ensemble-MIA/results/mia_results.csv
# mia_results_path = '/Users/maryam/PhD/My projects/Ensemble MIA joshward/Ensemble-MIA/results/mia_results.csv'  # Update with actual path if needed
mia_results_path = './results/mia_results.csv'
# Read the results file
df = pd.read_csv(mia_results_path)

# Group by attack type and compute average over seeds
# Assumes columns: 'attack', 'seed', 'value' (update as needed)

average_df = df.groupby(['attack_method', 'dataset', 'synth_size', 'attack_set_size', 'method']).mean(numeric_only=True).reset_index()
# average_df = df.groupby(['attack_method', 'dataset', 'synth_size', 'attack_set_size', 'method']).mean(numeric_only=True).reset_index()
average_df.drop(columns=['seed'], inplace=True, errors='ignore')
average_df.to_csv('mia_results_averaged.csv', index=False)

# Print or save the averaged results
print(average_df)



plot_dir = 'evaluation_plots'

# Calculate the ratio of attack_set_size to synth_size
average_df['attack_to_synth_ratio'] = average_df['attack_set_size'] / average_df['synth_size']

### categorize attacks and plot 
# cat 1: dcr, dcr_diff
# Draw separate plots for each attack and each unique attack_to_synth_ratio value,
# for each metric (auc_roc, tpr_at_fpr_0.001, tpr_at_fpr_0.1) and x_label (synth_size, attack_set_size).

attack_names = ['dcr', 'dpi', 'logan', 'classifier', 'domias', 'gen_lra', 'mc', 'majority_vote']
metrics = ['auc_roc', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.1']
x_labels = ['synth_size', 'attack_set_size']
ratio_label = 'attack_to_synth_ratio'
# for dataset in average_df['dataset'].unique():
#     dataset_dir = os.path.join(plot_dir, dataset)
#     os.makedirs(dataset_dir, exist_ok=True)
#     dataset_df = average_df[average_df['dataset'] == dataset]
#     for method in dataset_df['method'].unique():
#         method_dir = os.path.join(dataset_dir, method)
#         os.makedirs(method_dir, exist_ok=True)
#         method_df = dataset_df[dataset_df['method'] == method]
#         for attack in attack_names:
#             temp_attack = method_df[method_df['attack_method'].str.contains(attack)]
#             plot_path = os.path.join(method_dir, attack)
#             # os.makedirs(plot_path, exist_ok=True)
#             for metric in metrics:
#                 for x_label in x_labels:
#                     # Merge ratios that are close to each other (e.g., within 0.01)
#                     ratio_bins = temp_attack[ratio_label].round(1).unique()
#                     for ratio_bin in sorted(ratio_bins):
#                         subset = temp_attack[temp_attack[ratio_label].round(1) == ratio_bin]
#                         if subset.empty:
#                             continue
#                         title = f"{dataset} - {method} - {attack} - {metric} vs {x_label} (ratio~{ratio_bin:.2f})"
#                         save_dir = os.path.join(plot_path, f"{metric}_vs_{x_label}_ratio_{ratio_bin:.2f}")
#                         os.makedirs(save_dir, exist_ok=True)
#                         plot_comparison_plot(subset, metric, x_label, title, save_dir)
    


methods = {"dice_gradient","dice_kdtree","NICE","scfe"}
    # Additional plots as requested
    # For each attack, show effect of changing attack_set_size for fixed synth_size, and effect of changing synth_size for fixed attack ratio
for attack in attack_names:
    attack_df = average_df[average_df['attack_method'].str.contains(attack)]
    if attack_df.empty:
        continue
    

    # 1. For each fixed synth_size, plot metric vs attack_set_size, colored by attack_method (do not mix datasets)
    for dataset in attack_df['dataset'].unique():
        attack_plot_dir = os.path.join(plot_dir,dataset ,attack)
        os.makedirs(attack_plot_dir, exist_ok=True)
        for method in methods:
        # for method in attack_df['method'].unique():
            method_dir = os.path.join(attack_plot_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            dataset_subset = attack_df[(attack_df['dataset'] == dataset) & (attack_df['method'] == method)]
            for synth_size in sorted(dataset_subset['synth_size'].unique()):
                synth_subset = dataset_subset[dataset_subset['synth_size'] == synth_size]
                if synth_subset.empty:
                    continue
                for metric in metrics:
                    plt.figure(figsize=(8, 5))
                    for attack_method in synth_subset['attack_method'].unique():
                        attack_method_subset = synth_subset[synth_subset['attack_method'] == attack_method]
                        plt.plot(
                            attack_method_subset['attack_set_size'],
                            attack_method_subset[metric],
                            marker='o',
                            label=attack_method
                        )
                    plt.xlabel('attack_set_size')
                    plt.ylabel(metric)
                    plt.title(f"{attack} - {dataset} - synth_size={synth_size} - {metric} vs attack_set_size")
                    plt.legend()
                    plt.tight_layout()
                    save_dir = os.path.join(method_dir, f"synth_{synth_size}")
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, f"{metric}_vs_attack_set_size.png"))
                    plt.close()

    # 2. For each fixed attack_to_synth_ratio, plot metric vs synth_size, colored by attack_method (do not mix datasets)
    for dataset in attack_df['dataset'].unique():
        attack_plot_dir = os.path.join(plot_dir,dataset ,attack)
        # if dataset == 'acs_income':
        dataset_subset = attack_df[attack_df['dataset'] == dataset]
        for method in dataset_subset['method'].unique():
            method_dir = os.path.join(attack_plot_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            dataset_subset = dataset_subset[dataset_subset['method'] == method]
            for ratio_bin in sorted(dataset_subset['attack_to_synth_ratio'].round(2).unique()):
                ratio_subset = dataset_subset[dataset_subset['attack_to_synth_ratio'].round(2) == ratio_bin]
                if ratio_subset.empty:
                    continue
                for metric in metrics:
                    plt.figure(figsize=(8, 5))
                    for attack_method in ratio_subset['attack_method'].unique():
                        attack_method_subset = ratio_subset[ratio_subset['attack_method'] == attack_method]
                        plt.plot(
                            attack_method_subset['synth_size'],
                            attack_method_subset[metric],
                            marker='o',
                            label=attack_method
                        )
                    plt.xlabel('synth_size')
                    plt.ylabel(metric)
                    plt.title(f"{attack} - {dataset} - ratio~{ratio_bin:.2f} - {metric} vs synth_size")
                    plt.legend()
                    plt.tight_layout()
                    save_dir = os.path.join(attack_plot_dir, f"ratio_{ratio_bin:.2f}", f"{metric}_vs_synth_size")
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, f"{attack}_{dataset}_ratio_{ratio_bin:.2f}_{metric}_vs_synth_size.png"))
                    plt.close()



    # used to work:
    #     for dataset in attack_df['dataset'].unique():
    #     attack_plot_dir = os.path.join(plot_dir,dataset ,attack)
    #     os.makedirs(attack_plot_dir, exist_ok=True)
    #     if dataset == 'acs_income':
    #         dataset_subset = attack_df[attack_df['dataset'] == dataset]
    #         for synth_size in sorted(dataset_subset['synth_size'].unique()):
    #             synth_subset = dataset_subset[dataset_subset['synth_size'] == synth_size]
    #             if synth_subset.empty:
    #                 continue
    #             for metric in metrics:
    #                 plt.figure(figsize=(8, 5))
    #                 for attack_method in synth_subset['attack_method'].unique():
    #                     attack_method_subset = synth_subset[synth_subset['attack_method'] == attack_method]
    #                     plt.plot(
    #                         attack_method_subset['attack_set_size'],
    #                         attack_method_subset[metric],
    #                         marker='o',
    #                         label=attack_method
    #                     )
    #                 plt.xlabel('attack_set_size')
    #                 plt.ylabel(metric)
    #                 plt.title(f"{attack} - {dataset} - synth_size={synth_size} - {metric} vs attack_set_size")
    #                 plt.legend()
    #                 plt.tight_layout()
    #                 save_dir = os.path.join(attack_plot_dir, f"synth_{synth_size}", f"{metric}_vs_attack_set_size")
    #                 os.makedirs(save_dir, exist_ok=True)
    #                 plt.savefig(os.path.join(save_dir, f"{attack_method_subset['method']}_synth_{synth_size}_{metric}_vs_attack_set_size.png"))
    #                 plt.close()

    # # 2. For each fixed attack_to_synth_ratio, plot metric vs synth_size, colored by attack_method (do not mix datasets)
    # for dataset in attack_df['dataset'].unique():
    #     attack_plot_dir = os.path.join(plot_dir,dataset ,attack)
    #     if dataset == 'acs_income':
    #         dataset_subset = attack_df[attack_df['dataset'] == dataset]
    #         for ratio_bin in sorted(dataset_subset['attack_to_synth_ratio'].round(2).unique()):
    #             ratio_subset = dataset_subset[dataset_subset['attack_to_synth_ratio'].round(2) == ratio_bin]
    #             if ratio_subset.empty:
    #                 continue
    #             for metric in metrics:
    #                 plt.figure(figsize=(8, 5))
    #                 for attack_method in ratio_subset['attack_method'].unique():
    #                     attack_method_subset = ratio_subset[ratio_subset['attack_method'] == attack_method]
    #                     plt.plot(
    #                         attack_method_subset['synth_size'],
    #                         attack_method_subset[metric],
    #                         marker='o',
    #                         label=attack_method
    #                     )
    #                 plt.xlabel('synth_size')
    #                 plt.ylabel(metric)
    #                 plt.title(f"{attack} - {dataset} - ratio~{ratio_bin:.2f} - {metric} vs synth_size")
    #                 plt.legend()
    #                 plt.tight_layout()
    #                 save_dir = os.path.join(attack_plot_dir, f"ratio_{ratio_bin:.2f}", f"{metric}_vs_synth_size")
    #                 os.makedirs(save_dir, exist_ok=True)
    #                 plt.savefig(os.path.join(save_dir, f"{attack}_{dataset}_ratio_{ratio_bin:.2f}_{metric}_vs_synth_size.png"))
    #                 plt.close()








    # for attack in attack_names:
    #     attack_df = average_df[average_df['attack_method'].str.contains(attack)]
    #     if attack_df.empty:
    #         continue
    #     attack_plot_dir = os.path.join(plot_dir, attack)
    #     os.makedirs(attack_plot_dir, exist_ok=True)

    #     # 1. For each fixed synth_size, plot metric vs attack_set_size
    #     for synth_size in sorted(attack_df['synth_size'].unique()):
    #         synth_subset = attack_df[attack_df['synth_size'] == synth_size]
    #         if synth_subset.empty:
    #             continue
    #         for metric in metrics:
    #             title = f"{attack} - synth_size={synth_size} - {metric} vs attack_set_size"
    #             save_dir = os.path.join(attack_plot_dir, f"synth_{synth_size}", f"{metric}_vs_attack_set_size")
    #             os.makedirs(save_dir, exist_ok=True)
    #             plot_comparison_plot(synth_subset, metric, 'attack_set_size', title, save_dir)

    #     # 2. For each fixed attack_to_synth_ratio, plot metric vs synth_size
    #     for ratio_bin in sorted(attack_df['attack_to_synth_ratio'].round(2).unique()):
    #         ratio_subset = attack_df[attack_df['attack_to_synth_ratio'].round(2) == ratio_bin]
    #         if ratio_subset.empty:
    #             continue
    #         for metric in metrics:
    #             title = f"{attack} - ratio~{ratio_bin:.2f} - {metric} vs synth_size"
    #             save_dir = os.path.join(attack_plot_dir, f"ratio_{ratio_bin:.2f}", f"{metric}_vs_synth_size")
    #             os.makedirs(save_dir, exist_ok=True)
    #             plot_comparison_plot(ratio_subset, metric, 'synth_size', title, save_dir)


    # for attack in attack_names:
    #     temp_attack = average_df[average_df['attack_method'].str.contains(attack)]
    #     plot_path = os.path.join(plot_dir, attack)
    #     # os.makedirs(plot_path, exist_ok=True)
    #     for metric in metrics:
    #         for x_label in x_labels:
    #             # Merge ratios that are close to each other (e.g., within 0.01)
    #             ratio_bins = temp_attack[ratio_label].round(1).unique()
    #             for ratio_bin in sorted(ratio_bins):
    #                 subset = temp_attack[temp_attack[ratio_label].round(1) == ratio_bin]
    #                 if subset.empty:
    #                     continue
    #                 title = f"{attack} - {metric} vs {x_label} (ratio~{ratio_bin:.2f})"
    #                 save_dir = os.path.join(plot_path, f"{metric}_vs_{x_label}_ratio_{ratio_bin:.2f}")
    #                 os.makedirs(save_dir, exist_ok=True)
    #                 plot_comparison_plot(subset, metric, x_label, title, save_dir)
        # Also plot metric vs ratio for each attack
        # title = f"{attack} - {metric} vs {ratio_label}"
        # save_dir = os.path.join(plot_path, f"{metric}_vs_{ratio_label}")
        # os.makedirs(save_dir, exist_ok=True)
        # plot_comparison_plot(temp_attack, metric, ratio_label, title, save_dir)

# attack_names = ['dcr', 'dpi','logan','classifier','domias','gen_lra','mc','majority_vote']
# for attack in attack_names:
#     temp_attack = average_df[average_df['attack_method'].str.contains(attack)]
#     # dcr_attacks['attack_to_synth_ratio'] = dcr_attacks['attack_set_size'] / dcr_attacks['synth_size']
#     plot_path = plot_dir + '/' + attack 
#     os.makedirs(plot_path, exist_ok=True)
#     plot_comparison_plot(temp_attack, 'auc_roc', 'synth_size', '{} Attacks: Performance vs Synth Set Size'.format(attack),plot_path)
#     plot_comparison_plot(temp_attack, 'auc_roc', 'attack_set_size', '{} Attacks: Performance vs attack Set Size'.format(attack),plot_path)
#     ### attack_set_size /synth_size
#     # dcr_attacks['attack_to_synth_ratio'] = dcr_attacks['attack_set_size'] / dcr_attacks['synth_size']
#     plot_comparison_plot(temp_attack, 'auc_roc', 'attack_to_synth_ratio', '{} Attacks: Performance vs Synth Set Size'.format(attack),plot_path)

#     plot_comparison_plot(temp_attack, 'tpr_at_fpr_0.001', 'synth_size', '{} Attacks: Performance vs Synth Set Size'.format(attack),plot_path)
#     plot_comparison_plot(temp_attack, 'tpr_at_fpr_0.001', 'attack_set_size', '{} Attacks: Performance vs Synth Set Size'.format(attack),plot_path)
#     plot_comparison_plot(temp_attack, 'tpr_at_fpr_0.001', 'attack_to_synth_ratio', '{} Attacks: Performance vs Synth Set Size'.format(attack),plot_path)

#     plot_comparison_plot(temp_attack, 'tpr_at_fpr_0.1', 'synth_size', '{} Attacks: Performance vs Synth Set Size'.format(attack),plot_path)
#     plot_comparison_plot(temp_attack, 'tpr_at_fpr_0.1', 'attack_set_size', '{} Attacks: Performance vs Synth Set Size'.format(attack),plot_path)
#     plot_comparison_plot(temp_attack, 'tpr_at_fpr_0.1', 'attack_to_synth_ratio', '{} Attacks: Performance vs Synth Set Size'.format(attack),plot_path)

