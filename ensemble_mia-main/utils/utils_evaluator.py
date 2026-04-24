"""
MIA evaluation utilities and ensemble methods
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import os
import time
from synth_mia.attackers import (gen_lra, dcr, dpi, logan, dcr_diff, 
                                  domias, mc, classifier, density_estimate, 
                                  local_neighborhood)


class MIAEvaluator:
    """
    Membership Inference Attack evaluator with ensemble methods
    """
    
    def __init__(self):
        self.attackers = [
            (dcr(hyper_parameters={"distance_type": 1})),
            (dcr(hyper_parameters={"distance_type": 2})),
            (dcr_diff(hyper_parameters={"distance_type": 1})),
            (dcr_diff(hyper_parameters={"distance_type": 2})),
            
            (gen_lra(hyper_parameters={'k_nearest': 1})),
            (gen_lra(hyper_parameters={'k_nearest': 5})),
            (gen_lra(hyper_parameters={'k_nearest': 10})),
            (gen_lra(hyper_parameters={'k_nearest': 20})),
            (gen_lra(hyper_parameters={'k_nearest': 50})),
            
            (dpi(hyper_parameters={'distance': 'l2', 'k_nearest': 5})),
            (dpi(hyper_parameters={'distance': 'l2', 'k_nearest': 10})),
            (dpi(hyper_parameters={'distance': 'l2', 'k_nearest': 15})),
            (dpi(hyper_parameters={'distance': 'l2', 'k_nearest': 20})),
            (dpi(hyper_parameters={'distance': 'l1', 'k_nearest': 5})),
            (dpi(hyper_parameters={'distance': 'l1', 'k_nearest': 10})),
            (dpi(hyper_parameters={'distance': 'l1', 'k_nearest': 15})),
            (dpi(hyper_parameters={'distance': 'l1', 'k_nearest': 20})),
            (logan()),
            (domias()),
            (mc()),
            (classifier())
        ]

        self.model_names = [
            'dcr_l1',
            'dcr_l2',
            'dcr_diff_l1',
            'dcr_diff_l2',
            'gen_lra_k1',
            'gen_lra_k5', 
            'gen_lra_k10',
            'gen_lra_k20',
            'gen_lra_k50',
            'dpi_l2_k5',
            'dpi_l2_k10',
            'dpi_l2_k15',
            'dpi_l2_k20',
            'dpi_l1_k5',
            'dpi_l1_k10',
            'dpi_l1_k15',
            'dpi_l1_k20',
            'logan',
            'domias',
            'mc',
            'classifier'
        ]

    def _run_individual_attacks(self, mem, non_mem, synth, ref):
        """Run individual MIA attacks and return score matrix"""
        score_matrix = []
        true_labels = None

        for attacker in self.attackers:
            labels, scores = attacker.attack(mem, non_mem, synth, ref)
            score_matrix.append(scores)
            if true_labels is None:
                true_labels = labels
        
        score_matrix = np.array(score_matrix).T
        return pd.DataFrame(score_matrix, columns=self.model_names), true_labels

    def _create_ensemble_methods(self, score_df, embeddings_df=None):
        """Create ensemble methods from individual attack scores"""
        if embeddings_df is not None:
            base_df = embeddings_df.reset_index(drop=True)
        else:
            base_df = score_df.reset_index(drop=True)

        result_df = score_df.copy().reset_index(drop=True)

        # Statistical aggregation methods
        result_df['mean'] = base_df.mean(axis=1)
        result_df['median'] = base_df.median(axis=1)

        result_df['q25'] = base_df.quantile(0.25, axis=1)
        result_df['q75'] = base_df.quantile(0.75, axis=1)
        result_df['iqr'] = result_df['q75'] - result_df['q25']

        result_df['max'] = base_df.max(axis=1)
        result_df['min'] = base_df.min(axis=1)
        result_df['range'] = result_df['max'] - result_df['min']

        # Majority voting methods
        def majority_vote(base_df, threshold=None, percentile=None):
            """
            Majority voting based on binary predictions from each model.
            
            Args:
                base_df: DataFrame of scores
                threshold: Fixed threshold value (overrides percentile)
                percentile: Percentile (0-100) to use as threshold for each column
                        If None and threshold is None, uses 50th percentile (median)
            
            Returns proportion of models voting 'member' (1).
            """
            if threshold is not None:
                # Use fixed global threshold
                binary_votes = (base_df >= threshold).astype(int)
            elif percentile is not None:
                # Use column-wise percentile as threshold
                thresholds = base_df.quantile(percentile / 100.0, axis=0)
                binary_votes = (base_df >= thresholds).astype(int)
            else:
                # Default: use column-wise median (50th percentile)
                thresholds = base_df.median(axis=0)
                binary_votes = (base_df >= thresholds).astype(int)
            
            # Return proportion of positive votes (models predicting membership)
            return binary_votes.mean(axis=1)
        
        # Add majority voting methods
        result_df['majority_vote_median'] = majority_vote(base_df)
        for percentile in range(90, 99, 1):
            result_df[f'majority_vote_{percentile}'] = majority_vote(base_df, percentile=percentile)
        
        return result_df

    def _evaluate_methods(self, score_df, true_labels,dataset,method,size,synth_size):
        """Evaluate all methods using multiple metrics and save ROC AUC curves"""
        import matplotlib.pyplot as plt

        metrics = ["auc_roc", 'tpr_at_fpr_0', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01', 'tpr_at_fpr_0.1']

        results = {}
        evaluator = classifier()

        for col in score_df.columns:
            scores = np.array(pd.to_numeric(score_df[col], errors='coerce'))
            eval_result = evaluator.eval(true_labels, scores, use_decision_metrics=True)

            method_results = {}
            for metric in metrics:
                method_results[metric] = eval_result.get(metric, np.nan)

            results[col] = method_results
            

            # Draw and save ROC curve for this method
            fpr, tpr, _ = roc_curve(true_labels, scores)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {col}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            output_dir = f'outputs/{dataset}/{method}/sytnh_size_{synth_size}/attack_size_{size}'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'roc_curve_{col}.png'))
            plt.close()

            results[col] = method_results

        return results




    def _evaluate_resutls(self, score_df, true_labels,dataset,method,seed,size,synth_size,running_time):
        """Evaluate all methods using multiple metrics and save ROC/PR curves.

        Returns per-attack dict with: 'auc', 'fpr', 'tpr', 'pr_auc', 'precision', 'recall'.
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score

        metrics = ["auc_roc", 'tpr_at_fpr_0', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01', 'tpr_at_fpr_0.1']

        results = {}
        evaluator = classifier()

        for col in score_df.columns:
            scores = np.array(pd.to_numeric(score_df[col], errors='coerce'))
            eval_result = evaluator.eval(true_labels, scores, use_decision_metrics=True)

            method_results = {}
            for metric in metrics:
                method_results[metric] = eval_result.get(metric, np.nan)

            # results[col] = method_results
            

            # Draw and save ROC curve for this method
            fpr, tpr, _ = roc_curve(true_labels, scores)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {col}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            output_dir = f'outputs/{dataset}/{seed}/{method}/synth_size_{synth_size}/attack_size_{size}'
            # output_dir = f'outputs/{dataset}/{method}/sytnh_size_{synth_size}/attack_size_{size}'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'roc_curve_{col}.png'))
            plt.close()

            # Precision-Recall curve and AP
            precision, recall, _ = precision_recall_curve(true_labels, scores)
            pr_auc = average_precision_score(true_labels, scores)
            method_results['pr_auc'] = pr_auc
            method_results['precision'] = precision.tolist()
            method_results['recall'] = recall.tolist()
            method_results['runtime'] = running_time

            plt.figure()
            plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {col}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'pr_curve_{col}.png'))
            plt.close()

            results[col] = method_results

        return results
    ###original code with commented parts
    # def _evaluate_methods(self, score_df, true_labels):
    #     """Evaluate all methods using multiple metrics"""
    #     metrics = ["auc_roc", 'tpr_at_fpr_0', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01', 'tpr_at_fpr_0.1']

    #     results = {}
    #     evaluator = classifier()

    #     for col in score_df.columns:
    #         scores = np.array(pd.to_numeric(score_df[col], errors='coerce'))
    #         eval_result = evaluator.eval(true_labels, scores, use_decision_metrics=True)

    #         method_results = {}
    #         for metric in metrics:
    #             method_results[metric] = eval_result.get(metric, np.nan)

    #         results[col] = method_results

    #     return results


    def evaluate(self, mem, non_mem, ref, synth, dataset, seed, method,size,synth_size):
        """Main evaluation method"""
        t0 =  time.perf_counter()
        score_df, true_labels = self._run_individual_attacks(
            mem, non_mem, ref, synth
        )
        ###### here we have all scores and all true labels, it is possible to draw the graph
        
        final_score_df = self._create_ensemble_methods(score_df)
        running_time = time.perf_counter() - t0
        output_dir = f'outputs/{dataset}/{seed}/{method}/synth_size_{synth_size}/attack_size_{size}'
        os.makedirs(output_dir, exist_ok=True)
        final_score_df.to_csv(f'{output_dir}/all_scores.csv', index=False)
        # results = self._evaluate_methods(final_score_df, true_labels, dataset, method,size,synth_size)
        results = self._evaluate_resutls(final_score_df, true_labels, dataset, method,seed,size,synth_size,running_time)

        return results
