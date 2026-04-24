# *Mutatis Mutandis*: Revisiting the Comparator in Discrimination Testing

This is the repository for the working paper *Mutatis Mutandis: Revisiting the Comparator in Discrimination Testing*. 

We use both R and Python. For implementing the experiments in Section 4, use the law school dataset in data/. The scripts are in src/. Before running the discrimination tools via run_exp_, first create the counterfactual dataset via gen_cf_, which is stored in data/. We already provide both datasets, though. Use analysis_ for the figures. Under the current setup, the RStan models are not required. 


Local implementation:
Script to generate augmented dataset: 
src/get_cf_data_general.py

input arguments: 
dataset_name: adult and compas
augmentation_method: update_labels, add_comparators' 

update_labels: keeps original instances, updates their label to match the one of priviledged comparators
add comparators: do not change original data, for non_privilesged instances adds their comparator with the same label for original not_priviledged