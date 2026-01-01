########################################################################################
           “**Graph Classification with Topological Feature Augmentation**”
#########################################################################################
**Graph classification : Message passing (GINConv) :Graph-level pooling (global mean pooling)**
     **Structural/topological features (degree, clustering, PageRank, k-core, betweenness)**
##########################################################################################
**Task** :Given a graph, predict its class label.
############################################################################################
**Dataset** MUTAG (from TUDataset) Small, fast to download
Non-trivial structural patterns
Binary classification (imbalanced-ish)
#############################################################################################
**Why this is challenging**
Node features alone are weak
Structural/topological signals matter
Different feature combinations change performance subtly
#############################################################################################
**Objective Metric**: Macro F1-score (preferred)
Accuracy allowed for sanity check
###############################################################################################
**Constraints**
No external datasets
Must run under 10 minutes on CPU
Any GNN covered in lectures 1.1–4.6
No pretraining
##################################################################################################
