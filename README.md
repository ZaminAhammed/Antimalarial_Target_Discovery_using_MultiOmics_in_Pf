# Antimalarial_Target_Discovery_using_MultiOmics_in_Pf

## Abstract

The emergence of antimalarial resistance in Plasmodium parasites necessitates an urgent development of novel therapies. With the availability of vast amounts of multi-omics data, we propose to build an integrative machine learning (ML) model that can predict novel Plasmodium therapeutic targets. First, we compiled a multimodal dataset that integrates genetic variability (population genomics), gene expression (transcriptomics), gene essentiality (mutagenesis screens), selectivity (BLASTp), and ligandability (predicted pockets) for 5300 protein-coding genes of Plasmodium falciparum, the most lethal malaria parasite. Next, we systematically curated a positive training set consisting of 100 established antimalarial targets, and a negative training set consisting of 114 genes that are dispensable for parasite growth and survival. Using the standardized training set, we evaluated the performance of five different ML models using 10-fold cross validation: Logistic Regression, Support Vector Machine, Random Forest, eXtreme Gradient Boosting (XGB), and Tabular Prior-data Fitted Network (TabPFN). The best model so far has an F1 score of 0.934 (𝜎 = 0.055), using which potential therapeutic antimalarial targets have been identified from an unlabelled test set. These can be further prioritised based on their risk of developing resistance and validated experimentally in vitro cultures.

## Objectives
1.  Collect publicly available data that could directly or indirectly indicate the essentiality and therapeutic targetability of different P. falciparum genes.
2.  Integrate the data to develop a multimodal model that classifies genes as potential novel targets for therapeutic development.

## Workplan
![Screenshot 2025-05-28 221447](https://github.com/user-attachments/assets/ad9992c6-c336-48d6-9200-6c273e6c49c7)

##Codes
Run models_evaluation.py as:
  python3 models_evaltion.py /path/to/training/set

Run classify_genes.py as:
  python3 classify_genes.py /path/to/training/set /path/to/test/set /path/to/output/file.csv

## Conclusion
1. Features indicating key aspects of an ideal drug target including essentiality, druggability, and selectivity, were successfully integrated.
2. An XGB model performed the best when trained with a set of “positive” and “negative” genes curated by literature review and was used to classify genes in the testing set to identify potential drug targets.
3. Top 50 potential targets were shortlisted for experimental validation based on the probability of positive classification and include DNA/RNA-binding protein PfAlba4, highly druggable transporters such as PfNT4 and PfDMT2, four conserved proteins of unknown function with no homologs in the host, etc.
