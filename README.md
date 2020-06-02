## 1 Cassava Disease Classification by transfer learning
Classify pictures of cassava leaves into 1 of 5 disease categories (or healthy)
In this competition, we introduced a dataset of 5 fine-grained cassava leaf disease categories with 9,436 labeled images collected during a regular survey in Uganda, mostly crowdsourced from farmers taking images of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab in Makarere University, Kampala.
The dataset consists of leaf images of the cassava plant, with 9,436 annotated images and 12,595 unlabeled images of cassava leaves.

After preprocessing the data set we used the transfer concept which is when a model developed for one task is reused to work on a second task. In that case the correspondim]ng model is a called pretrained model.
here the pretrained model we used is se_resnext50_32x4d which is pretrained on ImageNet classification  with 1000 images. And so the most important thing of the transfer learning is the change of the number of classes according to the new number of classes. 



## 2 Data_challenge_on_kernel_methods

The motivation of this data-challenge project is to practice what we did in the Kernel Methods AMMI 2020 for machine learning.
The way we do the practice is to classify a data set about transcription factor. During this classification task we should predict whether a DNA sequence region 
is binding site to a specific transcription factor.
Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: 
bound or unbound.
In this challenge, we will work with three datasets corresponding to three different TFs.
For that, we have been provided 2000 data sample data labeled to learn our model which should be able to classifies as possible as 1000 sample unlabled.
