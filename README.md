-------------------------------------
CODE FOR CS231N PAPER
-------------------------------------
DISTILLING KNOWLEDGE FROM SPECIALIST
CONVOLUTIONAL NEURAL NETWORKS FOR
CLUSTERED CLASSIFICATION
-------------------------------------

Table of Contents
=================


Abstract
============
Most realistic datasets for computer vision tasks tend to have a large number of classes, which are unevenly distributed in the label space, and can even be clustered in categories, like in popular benchmark datasets such as ImageNet or CIFAR-100. Typical convolutional neural networks often fail to generalize well on these datasets, especially when the number or image per class is small. A natural idea, when one does not want to work with huge networks that are impossible to transfer to small devices (both for memory and time constraints), would be to train an ensemble of experts, each one specialized on a subset of the dataset's classes. However, those expert networks tend to overfit a lot. To address this issue, we propose to leverage the concept of knowledge distillation, recently proposed by Hinton \etal ~\cite{darkknowledge}, to train those networks. This technique can act as a very strong regularizer, and can allow us to achieve good results on this type of dataset, with a significant speed-up (both for training and prediction) and memory gain.

After introducing the theoretical foundations of knowledge distillation, we present the different components of the necessary pipeline in the case of specialist networks, and various ways of improving the results. We also show and discuss our experiments and results on a particular dataset, CIFAR-100, which classes presents a natural clustered structure.
