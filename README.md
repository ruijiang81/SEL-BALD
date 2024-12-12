## Implementation for SEL-BALD

**SEL-BALD: Deep Bayesian Active Learning for Selective Labeling with Instance Rejection**<br>
Ruijiang Gao, Mingzhang Yin, Maytal Saar-Tsechansky <br>

Abstract: *Machine learning systems are widely used in many high-stakes contexts in which
experimental designs for assigning treatments are infeasible. When evaluating
a decision instance is costly, such as investigating a fraud case, or evaluating a
biopsy decision, a sample-efficient strategy is needed. However, while existing
active learning methods assume humans will always label the instances selected by
the machine learning model, in many critical applications, humans may decline
to label instances selected by the machine learning model due to reasons such
as regulation constraint, domain knowledge, or algorithmic aversion, thus not
sample efficient. In this paper, we study the Active Learning with Instance
Rejection (ALIR) problem, which is an active learning problem that considers
the human discretion behavior for high-stakes decision making problems. We
propose new active learning algorithms under deep Bayesian active learning for
selective labeling (SEL-BALD) to address the ALIR problem. Our algorithms
consider how to acquire information for both the machine learning model and the
human discretion model. We conduct experiments on both synthetic and real-world
datasets to demonstrate the effectiveness of our proposed algorithms.*


### Run Experiments

```
bash run_syn.sh 
bash run_adult.sh 
bash run_mushroom.sh 
```

In ALIR, the benefit of Joint-BALD and UCB/TS variants are significant when the human discretion behaviors are unknown and heterogeneous. When human discretion behaviors are homogeneous, Naive-BALD remains optimal. When human discretion behaviors are known, e-BALD is optimal. 