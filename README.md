# Cross- and Intra-image Prototypical Learning (CIPL)

### Pytorch implementation for the paper "[Cross- and Intra-image Prototypical Learning for Multi-label Disease Diagnosis and Interpretation](https://ieeexplore.ieee.org/document/10887396)" at IEEE TMI 2025.


In this work, we present a Cross- and Intra-image Prototypical Learning (CIPL) framework, for accurate multi-label disease diagnosis and interpretation.
CIPL takes advantage of cross-image common semantics to disentangle the multiple diseases when learning the prototypes.
Two-level alignment-based regularisation strategy leverages consistent intra-image information to enhance interpretation robustness and predictive performance.


<div align=center>
<img width="800" height="400" src="https://github.com/cwangrun/CIPL/blob/master/arch/arch.png"/></dev>
</div>


## Datasets:
Chest X-ray ([NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)) and fundus images ([ODIR](https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72)) are publicly available.


## Training and Testing:
1. python main.py to train a model and test its disease diagnosis accuracy. Our trained models:
2. each prototype is visulised as the nearest non-repetitive training patch of the corresponding disease class (push.py).


## Interpretable reasoning:
CIPL relies on disentangled class prototypes learned from the training set as anchors for the diagnosis reasoning. For a given test image, run interpretable_reasoning.py to display a set of similarity (activation) maps of the test image to the prototypes of each disease class, tracing the decision process. 

<div align=center>
<img width="660" height="275" src="https://github.com/cwangrun/CIPL/blob/master/arch/reasoning.png"/></dev>
</div>



## Results:
1. Please refer to our paper for more results.

<div align=center>
<img width="460" height="255" src="https://github.com/cwangrun/CIPL/blob/master/arch/prototype.png"/></dev>
</div>



## Citation:
```
@article{wang2025cross,
  title={Cross-and Intra-image Prototypical Learning for Multi-label Disease Diagnosis and Interpretation},
  author={Wang, Chong and Liu, Fengbei and Chen, Yuanhong and Frazer, Helen and Carneiro, Gustavo},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```
