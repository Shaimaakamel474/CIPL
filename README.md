# Cross- and Intra-image Prototypical Learning (CIPL)

### Pytorch implementation for the paper "[Cross- and Intra-image Prototypical Learning for Multi-label Disease Diagnosis and Interpretation](https://ieeexplore.ieee.org/document/10887396)" at IEEE TMI 2025.


In this work, we present the Cross- and Intra-image Prototypical Learning (CIPL) framework for accurate multi-label disease diagnosis and interpretation.
CIPL takes advantage of cross-image common semantics to disentangle multiple diseases during the prototype learning, ensuring high-quality prototypes in the multi-label interpretation setting.
Additionally, a two-level alignment-based regularization strategy enhances interpretation robustness and predictive performance by enforcing consistent intra-image information.
Email: chongwangsmu@gmail.com.

<div align=center>
<img width="900" height="390" src="https://github.com/cwangrun/CIPL/blob/master/arch/arch.png"/></dev>
</div>


## Datasets:
Chest X-ray ([NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)) and fundus ([ODIR](https://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72)) images are publicly available.


## Training and Testing:
1. Run python main.py to train the model and evaluate its disease diagnosis accuracy. Our trained models are provided at [ChestX-ray14](https://drive.google.com/file/d/1svxfab5YG2BVoSKe99krhwWeqQyQFUqw/view?usp=drive_link) and [ODIR](https://drive.google.com/file/d/1ykIhO6d2AqFO0Wy4Rmr4VIzvTVeoQIaQ/view?usp=drive_link):
2. Each prototype is visualized as the nearest non-repetitive training patch representing its corresponding disease class using push.py.


## Interpretable reasoning:
CIPL leverages disentangled class prototypes, learned from the training set, as anchors for diagnostic reasoning.
To understand the decision process for a given test image, run interpretable_reasoning.py. 
This will generate a set of similarity (activation) maps that highlight the correspondence between the test image and the prototypes of each disease class, providing insights into the model's reasoning.

<div align=center>
<img width="630" height="400" src="https://github.com/cwangrun/CIPL/blob/master/arch/reasoning.png"/></dev>
</div>



## Results:
CIPL demonstrates high-quality visual prototypes that are both disentangled and accurate (aligning well with actual lesion signs), outperforming previous studies. For further details, please refer to our paper.

<div align=center>
<img width="850" height="400" src="https://github.com/cwangrun/CIPL/blob/master/arch/prototype.png"/></dev>
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
