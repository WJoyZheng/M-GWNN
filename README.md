# M-GWNN: Multi-granularity graph wavelet neural networks for semi-supervised node classification. Zheng et al., Neurocomputing 2020. [pdf](https://doi.org/10.1016/j.neucom.2020.10.033)

## Description

We propose a multi-granularity graph neural network approach that combines the proposed Louvain-variant algorithm and jump connections to improve node representation. We first iteratively apply the proposed Louvain-variant algorithm to aggregate nodes into supernodes to build a continuously coarsened hierarchical graph, and then use jump connections to symmetrically refine the coarsened graph back to the original graph. In addition, a multi-layer GWNN is applied to propagate information in different granularity graphs throughout the process. M-GWNN shows a significant improvement compared with related works and recent references for the semi-supervised node classification task on four graph standard datasets, indicating that the method can effectively obtain the global information of the graph and alleviate the speed of GWNN oversmoothing, thus achieving significant results.



## Requirements
- Python (3.6)
- Tensorflow (1.9.0)


## Usage

You can conduct node classification experiments on citation network (Cora, Citeseer, Pubmed) or NELL with the following commands:

Run example:
```bash
python train.py --dataset cora --epochs 200 --early_stopping 1000 --learning_rate 0.01 --coarsen_level 2 --dropout 0.8 --weight_decay 9e-3 --hidden 32  --wavelet_s 0.6 --threshold 9e-6 
```


## Cite
Please cite our paper if you use this code in your own work:

```
@article{zheng2020m,
  title={M-GWNN: Multi-granularity graph wavelet neural networks for semi-supervised node classification},
  author={Zheng, Wenjie and Qian, Fulan and Zhao, Shu and Zhang, Yanping},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier}
}
```
