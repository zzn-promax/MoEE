# Baseline papers and code sources

The table records the original papers and available author-maintained code
for the comparison methods. This repository does not distribute baseline
implementations or run them. Obtain each baseline from its own source and
follow that source's terms.

| CLI name | Original paper | Author-maintained / official code |
|---|---|---|
| `MLP` | [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0) | Generic two-layer PyTorch implementation; no paper-specific official repository |
| `GCN` | [Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/forum?id=SJU4ayYgl) | [tkipf/gcn](https://github.com/tkipf/gcn) |
| `GAT` | [Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ) | [PetarV-/GAT](https://github.com/PetarV-/GAT) |
| `GraphSAGE` | [Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) | [williamleif/GraphSAGE](https://github.com/williamleif/GraphSAGE) |
| `APPNP` | [Predict then Propagate: Graph Neural Networks Meet Personalized PageRank](https://openreview.net/forum?id=H1gL-2A9Ym) | [klicperajo/ppnp](https://github.com/klicperajo/ppnp) |
| `GCNII` | [Simple and Deep Graph Convolutional Networks](https://proceedings.mlr.press/v119/chen20v.html) | [chennnM/GCNII](https://github.com/chennnM/GCNII) |
| `H2GCN` | [Beyond Homophily in Graph Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/58ae23d878a47004366189884c2f8440-Abstract.html) | [GemsLab/H2GCN](https://github.com/GemsLab/H2GCN) |
| `GPRGNN` | [Adaptive Universal Generalized PageRank Graph Neural Network](https://openreview.net/forum?id=n6jl7fLxrP) | [jianhao2016/GPRGNN](https://github.com/jianhao2016/GPRGNN) |
| `GloGNN` | [Finding Global Homophily in Graph Neural Networks When Meeting Heterophily](https://proceedings.mlr.press/v162/li22ad.html) | [RecklessRonan/GloGNN](https://github.com/RecklessRonan/GloGNN) |
| `HOGGCN` | [Powerful Graph Convolutional Networks with Adaptive Propagation Mechanism for Homophily and Heterophily](https://ojs.aaai.org/index.php/AAAI/article/view/20340) | [hedongxiao-tju/HOG-GCN](https://github.com/hedongxiao-tju/HOG-GCN) |
| `OrderedGNN` | [Ordered GNN: Ordering Message Passing to Deal with Heterophily and Over-Smoothing](https://openreview.net/forum?id=wKPmPBHSnT6) | [LUMIA-Group/OrderedGNN](https://github.com/LUMIA-Group/OrderedGNN) |
| `M2MGNN` | [Sign Is Not a Remedy: Multiset-to-Multiset Message Passing for Learning on Heterophilic Graphs](https://proceedings.mlr.press/v235/liang24c.html) | [Jinx-byebye/m2mgnn](https://github.com/Jinx-byebye/m2mgnn) |
| `N2` | [Towards Dynamic Message Passing on Graphs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93b7e2780c4f6599837fdd3718c51fad-Abstract-Conference.html) | [sunjss/N2](https://github.com/sunjss/N2) |
| `GGCN` | [Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks](https://doi.org/10.1109/ICDM54844.2022.00169) | [Yujun-Yan/Heterophily_and_oversmoothing](https://github.com/Yujun-Yan/Heterophily_and_oversmoothing) |
| `GBKGNN` | [GBK-GNN: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily](https://doi.org/10.1145/3485447.3512201) | [Xzh0u/GBK-GNN](https://github.com/Xzh0u/GBK-GNN) |
| `MixHop` | [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](https://proceedings.mlr.press/v97/abu-el-haija19a.html) | [samihaija/mixhop](https://github.com/samihaija/mixhop) |
| `ACMGCN` | [Revisiting Heterophily for Graph Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html) | [SitaoLuan/ACM-GNN](https://github.com/SitaoLuan/ACM-GNN) |
| `GMoE` | [Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html) | [VITA-Group/Graph-Mixture-of-Experts](https://github.com/VITA-Group/Graph-Mixture-of-Experts) |
| `Mowst` | [Mixture of Weak and Strong Experts on Graphs](https://openreview.net/forum?id=wYvuY60SdD) | [facebookresearch/mowst-gnn](https://github.com/facebookresearch/mowst-gnn) |
| `DAMoE` | [DA-MoE: Addressing Depth-Sensitivity in Graph-Level Analysis through Mixture of Experts](https://doi.org/10.1016/j.neunet.2025.108064) | [official node-classification path, pinned commit](https://github.com/Celin-Yao/DA-MoE/tree/5658217d4312604a220c9c807ada39d04e216eb1) |
| `GNNMoE` | [Mixture of Experts Meets Decoupled Message Passing](https://doi.org/10.1145/3701716.3715462) | [official GAT path, pinned commit](https://github.com/GISec-Team/GNNMoE/tree/81b00fcbb7a2c16ceaa2ed6e99eb56d7c74f87a4) |

In the paper, DA-MoE uses the official node-classification path adapted from
its OGB datasets to the other benchmarks. GNNMoE uses the official GAT path:
the pinned upstream version references `GCNNorm` and `SAGENorm` but does not
define those modules.
