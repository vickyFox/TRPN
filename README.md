# TRPN
## Introduction
A pytorch implementation of the IJCAI2020 paper "[Transductive Relation-Propagation Network for Few-shot Learning](https://www.ijcai.org/Proceedings/2020/0112.pdf)". The code is based on [Edge-labeling Graph Neural Network for Few-shot Learning](https://github.com/khy0809/fewshot-egnn)

**Author:** Yuqing Ma, Shihao Bai, Shan An, Wei Liu, Aishan Liu, Xiantong Zhen and Xianglong Liu

**Abstract:** Few-shot learning, aiming to learn novel concepts from few labeled examples, is an interesting and very challenging problem with many practical advantages. To accomplish this task, one should concentrate on revealing the accurate relations of the support-query pairs. We propose a transductive relation-propagation graph neural network (TRPN) to explicitly model and propagate such relations across support-query pairs. Our TRPN treats the relation of each support-query pair as a graph node, named relational node, and resorts to the known relations between support samples, including both intra-class commonality and inter-class uniqueness, to guide the relation propagation in the graph, generating the discriminative relation embeddings for support-query pairs. A pseudo relational node is further introduced to propagate the query characteristics, and a fast, yet effective transductive learning strategy is devised to fully exploit the relation information among different queries. To the best of our knowledge, this is the first work that explicitly takes the relations of support-query pairs into consideration in few-shot learning, which might offer a new way to solve the few-shot learning problem. Extensive experiments conducted on several benchmark datasets demonstrate that our method can significantly outperform a variety of state-of-the-art few-shot learning methods.

## Requirements
* Python 3
* Python packages
  - pytorch 1.0.0
  - torchvision 0.2.2
  - matplotlib
  - numpy
  - pillow
  - tensorboardX

An NVIDIA GPU and CUDA 9.0 or higher. 

## Getting started
### mini-ImageNet
You can download miniImagenet dataset from [here](https://drive.google.com/drive/folders/15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w).

### tiered-ImageNet
You can download tieredImagenet dataset from [here](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view?usp=drive_open).


Because WRN has a large amount of parameters. You can save the extracted feature before the classifaction layer to increase train or test speed. Here we provide the features extracted by WRN:
* miniImageNet: [train](https://drive.google.com/file/d/1uJ5-NhdDkdkqRhyrQoXKgkqoLt3BqWSC/view?usp=sharing), [val](https://drive.google.com/file/d/1p_6kalUR-a2so1yOGUn1DCAXL3ftgl-r/view?usp=sharing), [test](https://drive.google.com/file/d/1z69BN3ReZfSwpOt3P1l1LPDdqigKdsfT/view?usp=sharing)
* tieredImageNet: [train](https://drive.google.com/file/d/1Hz1Z4jVj8O3NQejUnpKeR9UTAVVdDw8T/view?usp=sharing), [val](https://drive.google.com/file/d/1DQ-LsyWtFsi6oyTxnBa5nQrla6lY7x0M/view?usp=sharing), [test](https://drive.google.com/file/d/1dGtfL8EEplJmiXGgxmQNtI36FYKyp-XG/view?usp=sharing)

You also can use our [pretrained WRN model](https://drive.google.com/drive/folders/1o51s2F7_bpG2k6JOgE9loYtSRIdOH2qc) to generate features for mini or tiered by yourself

## Training
```
# ************************** miniImagenet, 5way 1shot  *****************************
$ python3 conv4_train.py --dataset mini --num_ways 5 --num_shots 1 
$ python3 WRN_train.py --dataset mini --num_ways 5 --num_shots 1 

# ************************** miniImagenet, 5way 5shot *****************************
$ python3 conv4_train.py --dataset mini --num_ways 5 --num_shots 5 
$ python3 WRN_train.py --dataset mini --num_ways 5 --num_shots 5 

# ************************** tieredImagenet, 5way 1shot *****************************
$ python3 conv4_train.py --dataset tiered --num_ways 5 --num_shots 1 
$ python3 WRN_train.py --dataset tiered --num_ways 5 --num_shots 1 

# ************************** tieredImagenet, 5way 5shot *****************************
$ python3 conv4_train.py --dataset tiered --num_ways 5 --num_shots 5 
$ python3 WRN_train.py --dataset tiered --num_ways 5 --num_shots 5 

# **************** miniImagenet, 5way 5shot, 20% labeled (semi) *********************
$ python3 conv4_train.py --dataset mini --num_ways 5 --num_shots 5 --num_unlabeled 4

```
You can download our pretrained model from [here](https://drive.google.com/drive/folders/1irkD1RFrbG03F3KGIWjwjyx62QD2akt5?usp=sharing) to reproduce the results of the paper.
## Testing
``` 
# ************************** miniImagenet, Cway Kshot *****************************
$ python3 conv4_eval.py --test_model your_path --dataset mini --num_ways C --num_shots K 
$ python3 WRN_eval.py --test_model your_path --dataset mini --num_ways C --num_shots K 


```
