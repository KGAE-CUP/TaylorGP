# TaylorGP
Taylor Genetic Programming for Symbolic Regression
# 1. Introduction
TaylorGP, A Symbolic Regression method, leverages a Taylor polynomial to approximate the symbolic equation that fits the dataset. It also utilizes the Taylor polynomial to extract the features of the symbolic equation: low order polynomial discrimination, variable separability, boundary, monotonic, and parity. GP is enhanced
by these Taylor polynomial techniques. Experiments are conducted on three kinds of benchmarks: classical SR, machine learning, and physics. The experimental results show that TaylorGP not only has
higher accuracy than the nine baseline methods, but also is faster in finding stable results.

This paper has been accepted by GECCO-22, see our paper for more.
# 2. Code
## Projects
<!-- Note that for the simplicity of experimental analysis, we divide the SRNet into 2 projects, namely 
**srnet-clas** and **srnet-reg**, for classification task and regression task respectively. It is easy
to combine both projects into one single project since the code of SRNet (package at `srnet-clas/CGPNet` 
or `srnet-reg/CGPNet`) is easy to implement for both classification task and regression task. -->

## Requirements
Make sure you have installed the following python version and pacakges before start running our code:
- python3.6~3.8
- scikit-learn 
- numpy 
- sympy 
- pandas 
- time 
- copy 
- itertools 
- timeout_decorator 
- scipy 
- joblib 
- numbers 
- itertools 
- abc 
- warnings 
- math

Our experiments were running in Ubuntu 18.04 with Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz. 

# 3. Experiments
<!-- For both regression and classification task, see `srnet-clas/README.md` and `srnet-reg/README.md` for
more details about how to reproduce our experimental results. -->

<!-- Here we show our experimental figures in our paper:
## Convergence 
<!-- The convergence curves for all dataset:
![Regression convergence curve of fitness](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-reg/IMG/trend_result.png)
![Classification convergence curve of fitness](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/trend_result.png) -->

## Semantics (Mathematical Expressions)
![Hidden Semantics](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/IMG/hidden_semantics.png)

Combining these expressions, we can obtain the overall expressions for all NNs:
![Semantics](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/IMG/semantics.png)

## Comparison
We compare SRNet to LIME and MAPLE on both regression and classification tasks.

### Regression
![Regression Comparison](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/IMG/reg_comparison.png)

### Classifiation
Decision boundary:

![DB of SRNet vs. LIME vs. MAPLE](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/local_db.png)

Accuracy:

![Acc of SRNet vs. LIME vs. MAPLE](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/accs.png) -->

# 4. Cite
Please **cite** our paper if you use the code.
