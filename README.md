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

## Examples

We provide an example to test whether the module required by Taylor GP is successfully installed: 
python TaylorGP.py

In addition, you can run the specified dataset through the following method: 
python TaylorGP.py --fileName="Feynman/F24.tsv"

# 3. Experiments

## DataSet

We evaluate the performance of TaylorGP on three kinds of benchmarks: classical Symbolic Regression Benchmarks (SRB), Penn
Machine Learning Benchmarks (PMLB), and Feynman Symbolic Regression Benchmarks (FSRB) .(You could get them from directories GECCO, PMLB and Feynman respectively).The distribution of the total 81 benchmark sizes by samples and features is shown in the following. 
<img src="https://github.com/KGAE-CUP/TaylorGP/blob/main/img/datasets_size.png" width="50%">

The details of these benchmarks are listed in the [appendix](https://github.com/KGAE-CUP/TaylorGP/blob/main/img/Appendix_Taylor_Symbolic_Regression_GECCO2022.pdf).

## Performance

We compare TaylorGP with two kinds of baseline algorithms \footnote{The nine baseline algorithms are implemented in [SRBench](https://github.com/cavalab/srbench) : four symbolic regression methods and five machine learning methods. The symbolic regression methods include [**GPlearn**](https://github.com/trevorstephens/gplearn), FFX , geometric semantic genetic programming (**GSGP**) and bayesian symbolic regression (**BSR**). The machine learning methods include linear regression (**LR**), kernel ridge regression (**KR**), random forest regression (**RF**), support vector machines (**SVM**), and **XGBoost** . 

As shown in the figure below , we illustrate the normalized R^2 scores of the ten algorithms running 30 times on all benchmarks. Since the normalized $R^2$ closer to 1 indicates better results, overall TaylorGP can find more accurate results than other algorithms.

<img src="https://github.com/KGAE-CUP/TaylorGP/blob/main/img/contact.jpg" width="50%">


Figure \ref{fig:GECCO_Feynman_ML_Box} illustrates that TaylorGP, when compared with the nine baseline algorithms, can obtain more accurate and stable results on the two benchmarks, SRB and FSRB. However, on the benchmark PMLB, the two algorithms, FFX and XGBoost, outperform TaylorGP. 

![描述](https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg)![描述](https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg)![描述](https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg)

<center class="half">
<img src="https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg" width="25%"/><img src="https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg" width="25%"/><img src="https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg" width="25%"/>
</center>
<figure class="third">
<img src="https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg" width="30%"/><img src="https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg" width="30%"/><img src="https://github.com/KGAE-CUP/TaylorGP/blob/main/img/GECCO.jpg" width="30%"/>
</figure>

<center class="half">
<img src="https://img-blog.csdn.net/2018061215200776?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="200"/><img src="https://img-blog.csdn.net/20180612152032532?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="200"/><img src="https://img-blog.csdn.net/20180612152100203?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="200"/>
</center>

# 4. Cite

Please **cite** our paper if you use the code.
