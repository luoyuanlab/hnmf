# Hybrid Non-negative Matrix Factorization
This repository hosts the core code to perform Hybrid Non-negative Matrix Factorization (HNMF) by alternating projected gradient method. This code is suitable for applications such as integrating phenotype (e.g., vitals, physiologic measurements) and genotype information for patient stratification. Under these scenarios, phenotype matrix typically has continuous entries while genotype matrix typically has count entries. Unlike previous methods, HNMF approximates phenotype matrix under Frobenius loss, and genotype matrix under Kullback-Leibler (KL) loss. We implemented an alternating projected gradient method to solve the approximation problem.

### Requirements
This code is tested on Matlab R2016a

### To run the code
```
[F, Gp, Gg, errorp, errorg] = hnmf(Xp, Xg, k, lambda, timelimit, maxiter, tol, maxsubiter)

Output:
  %% F - patient group matrix
  %% Gp - phenotype group matrix
  %% Gg - genotype group matrix
  %% errorp - phenotype matrix approximation error
  %% errorg - genotype matrix approximation error
  
Input:
  %% n - number of patients
  %% mp - number of phenotypes
  %% mg - number of genotypes
  %% k - number of patient groups
  %% Xp - patient by phenotype matrix, continuous value (n x mp)
  %% Xg - patient by genotype matrix, count value (n x mg)
  %% F - patient group assignment matrix (k x n)
  %% Gp - phenotype group assignment matrix (k x mp)
  %% Gg - genotype group assignment matrix (k x mg)
  %% initF,initGp,initGg - initial solution
  %% tol - tolerance for a relative stopping condition
  %% timelimit, maxiter - limit of time and iterations
```
### Citation
```
@article{luo2018hnmf,
  title={Integrating Hypertension Phenotype and Genotype with Hybrid Non-negative Matrix Factorization},
  author={Luo, Yuan and Mao, Chengsheng and Yang, Yiben and Wang, Fei and Ahmad, Faraz S. and Arnett, Donna and Irvin, Marguerite R. and Shah, Sanjiv J.},
  journal={Bioinformatics},
  url={https://doi.org/10.1093/bioinformatics/bty804},
  year={2018}
}
```
