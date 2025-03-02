# SAA for Hyperparameter Estimation in Bayesian Inverse Problems

Here we provide MATLAB implementations of the SAA approach for hyperparameter estimation, as well as two example scripts featuring a static seismic inverse problem and a dynamic seismic inverse problem.

## Project Description

In Bayesian inverse problems, it is common to consider several hyperparameters that define the prior and the noise model that must be estimated from the data. In particular, we are interested in linear inverse problems with additive Gaussian noise and Gaussian priors defined using Mat\'{e}rn covariance models. We estimate the hyperparameters using the maximum a posteriori (MAP) estimate of the marginalized posterior distribution.

We consider a stochastic average approximation (SAA) of the objective function and use the preconditioned Lanczos method to compute efficient approximations of the function and gradient evaluations 
 
## Installation 
### Software language

       MATLAB 9.14 (R2023a)
       For those without access to MATLAB, Octave provides an alternative platform.  
       Note that these codes have not been tested in Octave. 

### Requirements
The MainDrivers require the following package:

    "IR tools: A MATLAB Package of Iterative Regularization"
    by Silvia Gazzola, Per Christian Hansen and James G. Nagy
    https://github.com/jnagy1/IRtools.git

    genHyBR: https://github.com/juliannechung/genHyBR

    tensorlab: https://tensorlab.net

## How to Use
For Experiment 1 in [1], the main drivers include:
    
    Ex1_MC.m          Compares accuracy of Monte Carlo (MC) estimators for the
                                static seismic example, for different numbers of MC samples.                                
    Ex1_noise.m       Compares the accuracy of the Monte Carlo estimators for the
                                static seismic example, for various Matern kernels and for 
                                various noise levels.
    Ex1_opt.m         Compares the accuracy of the Monte Carlo estimators for the
                                static seismic example, at different function evaluations 
                                during optimization.                              
    Ex1_prec_rank.m   compares the accuracy of the Monte Carlo estimators for the
                                static seismic example, for various Matern kernels and for various
                                preconditioner ranks.
    Ex1_timings.m     Compares timings for the Monte Carlo estimators for the
                                static seismic example, for various numbers of measurements.
                                

To run these codes, open a MATLAB command window, and type 
     
     >> Ex1_MC [press enter/return]
     >> Ex1_noise [press enter/return]
     >> Ex1_opt [press enter/return]
     >> Ex1_prec_rank [press enter/return]
     >> Ex1_timings [press enter/return]

For Experiment 2 in [1], the main drivers include:

    >> Ex2_dynamic.m   % Driver for synthetic dynamic seismic tomography inverse problem

To run these codes, open a MATLAB command window, and type 
     
     >> Ex2_dynamic [press enter/return]
     
### Contributors
        Julianne Chung, 
        Department of Mathematics, Emory University
        
        Malena Sabaté Landman, 
        Department of Mathematics, Oxford University UK
        
        Scot M. Miller, 
        Department of Environmental Health and Engineering, Johns Hopkins University
        
        Arvind K. Saibaba, 
        Department of Mathematics, North Carolina State University
	
## Licensing

If you use this codes, you *must* cite the original authors:

       [1] "Efficient sample average approximation techniques for hyperparameter estimation in Bayesian inverse problems". 2024.


[MIT](LICENSE)

## Acknowledgement

This work was partially supported by the National Science Foundation under grants DMS-2411197, DMS-2208294, DMS-2341843, DMS-2026830, DMS-2411198, and DMS-2026835. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
