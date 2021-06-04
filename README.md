# spin-simulation

This is Julia code primarily used to study critical spin dressing for nEDM@SNS. It consists of 2 main packages:

1. Solve: Uses Julia's DifferentialEquations library to integrate the Bloch equations. The code supports arbitrary Bx(t), By(t), and Bz(t).
    The most important methods to know are run_simulations and NoiseIterator.
    This also includes code for noise generation. The spectrum of the generated noise can be modified with a filter.
2. Analysis: Contains a variety of convenience methods to analyze the results produced by Solve.
    The most important methods to know are aggregate_data and extract_noise.
    
Aside from these two, there are two other packages:

3. Utils: Various utility methods
4. Visualizations: A tool for visualizing the results produced by Solve (see SpinSolverAnalysis.ipynb example script).

See the example scripts for more info.
