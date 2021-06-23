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

## Installing using Docker
This code can be run using Docker.
1. Install Docker and Docker Compose at https://docs.docker.com/get-docker/.
2. Clone this github repository.
3. From the top-level of this repository (i.e. the directory containing docker-compose.yml), run `docker-compose up`. This will spin up a container and start a Jupyter notebook server. 
4. Check out the example notebooks in the examples directory. Note that Jupyter only has write access to the examples directory (this is due to permissions), and so you can only create and edit `.ipynb` files in the examples directory.
5. You can edit the source code (in Solve, Analysis, Utils, or Visualizations) from a text editor of your choice.
6. To stop the server, enter `Ctrl + c` in the command line.
7. To shut down the container, run `docker-compose down`. By default, any work you do in the examples directory is saved by Docker. If you want to reset the examples directory to its initial state, run `docker-compose down --volume` instead.

## Installing without Docker
To run this code without Docker, install Julia, Jupyter notebook, and the necessary packages:
1. You can install Julia here: https://julialang.org/downloads/
2. And you can install Jupyterlab/Jupyter notebook here: https://jupyter.org/install
3. After cloning this repository, run `julia install.jl` from the top-level of this repository. This will install the necessary packages.
4. Run `jupyter notebook` from anywhere to start the server.