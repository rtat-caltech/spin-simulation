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

## Using Docker
This code can be run using Docker.
1. Install Docker and Docker Compose at https://docs.docker.com/get-docker/.
2. Clone this github repository.
3. From the top-level of this repository (i.e. the directory containing docker-compose.yml), run `docker-compose up`. This will spin up a container and start a Jupyter notebook server.
4. To stop the server, enter `Ctrl + c` in the command line.
5. To shut down the container, run `docker-compose down`.
