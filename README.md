# Nested Sampling for Factor Graph (NSFG)
This is a collection of packages we implemented for directly drawing samples from posterior distributions in SLAM problems. The posterior distributions are represented as factor graphs here. Our packages get these factor graphs ready for using sampling approaches including the No-U-Turn sampler (NUTS), sequential Monte Carlo (NUTS), and nested sampling (NS). These sampling approaches are implemented in our dependencies such as [PyMC3](https://docs.pymc.io/en/v3/) and [dynesty](https://dynesty.readthedocs.io/en/stable/).

The following instruction was tested on Ubuntu 18.04 and 20.04 with Miniconda.

## Requirements on Ubuntu
```
sudo apt-get update
sudo apt-get install gcc libc6-dev
sudo apt-get install g++
sudo apt-get install gfortran
sudo apt-get install libgfortran3
(to install libgfortran3 for ubuntu20.04.md, follow https://gist.github.com/sakethramanujam/faf5b677b6505437dbdd82170ac55322)
sudo apt-get install libsuitesparse-dev
sudo apt-get install python3 python3-dev
```
We recommend to install NSFG using conda environments. The default env name in the environment.yml is nsfg.

## Installation
```
git clone git@github.com:MarineRoboticsGroup/nsfg.git
cd nsfg
conda env create -f environment.yml
conda activate nsfg
pip3 install --upgrade TransportMaps
pip3 install -r requirements.txt
python setup.py install
```

## Examples
We provide five simulated and one real-world examples to showcase results of NSFG and other solvers in comparison. You can find factor graph files ( * .fg) and python scripts that solve these factor graphs in the file folder `example`. We also provide a few python scripts to plot these results.
### General-purpose sampling techniques (NSFG, NS, NUTS, SMC)
Please refer to the script `example/simulated_dataset/ns_smc_nuts_batch.py` for an example that solves our factor graphs using NSFG, NS, NUTS, and SMC. NSFG and NS only differ in prior distributions. The priors in NS are predetermined uniform distributions while NSFG chooses a spanning tree from a SLAM factor graph to create the prior distribution. You can change the variable `parent_dirs` in that script to explore more examples.
```
cd example/simulated_dataset
python ns_smc_nuts_batch.py
```
### GTSAM
We also prepared C++ scripts to parse our factor graphs for [GTSAM](https://github.com/borglab/gtsam). To solve our factor graphs using GTSAM, you have to install [GTSAM](https://github.com/borglab/gtsam) first and then build our parser by running
```
cd src/external/gtsam
mkdir build
cd build
cmake ..
make
```
Now you should see `gtsam_solution` in the `build` folder. Go back to the example folder `example/simulated_dataset` and modify the path of `gtsam_solution` accordingly in the bash script `run_gtsam.sh`. To get GTSAM solutions, you can run
```
cd example/simulated_dataset
source run_gtsam.sh
```
### NF-iSAM
Please refer to the code base of [NF - iSAM](https://github.com/borglab/gtsam](https://github.com/MarineRoboticsGroup/NF-iSAM)
### Sampling importance resampling (SIR) particle filter
We also implemented the SIR particle filter. We tested it using our factor graphs produced from the Plaza1 dataset. Run an example of the particle filter as follows:
```
cd example/plaza1_dataset
python sir_batch.py
```
