# Nested Sampling for Factor Graph (NSFG)
This is a collection of packages we implemented for directly drawing samples from posterior distributions in SLAM problems. The posterior distributions are represented as factor graphs here. Our packages get these factor graphs ready for using sampling approaches including the No-U-Turn sampler, sequential Monte Carlo, and nested sampling. These samping approaches are implemented in our dependencies such as pymc3 and dynesty.

The following instruction was tested on Ubuntu 18.04 with Conda.

## Requirements on Ubuntu(>=18.04)
```
sudo apt-get install gcc libc6-dev
sudo apt-get install gfortran libgfortran3
```
and we recommend to install NSFG using conda environment. The default env name in the environment.yml is nsfg.

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

## Running an example
```
cd example
python dns_smc_nuts_batch.py
```
