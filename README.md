# Nested Sampling for Factor Graph (NSFG)
This is a collection of packages we implemented for directly drawing samples from posterior distributions in SLAM problems. The posterior distributions are represented as factor graphs here. Our packages get these factor graphs ready for using sampling approaches including the No-U-Turn sampler, sequential Monte Carlo (SMC), and nested sampling. These samping approaches are implemented in our dependencies such as pymc3 and dynesty.

The following instruction was tested on Ubuntu 18.04 with conda.

## Installation
```
git clone git@github.com:MarineRoboticsGroup/nsfg.git
cd nsfg
conda create -n nsfg python=3.8
conda activate nsfg
pip install -r requirements.txt
python src/setup.py install
```

## Running an example
```
python dns_smc_nuts_batch.py
```