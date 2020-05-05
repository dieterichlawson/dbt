# Distributed Bayesian Triangulation

Requires python3, JAX, numpy, scipy, and tensorboard.

To install the required dependencies, run:

```
# Make a virtualenv in the .env directory
python3 -m venv .env
# Activate the virtualenv
source .env/bin/activate
# Install requirements
pip3 install jax jaxlib tensorflow tensorboard scipy numpy matplotlib
```

The main file is `dbt.py`. An example run is:

```
python3 dbt.py \
  --num_points=3 \
  --censorship_temp=1. \
  --distance_threshold=5. \
  --distance_variance=0.005 \
  --logdir=/tmp/dbt
```

You can view summaries produced by the binary by running a `tensorboard` pointed at `logdir`.
