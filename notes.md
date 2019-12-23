## Install
```bash
pip install "trfl[tensorflow]"
git clone git@github.com:thibaultallart/bsuite.git
pip install -e "bsuite[baselines]"
pip install jupyter
python -m ipykernel install --user --name bsuite --display-name "bsuite"
```

## Example
run dqn on bandit
```bash 
python bsuite/baselines/dqn/run.py --bsuite_id=BANDIT
```

run random_list on bandit_list
```bash 
python bsuite/baselines/random_list/run.py --bsuite_id=BANDIT_LIST --overwrite=True
```