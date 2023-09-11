import yaml
from easydict import EasyDict

hparams_path = '/root/Competitions/DACON/4. JUMP AI 2023/lightning_logs/0.1st_RUN/2023-09-04/MLM/baseline_model[GNN+mol]/hparams.yaml'
with open(hparams_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

print(config)

args = EasyDict(config).args
print(args)

print(args.FF_hidden_dim)