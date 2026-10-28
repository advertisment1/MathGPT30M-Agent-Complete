import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('metrics');a=p.parse_args()
rows=[json.loads(x) for x in Path(a.metrics).read_text().splitlines() if x.strip()]
plt.plot([x['tokens_seen'] for x in rows],[x['train_loss'] for x in rows],label='train')
v=[x for x in rows if 'val_loss' in x];plt.plot([x['tokens_seen'] for x in v],[x['val_loss'] for x in v],label='validation')
plt.xlabel('Tokens seen');plt.ylabel('Cross entropy');plt.legend();plt.tight_layout();plt.savefig(Path(a.metrics).with_suffix('.png'),dpi=180)
