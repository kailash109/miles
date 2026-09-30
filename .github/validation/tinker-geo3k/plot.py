import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / 'plot-data.json').read_text())
plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot([x['batch'] for x in data['train']], [x['reward'] for x in data['train']])
axes[0].set_title('training reward')
axes[0].set_xlabel('batch')
axes[0].set_ylim(0, 1)
axes[1].plot([0, data['updates']], [data['evaluations']['eval-before']['accuracy'], data['evaluations']['eval-after']['accuracy']], marker='o')
axes[1].set_title('eval accuracy')
axes[1].set_xlabel('step')
axes[1].set_xticks([0, data['updates']])
axes[1].set_ylim(0, 1)
fig.tight_layout()
fig.savefig(root / 'geo3k-reward-accuracy.png', dpi=150)
