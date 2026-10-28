"""Deterministic experiment orchestration helpers and strict result parsing."""
import json
from pathlib import Path


def read_final(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    metrics=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    values=[m for m in metrics if 'val_loss' in m]
    if not values:
        raise ValueError(f'No validation records: {path}')
    last=values[-1]
    if not (0 < last['val_loss'] < float('inf')):
        raise ValueError(f'Invalid validation loss: {path}')
    return {'val_loss':last['val_loss'],'tokens_seen':last['tokens_seen'],
            'parameters':last['parameters'],'tok_per_s':last['tok_per_s'],
            'step':last['step']}


def choose(baseline, candidate, tolerance=0.002):
    """Only promote a candidate with lower validation loss by >= tolerance."""
    return candidate['val_loss'] < baseline['val_loss'] - tolerance
