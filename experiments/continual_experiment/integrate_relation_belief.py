"""Attach a trained response-law predictor to an inherited CDM checkpoint."""
import argparse
from pathlib import Path

from cdm.relation_belief import RelationBelief
from experiments.continual_chemistry.learned_belief import load_model
from .shared_tasks import load_shared_checkpoint, save_shared_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--predictor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior', type=float, default=.5)
    args = p.parse_args()
    model, _ = load_shared_checkpoint(args.parent)
    predictor, record = load_model(args.predictor)
    if not isinstance(predictor, RelationBelief):
        p.error('This integration uses the shared latent response-function model')
    if record['update'] != record['configuration']['updates']:
        p.error('Use the fixed final predictor checkpoint')
    if not 0 < args.prior < 1:
        p.error('The two predictor branches require a prior strictly between zero and one')
    policy = model.tasks['chemistry']
    policy.relation_belief = predictor
    policy.relation_prior = args.prior
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        p.error('Output already exists')
    save_shared_checkpoint(args.output, model, training=dict(
        parent=str(args.parent), predictor=str(args.predictor), predictor_updates=record['update'],
        inherited_core_updates=0, integration='Prequential evidence mixture; inherited action ranking and memories'))


if __name__ == '__main__':
    main()
