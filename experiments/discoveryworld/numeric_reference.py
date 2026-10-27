"""Train-only leave-one-instance-out OLS/ridge and finite-mixture reference.

Only a held instance's already received support frequencies fit its local
models. Its future labels score predictions; both queries are excluded from
scaling, regularization selection and empirical expert-prior estimation.
These are standard numerical references, not a deployed DiscoveryWorld policy.

Example (from the repository root)::

    python -m experiments.discoveryworld.numeric_reference --train-trajectories \
        runs/discoveryworld/reference/train \
        runs/discoveryworld/reference/campaign2/train \
        runs/discoveryworld/reference/calibration_training/train \
        --output runs/discoveryworld/calibration/numeric_reference --workers 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import gzip
import json
import subprocess
import numpy as np
from .calibration import CalibrationLog, FEATURE_QUANTITIES

QUANTITIES = tuple(FEATURE_QUANTITIES)
ALPHAS = (0., 1e-6, 1e-4, .01, .1, 1., 10., 100.)
TEMPERATURES = (1., 3., 10., 30., 100., 300., 1000., 3000.)


def extract(path):
    path = str(path)
    run = json.loads((Path(path).parent/'run.json').read_text())
    if run.get('split') != 'train':
        raise ValueError(f'This reference accepts training trajectories only: {path}')
    log = CalibrationLog()
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            event = {k:r[k] for k in ('index','action','action_result')}
            event['observation'] = {'ui':r['observation']['ui']}
            event['next_observation'] = {'ui':r['next_observation']['ui']}
            log.append_transition(event)
    samples = []
    for example in log.query_examples:
        o, label = example.opportunity, example.label
        if not o.clean:
            continue
        assert label.received_sequence >= o.decision_sequence
        assert o.query.object_handle not in {s.label.object_handle for s in o.support}
        assert all(record.sequence < o.decision_sequence for record in o.query.records)
        assert all(s.label.received_sequence < o.decision_sequence for s in o.support)
        samples.append({'seed':run['seed'],'source':path,'decision_sequence':o.decision_sequence,
                        'support_count':len(o.support),'query':asdict(o.query),
                        'support':[asdict(s) for s in o.support],
                        'evaluation_only_target':asdict(label)})
    return {'source':path,'sha256':sha256(Path(path).read_bytes()).hexdigest(), 'samples':samples}


def quantity_vector(query):
    """Nine native-unit means; latest received condition per named quantity.

    Accepts CrystalQuery or its asdict representation. Original records retain
    all condition groups, statistics and units. No cross-location average is
    formed, and no frequency label participates in condition selection.
    """
    query = asdict(query) if not isinstance(query, dict) else query
    groups = [query['features'][name] for name in QUANTITIES]
    if any(not rows for rows in groups):
        raise ValueError('All nine instrument quantities must be observed')
    selected = [max(rows, key=lambda row: (row['last_sequence'], row['first_sequence'])) for rows in groups]
    return np.array([row['mean'] for row in selected], dtype=np.float64)


def affine_fit(X, y, x):
    """OLS per column: query predictions[D], support fits[N,D], support MSE[D]."""
    X, y, x = np.asarray(X), np.asarray(y), np.asarray(x)
    center,ym = X.mean(axis=0),y.mean()
    variance = np.sum((X-center)**2,axis=0)
    cov = np.sum((X-center)*(y-ym)[:,None],axis=0)
    slope = np.divide(cov,variance,out=np.zeros_like(cov),where=variance>0)
    fit = ym + (X-center)*slope
    prediction = ym + (x-center)*slope
    mse = np.mean((fit-y[:,None])**2,axis=0)
    return prediction,fit,mse


def ridge(data,scale,alpha):
    X,y,x = data['X']/scale,data['y'],data['x']/scale
    center,ym = X.mean(axis=0),y.mean()
    z = X-center
    if alpha == 0:
        w = np.linalg.pinv(z) @ (y-ym)
    else:
        w = z.T @ np.linalg.solve(z@z.T+alpha*np.eye(len(y)), y-ym)
    return float(ym+(x-center)@w),ym+z@w


def mixture(data,temperature,prior=None):
    pred,fit,mse = data['affine']
    prior = np.ones(len(pred))/len(pred) if prior is None else prior
    logits = np.log(prior) - mse/(2*temperature**2)
    weights = np.exp(logits-logits.max()); weights /= weights.sum()
    return float(pred@weights),fit@weights,weights


def learned_prior(train,temperature):
    # Fit expert frequencies on OTHER instances only by Gaussian mixture likelihood.
    # Test query targets are absent. The floor is a symmetric Dirichlet pseudocount.
    errors = np.stack([((d['affine'][0]-d['target'])/temperature)**2 for d in train])
    scores = -.5 * errors
    weights = np.ones(len(QUANTITIES))/len(QUANTITIES)
    for _ in range(30):
        logits = scores + np.log(weights)
        resp = np.exp(logits-logits.max(axis=1,keepdims=True))
        resp /= resp.sum(axis=1,keepdims=True)
        weights = (resp.sum(axis=0)+.1)/(len(train)+.1*len(weights))
    return weights


def stats(rows):
    errors = np.array([abs(r['prediction_hz']-r['target_hz']) for r in rows])
    fit_errors = np.concatenate([r['support_abs_errors_hz'] for r in rows])
    return {'queries':len(rows),'future_mae_hz':float(errors.mean()),
            'future_median_absolute_error_hz':float(np.median(errors)),
            'future_rmse_hz':float(np.sqrt(np.mean(errors**2))),
            'future_max_absolute_error_hz':float(errors.max()),
            'within_2hz_of_public_working_setting':int(np.sum(errors<2)),
            'known_support_mae_hz':float(fit_errors.mean()),
            'known_support_max_absolute_error_hz':float(fit_errors.max())}


def evaluate_samples(raw):
    """LOIO analysis of causal examples; seed identifies folds, never features."""
    data=[]
    for r in raw:
        d = {'seed':r['seed'],'source':r['source'],'decision_sequence':r['decision_sequence'],
             'support_count':r['support_count'],'x':quantity_vector(r['query']),
             'X':np.stack([quantity_vector(s['query']) for s in r['support']]),
             'y':np.array([s['label']['frequency_hz'] for s in r['support']]),
             'target':r['evaluation_only_target']['frequency_hz']}
        d['affine']=affine_fit(d['X'],d['y'],d['x']); data.append(d)
    if len({d['seed'] for d in data}) < 2:
        raise ValueError('LOIO requires at least two usable training instances')
    results=[]
    for seed in sorted({d['seed'] for d in data}):
        train=[d for d in data if d['seed']!=seed]
        test=[d for d in data if d['seed']==seed]
        scale=np.std(np.vstack([np.vstack([d['X'],d['x']]) for d in train]),axis=0)
        scale=np.where(scale>0,scale,1.)
        for d in test:
            same_n=[other for other in train if other['support_count']==d['support_count']]
            alpha=min(ALPHAS,key=lambda a:np.mean([abs(ridge(other,scale,a)[0]-other['target']) for other in same_n]))
            temperature=min(TEMPERATURES,key=lambda t:np.mean([abs(mixture(other,t)[0]-other['target']) for other in same_n]))
            # Training-only empirical expert prior, with the same CV-selected temperature.
            prior=learned_prior(same_n,temperature)
            mean=float(d['y'].mean())
            candidates={'support_mean':(mean,np.full(len(d['y']),mean),{}),
                        'ridge_min_norm':(*ridge(d,scale,0.),{'alpha':0.}),
                        'ridge_train_fold_selected':(*ridge(d,scale,alpha),{'alpha':alpha}),
                        'affine_uniform_prior':(*mixture(d,temperature)[:2],{'temperature_hz':temperature,'weights':mixture(d,temperature)[2].tolist()}),
                        'affine_train_fold_prior':(*mixture(d,temperature,prior)[:2],{'temperature_hz':temperature,'prior':prior.tolist(),'weights':mixture(d,temperature,prior)[2].tolist()})}
            for j,name in enumerate(QUANTITIES):
                candidates['affine_quantity_'+name]=(float(d['affine'][0][j]),d['affine'][1][:,j],{})
            for name,(prediction,fit,settings) in candidates.items():
                results.append({'seed':seed,'source':d['source'],'decision_sequence':d['decision_sequence'],
                                'support_count':d['support_count'],'method':name,'prediction_hz':prediction,
                                'target_hz':d['target'],'support_targets_hz':d['y'].tolist(),
                                'support_predictions_hz':fit.tolist(),
                                'support_abs_errors_hz':np.abs(fit-d['y']).tolist(),'settings':settings})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-trajectories', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    paths = sorted({str(path.resolve()) for root in args.train_trajectories
                    for path in ([root] if root.is_file() else root.rglob('trajectory.jsonl'))})
    if not paths:
        raise ValueError('No training trajectories found')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        episodes = list(pool.map(extract, paths))
    raw = [row for episode in episodes for row in episode['samples']]
    identities = [rows[0]['seed'] for episode in episodes if (rows := episode['samples'])]
    if len(identities) != len(set(identities)):
        raise ValueError('Use only one source trajectory per training instance')
    results = evaluate_samples(raw)
    vectors = [query for row in raw for query in [row['query'], *(support['query'] for support in row['support'])]]
    methods={name:{'all':stats([r for r in results if r['method']==name]),
                   'by_support_count':{str(n):stats([r for r in results if r['method']==name and r['support_count']==n]) for n in (2,3)}}
             for name in sorted({r['method'] for r in results})}
    report={'schema':'public_numeric_reference_v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'instances':len(identities),'queries':len(raw),'zero_query_inputs':[e['source'] for e in episodes if not e['samples']],
            'condition_audit':{'vectors':len(vectors),'quantity_rows':sum(len(query['features'][name]) for query in vectors for name in QUANTITIES),
                               'vectors_with_multiple_condition_groups':sum(any(len(query['features'][name])>1 for name in QUANTITIES) for query in vectors)},
            'methodology':{'split':'Leave one entire instance out of training data only; both queries of that instance are excluded from scaling/hyperparameter/prior fitting.',
                           'features':list(QUANTITIES),'representation':'Native-unit mean of latest received condition per named quantity, selected by last_sequence then first_sequence, never labels. Full condition rows, units, sample counts/ranges and causal sequences are preserved in public_inputs.json.gz; no cross-condition pooling.',
                           'ridge':'Fit held-instance then-available support using standardized nine-dimensional inputs and unpenalized intercept. Standardization uses other instances only. Alpha selected by their future-query MAE separately for support2/3.',
                           'affine':'Standard OLS slope/intercept per quantity on held-instance support. Mixture posterior is prior_j * exp(-support_MSE_j/(2*temperature^2)); no held-target-dependent weight. Temperature selected using other instances only. Empirical-prior variant estimates expert frequencies with 30 EM steps and .1 symmetric pseudocount using other instances future labels only; it is not a discoveryworld rule label or an original formula.',
                           'grids':{'ridge_alphas':ALPHAS,'mixture_temperatures_hz':TEMPERATURES},
                           'limits':['Numerical reference, not the main CDM; no policy deployment or native activation test. Within2Hz refers to an observed public valid setting, not hidden exact resonance.',
                                     'Two support points interpolate every nonconstant single-variable affine candidate, so support residual alone cannot identify one. A third point can distinguish the candidates represented here, but near-collinearity or noisy conditions can remain ambiguous.',
                                     'A nine-dimensional affine function has ten coefficients. Two or three supports underdetermine a general sum of quantities; a mixture of single-variable affine experts is not a general additive or nonlinear law learner. Low three-support error on these training instances does not establish held-law transfer.',
                                     'No validation, heldout, seed modulo, hidden rule, or held-instance future target enters predictions. The empirical expert prior reflects only the other training instances, so it favors their observed patterns.',
                                     'Unconstrained numerical predictions are not clipped to the native control range. Ill-conditioned two-point slopes can extrapolate far outside it.']},
            'methods':methods}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'public_inputs.json.gz').write_bytes(gzip.compress(json.dumps(episodes).encode()))
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    for name,groups in methods.items():
        if name.startswith('affine_quantity_'): continue
        print(name,json.dumps(groups['by_support_count']),flush=True)
    print('saved',args.output,flush=True)

if __name__=='__main__': main()
