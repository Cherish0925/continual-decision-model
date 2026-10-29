"""Approximate Bayesian experimental design without a neural action policy.

Nine affine relation hypotheses share the train-only numerical prior used by
CDM. Public measurement uncertainty enters each recursive coefficient update.
Measurement bundles are valued by Monte Carlo one-step improvement in the best
remaining target-success probability, divided by observed execution cost.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr, logsumexp

GROUPS = ((0,), (1,), (2,), (3,), (4, 5, 6, 7, 8))


def union(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


@dataclass
class Ledger:
    x: np.ndarray
    variance: np.ndarray
    seen: np.ndarray
    y: np.ndarray
    known: np.ndarray
    counts: np.ndarray
    query: int
    failures: list

    def copy(self):
        return Ledger(self.x.copy(), self.variance.copy(), self.seen.copy(), self.y.copy(),
                      self.known.copy(), self.counts.copy(), self.query, list(self.failures))


class BayesianDesigner:
    def __init__(self, prior, marginal, *, mode="reset", samples=8, seed=0, forgetting=.95, iterations=3):
        self.norm = prior.normalization
        self.x_mean = np.asarray(self.norm['x_mean'])
        self.x_scale = np.asarray(self.norm['x_scale'])
        self.y_mean = self.norm['y_mean_hz']; self.y_scale = self.norm['y_scale_hz']
        p = prior._prior()
        self.fresh = (p['coefficient_mean'].detach().numpy().copy(),
                      p['coefficient_covariance'].detach().numpy().copy(),
                      p['log_weights'].detach().numpy().copy())
        self.marginal = marginal
        self.mode, self.samples, self.forgetting = mode, samples, forgetting
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)
        self.memories = []
        # Cost estimates from a previously completed development cohort only.
        self.cost = np.array([2.59, 2.76, 8.40, 2.69, 2.57, 19.24])
        self.cost_counts = np.ones(6)*10

    def contexts(self):
        contexts = [self.fresh]
        age = [0.]
        if self.mode == 'forgetting':
            for index, (mean, cov, weights) in enumerate(self.memories):
                contexts.append((mean, cov + np.eye(2)[None]*.0025, weights))
                age.append(math.log(self.forgetting)*(len(self.memories)-index))
        mass = np.full(len(contexts), -math.log(len(contexts))) + np.array(age)
        mass -= logsumexp(mass)
        return contexts, mass

    def posterior(self, ledger):
        contexts, context_mass = self.contexts()
        prior_means = np.stack([c[0] for c in contexts])
        prior_cov = np.stack([c[1] for c in contexts])
        base_logw = np.stack([c[2] for c in contexts]) + context_mass[:, None]
        xn = (ledger.x-self.x_mean)/self.x_scale
        vx = ledger.variance/self.x_scale**2
        moment = prior_means[...,1]**2+prior_cov[...,1,1]
        # Refit from the same prior, never append labels twice. Errors-in-input
        # variance depends on the fitted slope; three moment updates avoid
        # treating the near-zero initial slope as the final noise scale.
        for _ in range(max(1,self.iterations)):
            means, cov, logw = prior_means.copy(), prior_cov.copy(), base_logw.copy()
            for obj in np.flatnonzero(ledger.known):
                target = (ledger.y[obj]-self.y_mean)/self.y_scale
                phi = np.stack((np.ones(9), xn[obj]), -1)
                prediction = (means*phi).sum(-1)
                projected = np.einsum('cqij,qj->cqi', cov, phi)
                noise_moment = (means[...,1]**2+cov[...,1,1]) if self.iterations == 0 else moment
                measured_noise = (2/math.sqrt(3)/self.y_scale)**2 + noise_moment*vx[obj]
                variance = (projected*phi).sum(-1) + measured_noise
                missing = ~ledger.seen[obj]
                missing_variance = cov[...,0,0]+means[...,1]**2+cov[...,1,1]+1.
                effective_prediction = np.where(missing[None], means[...,0], prediction)
                effective_variance = np.where(missing[None], missing_variance, variance).clip(1e-12)
                residual = target-effective_prediction
                logw += -.5*(residual**2/effective_variance+np.log(2*np.pi*effective_variance))
                gain = projected/variance[...,None].clip(1e-12)
                means += np.where(missing[None,:,None], 0., gain*residual[...,None])
                cov -= np.where(missing[None,:,None,None], 0.,
                                gain[..., :, None]*projected[...,None,:])
                cov = (cov+cov.swapaxes(-1,-2))*.5
            moment = means[...,1]**2+cov[...,1,1]
        logw -= logsumexp(logw)
        return {'means':means, 'cov':cov, 'weights':np.exp(logw)}

    def predict(self, ledger, posterior):
        q=ledger.query
        xn=(ledger.x[q]-self.x_mean)/self.x_scale
        phi=np.stack((np.ones(9),xn),-1)
        mean=(posterior['means']*phi).sum(-1)*self.y_scale+self.y_mean
        variance=np.einsum('qi,cqij,qj->cq',phi,posterior['cov'],phi)
        variance += (posterior['means'][...,1]**2+posterior['cov'][...,1,1])*(ledger.variance[q]/self.x_scale**2)
        variance = variance*self.y_scale**2
        # Predict latent resonance; the activation tolerance is applied once
        # by best_test, not added again as future label noise.
        std=np.sqrt(np.maximum(variance,1e-8))
        weights=posterior['weights'].copy()
        intervals=union([(x-2,x+2) for x in ledger.failures])
        remaining=np.ones_like(mean)
        for lo,hi in intervals:
            remaining-=ndtr((hi-mean)/std)-ndtr((lo-mean)/std)
        weights*=np.maximum(remaining,1e-14)
        weights/=weights.sum()
        return mean,std,weights,intervals,remaining

    def best_test(self, ledger, posterior, control, current, *, local_window=2.):
        means,std,weights,excluded,remaining=self.predict(ledger,posterior)
        eligible=ledger.seen[ledger.query][None] & ((ledger.seen & ledger.known[:,None]).sum(0)[None]>=2)
        candidates=means[eligible.repeat(len(means),axis=0)].tolist()
        candidates += [edge+delta for interval in excluded for edge in interval for delta in (-2,2)]
        if not candidates:
            return {'probability':0.,'setting':current,'std':1.,'relation':0}
        low,high=control['lower_hz'],control['upper_hz']; step=control['minimum_step_hz']
        candidates=np.asarray(candidates)
        candidates=np.clip(current+np.round((candidates-current)/step)*step,low,high)
        candidates=np.unique(candidates)
        if ledger.failures:
            candidates=candidates[np.min(abs(candidates[:,None]-np.array(ledger.failures)[None]),axis=1)>step*.25]
        if not len(candidates):
            return {'probability':0.,'setting':current,'std':1.,'relation':0}
        left=candidates[:,None,None]-local_window;right=candidates[:,None,None]+local_window
        mass=ndtr((right-means)/std)-ndtr((left-means)/std)
        for lo,hi in excluded:
            l=np.maximum(left,lo);r=np.minimum(right,hi)
            mass-=np.where(r>l,ndtr((r-means)/std)-ndtr((l-means)/std),0)
        conditional=np.maximum(mass,0)/np.maximum(remaining,1e-14)
        prob=(conditional*weights).sum((1,2))
        # Public adjustment distance affects cost, not physical likelihood.
        travel=1+np.log1p(abs(candidates-current))/10
        k=np.argmax(prob/travel)
        contribution=conditional[k]*weights
        c,r=np.unravel_index(contribution.argmax(),contribution.shape)
        return {'probability':float(prob[k]),'setting':float(candidates[k]),
                'std':float(std[c,r]),'relation':int(r)}

    def expected_measurement_value(self, ledger, posterior, group, objects, control, current):
        quantities=GROUPS[group]
        values=[]
        flatw=posterior['weights'].ravel()
        for _ in range(self.samples):
            imagined=ledger.copy()
            c,r=np.unravel_index(self.rng.choice(len(flatw),p=flatw),posterior['weights'].shape)
            cov=posterior['cov'][c,r];cov=(cov+cov.T)*.5
            eig,vec=np.linalg.eigh(cov)
            beta=posterior['means'][c,r]+vec@(np.sqrt(np.maximum(eig,0))*self.rng.standard_normal(2))
            for obj in objects:
                for quantity in quantities:
                    if ledger.seen[obj,quantity]:
                        # Repeated radiation narrows its public feasible interval.
                        imagined.variance[obj,quantity]*=.5
                        continue
                    x=float(self.rng.choice(self.marginal[quantity]))
                    if quantity==r and ledger.known[obj] and abs(beta[1])>.02:
                        inferred=((ledger.y[obj]-self.y_mean)/self.y_scale-beta[0])/beta[1]
                        x=inferred*self.x_scale[r]+self.x_mean[r]
                    imagined.x[obj,quantity]=x
                    imagined.variance[obj,quantity]=(.05**2/12 if quantity==2 else .01**2/12)
                    imagined.seen[obj,quantity]=True;imagined.counts[obj,quantity]+=1
            post=self.posterior(imagined)
            values.append(self.best_test(imagined,post,control,current)['probability'])
        return float(np.mean(values))

    def choose(self, ledger, control, current):
        posterior=self.posterior(ledger)
        test=self.best_test(ledger,posterior,control,current)
        best_score=test['probability']/self.cost[5]
        choice=(20+test['relation'],test['setting'],test['std'])
        diagnostics={'test':test,'measurement_plans':[]}
        for group,quantities in enumerate(GROUPS):
            needed=[int(obj) for obj in (*np.flatnonzero(ledger.known),ledger.query)
                    if not ledger.seen[obj,list(quantities)].all()]
            if not needed and group==2:
                # Repeat the largest sensitivity-weighted radiation uncertainty.
                contributing=[int(x) for x in (*np.flatnonzero(ledger.known),ledger.query)]
                needed=[max(contributing,key=lambda o:ledger.variance[o,2])]
                if ledger.counts[needed[0],2]>=12:
                    needed=[]
            if not needed:continue
            expected=self.expected_measurement_value(ledger,posterior,group,needed,control,current)
            score=max(0.,expected-test['probability'])/(len(needed)*self.cost[group])
            diagnostics['measurement_plans'].append({'group':group,'objects':needed,'expected_probability':expected,'score':score})
            if score>best_score:
                best_score=score;choice=(needed[0]*5+group,None,None)
        if best_score<=0:
            # All finite proposals have been excluded. Obtain the most uncertain
            # remaining measurement instead of repeating an executed setting.
            options=[(o*5+g) for o in (*np.flatnonzero(ledger.known),ledger.query)
                     for g,qs in enumerate(GROUPS) if not ledger.seen[o,list(qs)].all()]
            if options:choice=(int(options[0]),None,None)
        diagnostics['score']=float(best_score)
        return choice,diagnostics,posterior

    def observe_cost(self, action, cost):
        group=action%5 if action<20 else 5
        self.cost_counts[group]+=1
        self.cost[group]+=(cost-self.cost[group])/self.cost_counts[group]

    def remember(self, ledger):
        if self.mode!='forgetting':return
        p=self.posterior(ledger)
        weights=p['weights'].sum(0);weights/=weights.sum()
        conditional=p['weights']/p['weights'].sum(0,keepdims=True).clip(1e-30)
        mean=(conditional[...,None]*p['means']).sum(0)
        displacement=p['means']-mean
        cov=(conditional[...,None,None]*(p['cov']+displacement[..., :, None]*displacement[...,None,:])).sum(0)
        self.memories.append((mean,cov,np.log(weights.clip(1e-30))))
        self.memories=self.memories[-16:]
