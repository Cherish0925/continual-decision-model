"""Act using learned response functions or conditional-prediction references."""
import numpy as np
import torch

from cdm.relation_belief import ConditionalRelation, RelationBelief, RelationBeliefConfig
from .recipes import primitive_recipes


def normalized(log_mass):
    result = np.exp(log_mass - np.max(log_mass))
    return result / result.sum()


def completion_cost(functions, weights, costs, valid, remaining):
    """Expected cost of a finite, non-repeating completion order.

    The order is fixed from the current success marginals. Survival along that
    order retains the shared latent law. A failed task adds 200 operation units,
    matching 2*completion - operations/100 in the environment's return.
    """
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return float(remaining + 200)
    success = functions[:, candidates, 0]
    order = np.argsort(-(weights @ success) / costs[candidates], kind='stable')
    success = success[:, order]
    trial_costs = costs[candidates[order]]
    used = np.cumsum(trial_costs)
    prior_used = used - trial_costs
    charged = np.clip(remaining - prior_used, 0, trial_costs)
    survived = np.cumprod(1-success, axis=-1)
    prior_survival = np.concatenate((np.ones((len(weights),1)), survived[:,:-1]), -1)
    first_success = prior_survival * success * (used <= remaining)[None]
    exit_cost = np.clip(remaining-used, 0, 4)
    expected = weights @ ((prior_survival * charged).sum(-1) + (first_success * exit_cost).sum(-1))
    completed = weights @ (first_success * (used+4 <= remaining)[None]).sum(-1)
    return float(expected + 200*(1-completed))


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = (RelationBelief(RelationBeliefConfig(**saved['model_config'])) if saved['kind']=='functions'
             else ConditionalRelation(**saved['model_config']))
    model.load_state_dict(saved['model'])
    return model.eval(), saved


class LearnedExperiments:
    def __init__(self, model, n, capacity=12, reuse=.45, selection='information', shortlist=12, budget=1000):
        self.model, self.n = model, n
        self.capacity, self.reuse = capacity, reuse
        self.selection, self.shortlist = selection, shortlist
        self.budget = budget
        self.recipes = np.asarray(primitive_recipes(n))
        counts = torch.zeros(len(self.recipes),8)
        counts[:,:n] = torch.as_tensor(self.recipes)
        self.queries = counts / counts.sum(-1,keepdim=True)
        self.components = torch.arange(8)<n
        self.archives, self.history = [], []
        self.is_functions = isinstance(model,RelationBelief)
        with torch.no_grad():
            if self.is_functions:
                self.functions = model(self.queries,self.components).exp().double().numpy()
                self.fresh = model.log_prior(self.components).exp().double().numpy()
        self.start()

    def _function_posterior(self, history):
        logp=np.log(self.fresh)
        for action,before,after in history:
            p=self.functions[:,action,after] if after<before else self.functions[:,action,before:].sum(-1)
            logp=logp+np.log(p.clip(1e-30))
        return normalized(logp)

    def start(self):
        if self.history and self.capacity:
            self.archives.append(self.history[-32:].copy())
            self.archives=self.archives[-self.capacity:]
        self.history=[]
        if self.is_functions:
            self.posterior=self.fresh.copy()
            if self.archives:
                old=np.mean([self._function_posterior(h) for h in self.archives],axis=0)
                self.posterior=(1-self.reuse)*self.fresh+self.reuse*old
            self.posterior/=self.posterior.sum()
        else:
            self.old = self._conditional(self.archives) if self.archives else np.empty((0,len(self.queries),4))
            self.posterior=np.array([1.] if not self.archives else
                                    [1-self.reuse]+[self.reuse/len(self.archives)]*len(self.archives))
        self.returns=0
        self.forecasts=None

    @torch.no_grad()
    def _conditional(self,histories):
        b=len(histories); length=max(1,max(map(len,histories)))
        support=torch.zeros(b,length,8)
        before=torch.zeros(b,length,dtype=torch.long)
        after=torch.zeros_like(before)
        valid=torch.zeros_like(before,dtype=torch.bool)
        for i,h in enumerate(histories):
            for j,(a,pre,post) in enumerate(h):
                support[i,j]=self.queries[a]; before[i,j]=pre; after[i,j]=post; valid[i,j]=True
        logits=self.model(self.queries[None].expand(b,-1,-1),self.components[None].expand(b,-1),
                          support,before,after,valid)
        return logits.softmax(-1).double().numpy()

    def all_functions(self):
        if self.is_functions: return self.functions
        if self.forecasts is None:
            self.forecasts=np.concatenate((self._conditional([self.history]),self.old),axis=0)
        return self.forecasts

    @staticmethod
    def observe(functions,current):
        result=functions.copy()
        result[...,current]=result[...,current:].sum(-1)
        result[...,current+1:]=0
        return result

    def probabilities(self,current):
        return np.einsum('k,kal->al',self.posterior,self.observe(self.all_functions(),current))

    def receive(self,action,before,after):
        likelihood=self.observe(self.all_functions(),before)[:,action,after]
        self.posterior=normalized(np.log(self.posterior.clip(1e-300))+np.log(likelihood.clip(1e-30)))
        self.history.append((int(action),int(before),int(after)))
        self.returns+=1
        self.forecasts=None

    def choose(self,current,valid):
        functions=self.all_functions()
        observed=self.observe(functions,current)
        p=np.einsum('k,kal->al',self.posterior,observed)
        entropy=-(p*np.log(p.clip(1e-30))).sum(-1)
        within=-(observed*np.log(observed.clip(1e-30))).sum(-1)
        info=np.maximum(entropy-self.posterior@within,0)
        cost=6+self.recipes.sum(-1)+(self.recipes>0).sum(-1)+2*(self.returns>0)
        direct=np.where(valid,p[:,0]/cost,-np.inf)
        if self.selection=='greedy': return int(direct.argmax())
        if self.selection=='information':
            return int(np.where(valid,(p[:,0]+.5*info)/cost,-np.inf).argmax())
        candidates=[]
        for ranking in (np.argsort(-direct),np.argsort(-np.where(valid,entropy/cost,-np.inf))):
            added=0
            for a in ranking:
                if valid[a] and a not in candidates:
                    candidates.append(int(a)); added+=1
                if added>=self.shortlist//2: break
        finite = self.selection in ('finite_plan','finite_marginal')
        update = self.selection not in ('marginal_plan','finite_marginal')
        # Public preparation and already executed experiment costs.
        spent = 14 + sum(6 + self.recipes[a].sum() + (self.recipes[a]>0).sum() + 2*(i>0)
                         for i,(a,_,_) in enumerate(self.history))
        remaining = max(0,self.budget-spent)
        future_fresh={}
        if not self.is_functions and update:
            cases=[(a,y) for a in candidates for y in range(1,current+1)]
            predicted=self._conditional([self.history+[(a,current,y)] for a,y in cases])
            future_fresh=dict(zip(cases,predicted))
        values=[]
        for a in candidates:
            if finite and cost[a]>remaining:
                values.append(float(remaining+200)); continue
            total=float(cost[a]+p[a,0]*(min(4,remaining-cost[a]) if finite else 4))
            if finite and cost[a]+4>remaining:
                total += 200*p[a,0]
            for y in range(1,current+1):
                weights=self.posterior
                future=functions
                if update:
                    weights=normalized(np.log(weights.clip(1e-300))+np.log(observed[:,a,y].clip(1e-30)))
                    if not self.is_functions:
                        future=np.concatenate((future_fresh[(a,y)][None],self.old),axis=0)
                success=weights@future[:,:,0]
                next_cost=8+self.recipes.sum(-1)+(self.recipes>0).sum(-1)
                feasible=valid.copy(); feasible[a]=False
                if finite:
                    continuation=completion_cost(future,weights,next_cost,feasible,remaining-cost[a])
                    total+=p[a,y]*continuation
                else:
                    continuation=np.where(feasible,next_cost/success.clip(1e-8),np.inf).min()
                    total+=p[a,y]*(min(float(continuation),1000.)+4)
            values.append(total)
        return candidates[int(np.argmin(values))]
