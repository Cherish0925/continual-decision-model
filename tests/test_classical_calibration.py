import numpy as np
from scipy.special import ndtr
from experiments.classical_calibration.model import BayesianDesigner, Ledger
from experiments.discoveryworld.calibration_mixture import AffineMixture


def fixture():
    prior=AffineMixture({'x_mean':[0.]*9,'x_scale':[1.]*9,'y_mean_hz':100.,'y_scale_hz':10.},prior_mode='shared')
    designer=BayesianDesigner(prior,[np.array([-1.,0.,1.]) for _ in range(9)])
    ledger=Ledger(np.zeros((4,9)),np.zeros((4,9)),np.ones((4,9),bool),
                  np.array([90.,110.,0.,0.]),np.array([True,True,False,False]),np.ones((4,9),int),2,[])
    post={'means':np.zeros((1,9,2)),'cov':np.tile(np.diag([.25,0.]),(1,9,1,1)),
          'weights':np.ones((1,9))/9}
    return designer,ledger,post


def test_success_interval_is_applied_to_latent_resonance_once():
    designer,ledger,post=fixture()
    mean,std,*_=designer.predict(ledger,post)
    np.testing.assert_allclose(mean,100.)
    np.testing.assert_allclose(std,5.)
    test=designer.best_test(ledger,post,{'lower_hz':0.,'upper_hz':200.,'minimum_step_hz':1.},100.)
    np.testing.assert_allclose(test['probability'],ndtr(.4)-ndtr(-.4))


def test_missing_query_quantity_retains_its_marginal_uncertainty():
    designer,ledger,post=fixture()
    ledger.seen[2]=False;ledger.variance[2]=1.
    post['means'][...,1]=2.;post['cov'][...,0,0]=0.
    _,std,*_=designer.predict(ledger,post)
    np.testing.assert_allclose(std,20.)


def test_received_failure_conditions_probability_without_banning_relation():
    designer,ledger,post=fixture();ledger.failures=[100.]
    test=designer.best_test(ledger,post,{'lower_hz':0.,'upper_hz':200.,'minimum_step_hz':1.},100.)
    assert test['setting'] in (96.,104.)
    expected=(ndtr(1.2)-ndtr(.4))/(1-(ndtr(.4)-ndtr(-.4)))
    np.testing.assert_allclose(test['probability'],expected)
