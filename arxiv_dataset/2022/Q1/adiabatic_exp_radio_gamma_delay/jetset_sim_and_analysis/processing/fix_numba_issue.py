from jetset.model_manager import  FitModel
from jetset.jet_model import Jet

def load_model():
    try:
        fit_model=FitModel.load_model('processing/fit_model_lsb_mjd56302.pkl')
    except:
        print('fixing numba issue and regenerating model')
        jet=Jet(electron_distribution='lppl',name='jet_leptonic')
        jet.parameters.gmin.val=1.586653e+01
        jet.parameters.gmax.val=5.181316e+05
        jet.parameters.N.val=1.308777e+02
        jet.parameters.gamma0_log_parab.val=3.573558e+03
        jet.parameters.s.val=1.708731e+00
        jet.parameters.r.val=1.074723e+00
        jet.parameters.R.val=2.821653e+15
        jet.parameters.R_H.val=1.000000e+17
        jet.parameters.B.val=2.000000e-01
        jet.parameters.beam_obj.val=4.997505e+01
        jet.parameters.z_cosm.val=3.080000e-02
        jet.parameters.z_cosm.frozen=True
        fit_model=FitModel(jet=jet)
        fit_model.nu_max_fit=1E29
        fit_model.nu_min_fit=1E11
        fit_model.name='SSC-best-fit-lsb'
        fit_model.eval()
        fit_model.save_model('processing/fit_model_lsb_mjd56302.pkl')
    

    return fit_model