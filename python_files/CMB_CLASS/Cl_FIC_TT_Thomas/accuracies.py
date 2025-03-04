########### Define CLASS Accuracy Settings for Mass Power Spectra ###############

# High Accuracy Settings (i.e. baseline for comparison)
high_acc = {
    'l_linstep': 2,
    'l_logstep': 1.026,
    'q_linstep': 0.5,
    'q_logstep_spline': 2.4,
    'l_switch_limber' : 2500.0,
    'l_switch_limber_for_nc_local_over_z': 2500.0,
    'l_switch_limber_for_nc_los_over_z': 2500.0,
    'write warnings' : 'yes'}

# Very High Accuracy Settings (As high as found we could push it)
v_high_acc = {
    'l_linstep': 2,
    'l_logstep': 1.007,
    'q_linstep': 0.19,
    'q_logstep_spline': 2.0,
    'l_switch_limber' : 2500.0,
    'l_switch_limber_for_nc_local_over_z': 2500.0,
    'l_switch_limber_for_nc_los_over_z': 2500.0,
    'write warnings' : 'yes',
    'input_verbose' : 1,
    'background_verbose' : 1,
    'thermodynamics_verbose' : 1,
    'perturbations_verbose' : 1,
    'transfer_verbose' : 1,
    'primordial_verbose' : 1,
    'spectra_verbose' : 1,
    'nonlinear_verbose' : 1,
    'lensing_verbose' : 1,
    'output_verbose' : 1
    }
    
"""v_high_acc = {
    'l_linstep': 2,
    'l_logstep': 1.007,
    'q_linstep': 0.19,
    'q_logstep_spline': 2.0,
    'l_switch_limber' : 2500.0,
    'l_switch_limber_for_nc_local_over_z': 2500.0,
    'l_switch_limber_for_nc_los_over_z': 2500.0,
    'k_step_trans':0.08,
    'write warnings' : 'yes',
    'input_verbose' : 1,
    'background_verbose' : 1,
    'thermodynamics_verbose' : 1,
    'perturbations_verbose' : 1,
    'transfer_verbose' : 1,
    'primordial_verbose' : 1,
    'spectra_verbose' : 1,
    'nonlinear_verbose' : 1,
    'lensing_verbose' : 1,
    'output_verbose' : 1
    }"""