import matplotlib.pyplot as plt
import astropy.units as u
import eispac

if __name__ == '__main__':
    # input data and template files
    data_filepath = '../../src/EIS/level1/EastOffLimbFullSpectra/eis_20170821_205401.data.h5'
    fe_10_184_template_filepath = '../../sav/EIS/fit_template/fe_10_184_536.1c.template.h5'
    fe_10_257_template_filepath = '../../sav/EIS/fit_template/fe_10_257_262.4c.template.h5'
    fe_11_182_template_filepath = '../../sav/EIS/fit_template/fe_11_188_216.2c.template.h5'
    fe_11_188_template_filepath = '../../sav/EIS/fit_template/fe_11_188_216.2c.template.h5'
    fe_11_257_template_filepath = '../../sav/EIS/fit_template/fe_11_257_547.4c.template.h5'
    fe_12_186_template_filepath = '../../sav/EIS/fit_template/fe_12_186_880.1c.template.h5'
    fe_12_192_template_filepath = '../../sav/EIS/fit_template/fe_12_192_394.1c.template.h5'
    fe_12_195_template_filepath = '../../sav/EIS/fit_template/fe_12_195_119.1c.template.h5'
    fe_14_264_template_filepath = '../../sav/EIS/fit_template/fe_14_264_787.1c.template.h5'
    si_10_258_template_filepath = '../../sav/EIS/fit_template/si_10_258_375.1c.template.h5'
    si_10_261_template_filepath = '../../sav/EIS/fit_template/si_10_261_058.1c.template.h5'

    fit_res = []

    fe_10_184_tmplt = eispac.read_template(fe_10_184_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_10_184_tmplt.central_wave)
    fe_10_184_fit_res = eispac.fit_spectra(data_cube, fe_10_184_tmplt, ncpu='max')
    fit_res.append(fe_10_184_fit_res)

    fe_10_257_tmplt = eispac.read_template(fe_10_257_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_10_257_tmplt.central_wave)
    fe_10_257_fit_res = eispac.fit_spectra(data_cube, fe_10_257_tmplt, ncpu='max')
    fit_res.append(fe_10_257_fit_res)

    fe_11_182_tmplt = eispac.read_template(fe_11_182_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_11_182_tmplt.central_wave)
    fe_11_182_fit_res = eispac.fit_spectra(data_cube, fe_11_182_tmplt, ncpu='max')
    fit_res.append(fe_11_182_fit_res)

    fe_11_188_tmplt = eispac.read_template(fe_11_188_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_11_188_tmplt.central_wave)
    fe_11_188_fit_res = eispac.fit_spectra(data_cube, fe_11_188_tmplt, ncpu='max')
    fit_res.append(fe_11_188_fit_res)

    # fe_11_257_tmplt = eispac.read_template(fe_11_257_template_filepath)
    # data_cube = eispac.read_cube(data_filepath, fe_11_257_tmplt.central_wave)
    # fe_11_257_fit_res = eispac.fit_spectra(data_cube, fe_11_257_tmplt, ncpu='max')
    # fit_res.append(fe_11_257_fit_res)

    fe_12_186_tmplt = eispac.read_template(fe_12_186_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_12_186_tmplt.central_wave)
    fe_12_186_fit_res = eispac.fit_spectra(data_cube, fe_12_186_tmplt, ncpu='max')
    fit_res.append(fe_12_186_fit_res)

    fe_12_192_tmplt = eispac.read_template(fe_12_192_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_12_192_tmplt.central_wave)
    fe_12_192_fit_res = eispac.fit_spectra(data_cube, fe_12_192_tmplt, ncpu='max')
    fit_res.append(fe_12_192_fit_res)

    fe_12_195_tmplt = eispac.read_template(fe_12_195_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_12_195_tmplt.central_wave)
    fe_12_195_fit_res = eispac.fit_spectra(data_cube, fe_12_195_tmplt, ncpu='max')
    fit_res.append(fe_12_195_fit_res)

    fe_14_264_tmplt = eispac.read_template(fe_14_264_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_14_264_tmplt.central_wave)
    fe_14_264_fit_res = eispac.fit_spectra(data_cube, fe_14_264_tmplt, ncpu='max')
    fit_res.append(fe_14_264_fit_res)

    si_10_258_tmplt = eispac.read_template(si_10_258_template_filepath)
    data_cube = eispac.read_cube(data_filepath, si_10_258_tmplt.central_wave)
    si_10_258_fit_res = eispac.fit_spectra(data_cube, si_10_258_tmplt, ncpu='max')
    fit_res.append(si_10_258_fit_res)

    si_10_261_tmplt = eispac.read_template(si_10_261_template_filepath)
    data_cube = eispac.read_cube(data_filepath, si_10_261_tmplt.central_wave)
    si_10_261_fit_res = eispac.fit_spectra(data_cube, si_10_261_tmplt, ncpu='max')
    fit_res.append(si_10_261_fit_res)

    for fit_res_ in fit_res:
        eispac.save_fit(fit_res_, save_dir='../../sav/EIS/EQSPY/eispac_fit/')
