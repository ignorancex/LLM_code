import autofit as af
import autolens as al
from autofit.non_linear.grid import sensitivity as s
from . import slam_util

from typing import Union, Tuple, ClassVar, Optional
import numpy as np


def detection(
    settings_autofit: af.SettingsSearch,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    mass_results: af.ResultsCollection,
    subhalo_mass: af.Model = af.Model(al.mp.SphNFWMCRLudlow),
    grid_dimension_arcsec: float = 3.0,
    number_of_steps: Union[Tuple[int], int] = 5,
    end_with_stochastic_extension: bool = False
) -> af.ResultsCollection:
    """
    The SLaM SUBHALO PIPELINE for fitting imaging data with or without a lens light component, where it is assumed
    that the subhalo is at the same redshift as the lens galaxy.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    mass_results
        The results of the SLaM MASS PIPELINE which ran before this pipeline.
    subhalo_mass
        The `MassProfile` used to fit the subhalo in this pipeline.
    grid_dimension_arcsec
        the arc-second dimensions of the grid in the y and x directions. An input value of 3.0" means the grid in
        all four directions extends to 3.0" giving it dimensions 6.0" x 6.0".
    number_of_steps
        The 2D dimensions of the grid (e.g. number_of_steps x number_of_steps) that the subhalo search is performed for.
    number_of_cores
        The number of cores used to perform the non-linear search grid search. If 1, each model-fit on the grid is
        performed in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the SUBHALO PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using MASS PIPELINE's mass distribution [Priors initialized from MASS PIPELINE].
     - The source galaxy's light is parametric or an inversion depending on the previous MASS PIPELINE [Model and 
     priors initialized from MASS PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the MASS PIPELINE. This model will be used to perform Bayesian model comparison with models that include a 
    subhalo, to determine if a subhalo is detected.
    """

    source = slam_util.source__from_result_model_if_parametric(
        result=mass_results.last, setup_hyper=setup_hyper
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=mass_results.last.model.galaxies.lens, source=source
        ),
        hyper_image_sky=setup_hyper.hyper_image_sky_from(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from(
            result=mass_results.last
        ),
    )

    search_no_subhalo = af.DynestyStatic(
        name="subhalo[1]_mass[total_refine]", **settings_autofit.search_dict, nlive=100
    )

    result_1 = search_no_subhalo.fit(
        model=model, analysis=analysis, **settings_autofit.fit_dict
    )

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    In search 2 of the SUBHALO PIPELINE we perform a [number_of_steps x number_of_steps] grid search of non-linear
    searches where:

     - The lens galaxy mass is modeled using MASS PIPELINE's mass distribution [Priors initialized from MASS PIPELINE].
     - The source galaxy's light is parametric or an inversion depending on the previous MASS PIPELINE [Model and 
     priors initialized from MASS PIPELINE].
     - The subhalo redshift is fixed to that of the lens galaxy.
     - Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
     - The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    This search aims to detect a dark matter subhalo.
    """

    subhalo = af.Model(
        al.Galaxy, redshift=result_1.instance.galaxies.lens.redshift, mass=subhalo_mass
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    subhalo.mass.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )

    subhalo.mass.redshift_object = result_1.instance.galaxies.lens.redshift
    subhalo.mass.redshift_source = result_1.instance.galaxies.source.redshift

    source = slam_util.source__from_result_model_if_parametric(
        result=mass_results.last, setup_hyper=setup_hyper
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=mass_results.last.model.galaxies.lens, subhalo=subhalo, source=source
        ),
        hyper_image_sky=setup_hyper.hyper_image_sky_from(
            result=mass_results.last, as_model=True
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from(
            result=mass_results.last
        ),
    )

    search = af.DynestyStatic(
        name="subhalo[2]_mass[total]_source_subhalo[search_lens_plane]",
        unique_tag=settings_autofit.unique_tag,
        # number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=50,
        walks=5,
        facc=0.2,
    )

    subhalo_grid_search = af.SearchGridSearch(
        search=search, number_of_steps=number_of_steps, number_of_cores=70
    )

    subhalo_result = subhalo_grid_search.fit(
        model=model,
        analysis=analysis,
        grid_priors=[
            model.galaxies.subhalo.mass.centre_0,
            model.galaxies.subhalo.mass.centre_1,
        ],
        info=settings_autofit.info,
        parent=search_no_subhalo,
    )
    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    In search 3 of the SUBHALO PIPELINE we refit the lens and source models above but now including a subhalo, where 
    the subhalo model is initalized from the highest evidence model of the subhalo grid search.

     - The lens galaxy mass is modeled using MASS PIPELINE's mass distribution [Priors initialized from MASS PIPELINE].
     - The source galaxy's light is parametric or an inversion depending on the previous MASS PIPELINE [Model and 
     priors initialized from MASS PIPELINE].
     - The subhalo redshift is fixed to that of the lens galaxy.
     - Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
     - The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    This search aims to refine the parameter estimates and errors of a dark matter subhalo detected in the grid search
    above.
    """

    subhalo = af.Model(
        al.Galaxy, redshift=result_1.instance.galaxies.lens.redshift, mass=subhalo_mass
    )

    subhalo.mass.mass_at_200 = subhalo_result.model.galaxies.subhalo.mass.mass_at_200
    subhalo.mass.centre = subhalo_result.model.galaxies.subhalo.mass.centre

    subhalo.mass.redshift_object = subhalo_result.instance.galaxies.lens.redshift
    subhalo.mass.redshift_source = subhalo_result.instance.galaxies.source.redshift

    model = af.Collection(
        galaxies=af.Collection(
            lens=subhalo_result.model.galaxies.lens,
            subhalo=subhalo,
            source=subhalo_result.model.galaxies.source,
        ),
        hyper_image_sky=setup_hyper.hyper_image_sky_from(
            result=subhalo_result.last, as_model=True
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from(
            result=subhalo_result.last
        ),
    )

    search = af.DynestyStatic(
        name="subhalo[3]_subhalo[single_plane_refine]",
        **settings_autofit.search_dict,
        nlive=100,
    )

    result_3 = search.fit(model=model, analysis=analysis, **settings_autofit.fit_dict)

    if end_with_stochastic_extension:

        extensions.stochastic_fit(
            result=result_3, analysis=analysis, search_previous=search, **settings_autofit.fit_dict
        )

    return af.ResultsCollection([result_1, subhalo_result, result_3])
