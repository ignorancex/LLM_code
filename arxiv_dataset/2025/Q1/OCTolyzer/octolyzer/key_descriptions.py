# Description of all the columns found in the metadata sheet
meta_cols = {
    'Filename':'Filename of the SLO+OCT file analyse.',
    'FAILED':'Boolean flag on whether file unexpectedly failed to be analysed.',
    'eye':'Type of eye, either Right or Left.',
    
    'bscan_type':'Type of OCT scan acquired. One of H(orizontal)-line, V(ertical)-line;A(rtery)V(ein)-line, P(osterior)pole and Peripapillary.',
    'bscan_resolution_x': 'Number of columns of B-scan, typically 768 or 1536 for Heidelberg.',
    'bscan_resolution_y': 'Number of rows of B-scan, typically 768 or 496 for Heidelberg.',
    'bscan_scale_z': 'Micron distance between successive B-scans in a Posterior pole acquisition. Is 0 for all other Bscan_types.',
    'bscan_scale_x': 'Pixel lengthscale in the horizontal direction B-scan/SLO, measured in microns per pixel.',
    'bscan_scale_y': 'Pixel lengthscale in the vertical direction in the B-scan, measured in microns per pixel.',
    'scale_units': 'Units of the lengthscales, this is fixed as microns per pixel.',
    
    'avg_quality': 'Heidelberg-provided signal-to-noise ratio of the B-scan(s).',
    'retinal_layers_N': 'Number of retinal layer segmentations extracted from metadata.',
    'scan_focus': 'Scan focus of the acquisition, in Dioptres. This decides the scaling and is a gross measure of refractive error.',
    'visit_date': 'Date of acquisition.',
    'exam_time': 'Time of acquisition.',
    
    'slo_resolution_px': 'Number of rows/columns in the square-shaped SLO image (typically 768 or 1536).',
    'field_of_view_mm': 'Field of view captured during acquisition, usually between 8 and 9 mm if field size is 30 degrees.',
    'slo_scale_xy': 'Pixel lengthscale of the SLO image, and is typically the same for both directions.',
    'location': 'Whether scan is macula-centred or disc-centred. Is either "macular" or "peripapillary"',
    'field_size_degrees': 'Field of view in degrees, typically 30.',
    'slo_modality': ' Modality used for SLO image capture. OCTolyzer supports grayscale NIR cSLO images currently.',
    
    'bscan_fovea_x': 'Horizontal pixel position of the fovea on the OCT B-scan (if visible in one of the scans, only relevant for macular OCT).',
    'bscan_fovea_y': 'Vertical pixel position of the fovea on the OCT B-scan (if visible in one of the scans, only relevant for macular OCT).',
    'bscan_missing_fovea':'Boolean value flagging whether fovea is missing from OCT data (either due to acquisition or segmentation failure).',
    'slo_fovea_x': 'Horizontal pixel position of the fovea on the SLO image, if visible.',
    'slo_fovea_y': 'Vertical pixel position of the fovea on the SLO image, if visible.',
    'acquisition_angle_degrees': 'Angle of elevation from horizontal image axis of acquisition for Posterior pole scans.',
    'slo_missing_fovea':'Boolean value flagging whether fovea is missing from SLO data (either due to acquisition or segmentation failure).',
    
    'optic_disc_overlap_index_%':'% of the optic disc diameter, defining how off-centre a peripapillary image acquisition is from the optic disc centre.',
    'optic_disc_overlap_warning': 'Boolean value, flagging if the overlap index is greater than 15%, the empirical cut-off to warn end-user of an off-centre scan.',
    'optic_disc_x': 'Horizontal pixel position of the optic disc centre on the SLO image, if visible.',
    'optic_disc_y': 'Vertical pixel position of the optic disc centre on the SLO image, if visible.',
    'optic_disc_radius_px': 'Pixel radius of the optic disc.',
    
    'thickness_units':'Units of measurement for thickness, always in microns (micrometres).',
    'vascular_index_units':'Units of measurement for choroid vascular index, always dimensionless (no units, but is a ratio between 0 and 1).',
    'vessel_density_units':'Units of measurement for choroid vessel density, always in micron2 (square microns)',
    'area_units':'Units of measurements for area, always in mm2 (square millimetres).',
    'volume_units':'Units of measurements for volume, always in mm3 (cubic millimetres).',
    'linescan_area_ROI_microns':'For single-line, macular OCT, this is the micron distance defining the fovea-centred region of interest.',
    'choroid_measure_type':'Whether the choroid is measured column-wise (per A-scan) or perpendicularly. Always per A-scan for peripapillary OCT.',
    'missing_retinal_oct_linescan_measurements':'Whether OCT retinal measurements could not be computed for H-/V-linescans due to too large an ROI or too short a segmentation.',
    'missing_choroid_oct_linescan_measurements':'Whether OCT choroidal measurements could not be computed for H-/V-linescans due to too large an ROI or too short a segmentation.',

    'acquisition_radius_px': 'Pixel radius of the acquisition line around the optic disc for peripapillary OCT.',
    'acquisition_radius_mm': 'Millimetre radius of the acquisition line around the optic disc for peripapillary OCT.',
    'acquisition_optic_disc_center_x': 'Horizontal pixel position of the optic disc centre, as selected by the user during peripapillary OCT acquisition.',
    'acquisition_optic_disc_center_y': 'Vertical pixel position of the optic disc centre, as selected by the user during peripapillary OCT acquisition.'
}


lsr_cols = {
    'Filename':'Name of the file for easy look-up with metadata sheet.',
    'retina_area_[mm2]_i':"Retinal area for the i'th B-scan covering pre-specified fovea-centred region of interest (in square millimetres).",
    'retina_subfoveal_thickness_[um]_i':"Retinal thickness underneath the fovea for the i'th B-scan covering pre-specified fovea-centred region of interest (in microns).",
    'retina_thickness_[um]_i':"Average retinal thickness for the i'th B-scan (in microns).",

    'outer_retina_area_[mm2]_i':"Outer retinal area (from external limiting membrane (ELM) to Bruch's membrane (BM)) for the i'th B-scan covering pre-specified fovea-centred region of interest (in square millimetres).",
    'outer_retina_subfoveal_thickness_[um]_i':"Outer retinal thickness (from external limiting membrane (ELM) to Bruch's membrane (BM)) underneath the fovea for the i'th B-scan covering pre-specified fovea-centred region of interest (in microns).",
    'outer_retina_thickness_[um]_i':"Average outer retinal thickness (from external limiting membrane (ELM) to Bruch's membrane (BM)) for the i'th B-scan (in microns).",
    
    'inner_retina_area_[mm2]_i':"Inner retinal area (from inner limiting membrane (ILM) to external limiting membrane (ELM)) for the i'th B-scan covering pre-specified fovea-centred region of interest (in square millimetres).",
    'inner_retina_subfoveal_thickness_[um]_i':"Inner retinal thickness (from inner limiting membrane (ILM) to external limiting membrane (ELM)) underneath the fovea for the i'th B-scan covering pre-specified fovea-centred region of interest (in microns).",
    'inner_retina_thickness_[um]_i':"Average outer retinal thickness (from inner limiting membrane (ILM) to external limiting membrane (ELM)) for the i'th B-scan (in microns).",
    
    "{Layer1}_{Layer2}_area_[mm2]_i":"Area of the {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) layer (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) for the i'th B-scan covering pre-specified fovea-centred region of interest (in square millimetres).",
    "{Layer1}_{Layer2}_subfoveal_thickness_[um]_i":"Thickness underneath the fovea of the {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) layer (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) for the i'th B-scan covering pre-specified fovea-centred region of interest (in microns).",
    "{Layer1}_{Layer2}_thickness_[um]_i":"Area of the {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) layer (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) for the i'th B-scan covering pre-specified fovea-centred region of interest (in microns).",
    
    'CHORupper_CHORlower_area_[mm2]_i':"Choroid area for the i'th B-scan covering pre-specified fovea-centred region of interest (in square millimetres).",
    'CHORupper_CHORlower_subfoveal_thickness_[um]_i':"Choroid thickness underneath the fovea for the i'th B-scan (in microns).",
    'CHORupper_CHORlower_thickness_[um]_i':"Average choroid thickness for the i'th B-scan covering pre-specified fovea-centred region of interest (in microns).",
    'CHORupper_CHORlower_vascular_index_i':"Choroid vascularity index (CVI) for the i'th B-scan covering pre-specified fovea-centred region of interest (dimensionless).",
    'CHORupper_CHORlower_vessel_area_[mm2]_i':"Choroid vessel area for the i'th B-scan covering pre-specified fovea-centred region of interest (in square millimetres)."
}


slo_cols = {
    'Filename':'Name of the file for easy look-up with metadata sheet.',	
    'fractal_dimension_binary_whole':'Fractal dimension across all-vessels (arteries and veins) for the whole SLO image (dimensionless).',	
    'vessel_density_binary_whole':'Vessel density dimension across all-vessels (arteries and veins) for the whole SLO image (dimensionless).',	
    'tortuosity_density_binary_whole':'Tortuosity density across all-vessels (arteries and veins) for the whole SLO image (dimensionless).',	
    'average_global_calibre_binary_whole':'Vessel width calculated globally across all-vessels (arteries and veins) for the whole SLO image (in microns or pixels).',	
    'average_local_calibre_binary_whole':'Vessel width averaged over individual vessel segments across all-vessels (arteries and veins) for the whole SLO image (in microns or pixels).',

    'fractal_dimension_artery_whole':'Fractal dimension across arteries for the whole SLO image (dimensionless).',	
    'vessel_density_artery_whole':'Vessel density dimension across arteries for the whole SLO image (dimensionless).',	
    'tortuosity_density_artery_whole':'Tortuosity density across arteries for the whole SLO image (dimensionless).',	
    'average_global_calibre_artery_whole':'Vessel width calculated globally across arteries for the whole SLO image (in microns or pixels).',	
    'average_local_calibre_artery_whole':'Vessel width averaged over individual vessel segments across arteries for the whole SLO image (in microns or pixels).',
    'CRAE_Knudtson_artery_whole':'Central Retinal Artery Equivalent (large arterial vessel thickness computed using the Knudston approach) for the whole SLO image (in microns or pixels)',

    'fractal_dimension_vein_whole':'Fractal dimension across veins for the whole SLO image (dimensionless).',	
    'vessel_density_vein_whole':'Vessel density dimension across veins for the whole SLO image (dimensionless).',	
    'tortuosity_density_vein_whole':'Tortuosity density across veins for the whole SLO image (dimensionless).',	
    'average_global_calibre_vein_whole':'Vessel width calculated globally across veins for the whole SLO image (in microns or pixels).',	
    'average_local_calibre_vein_whole':'Vessel width averaged over individual vessel segments across veins for the whole SLO image (in microns or pixels).',	
    'CRVE_Knudtson_vein_whole':'Central Retinal Vein Equivalent (large vein vessel thickness computed using the Knudston approach) for the whole SLO image (in microns or pixels)',
    'AVR_whole':'Arterioveule ratio (CRAE divided by CRVE) for the whole SLO image (dimensionless).',


    'tortuosity_density_binary_C':'Tortuosity density across all-vessels (arteries and veins) for zone C around the optic disc (0D-2D, D is optic disc diameter) (dimensionless).',	
    'average_local_calibre_binary_C':'Vessel width averaged over individual vessel segments across all-vessels (arteries an veins) for zone C around the optic disc (0D-2D, D is optic disc diameter) (in microns or pixels).',

    'tortuosity_density_artery_C':'Tortuosity density across arteries for Zone C around the optic disc (0D-2D, D is optic disc diameter) (dimensionless).',	
    'average_local_calibre_artery_C':'Vessel width averaged over individual vessel segments across arteries for zone C around the optic disc (0D-2D, D is optic disc diameter) (in microns or pixels).',
    'CRAE_Knudtson_artery_C':'Central Retinal Artery Equivalent (large arterial vessel thickness computed using the Knudston approach) zone C around the optic disc (0D-2D, D is optic disc diameter) (in microns or pixels)',	

    'tortuosity_density_vein_C':'Tortuosity density across arteries for Zone C around the optic disc (0D-2D, D is optic disc diameter) (dimensionless).',	
    'average_local_calibre_vein_C':'Vessel width averaged over individual vessel segments across veins for zone C around the optic disc (0D-2D, D is optic disc diameter) (in microns or pixels).',
    'CRVE_Knudtson_vein_C':'Central Retinal Vein Equivalent (large vein vessel thickness computed using the Knudston approach) for zone C around the optic disc (0D-2D, D is optic disc diameter) (in microns or pixels)',	
    'AVR_C':'Arterioveule ratio (CRAE divided by CRVE) for zone C around the optic disc (0D-2D, D is optic disc diameter) (dimensionless).',	


    'tortuosity_density_binary_B':'Tortuosity density across all-vessels (arteries and veins) for Zone B around the optic disc (0.5D-1D, D is optic disc diameter) (dimensionless).',		
    'average_local_calibre_binary_B':'Vessel width averaged over individual vessel segments across all-vessels (arteries an veins) for zone B around the optic disc (0.5D-1D, D is optic disc diameter) (in microns or pixels).',

    'tortuosity_density_artery_B':'Tortuosity density across arteries for Zone B around the optic disc (0D.5-1D, D is optic disc diameter) (dimensionless).',	
    'average_local_calibre_artery_B':'Vessel width averaged over individual vessel segments across arteries for zone B around the optic disc (0.5D-1D, D is optic disc diameter) (in microns or pixels).',
    'CRAE_Knudtson_artery_B':'Central Retinal Artery Equivalent (large arterial vessel thickness computed using the Knudston approach) zone B around the optic disc (0.5D-1D, D is optic disc diameter) (in microns or pixels)',	

    'tortuosity_density_vein_B':'Tortuosity density across veins for Zone B around the optic disc (0.5D-1D, D is optic disc diameter) (dimensionless).',	
    'average_local_calibre_vein_B':'Vessel width averaged over individual vessel segments across veins for zone B around the optic disc (0.5D-1D, D is optic disc diameter) (in microns or pixels).',	
    'CRVE_Knudtson_vein_B':'Central Retinal Vein Equivalent (large vein vessel thickness computed using the Knudston approach) for zone C around the optic disc (0D-2D, D is optic disc diameter) (in microns or pixels)',	
    'AVR_B':'Arterioveule ratio (CRAE divided by CRVE) for zone B around the optic disc (0.5D-1D, D is optic disc diameter) (dimensionless).'
}

peri_cols = {
    'Filename': 'Name of the file for easy look-up with metadata sheet.',
    
    'retina_All_[um]': 'Average retinal thickness across the entire peripapillary region (in microns).',
    'retina_N/T': 'Ratio of nasal to temporal retinal thickness in the peripapillary region (dimensionless).',
    'retina_PMB_[um]': 'Retinal thickness in the papillomacular bundle (PMB) region (in microns).',
    'retina_infero_nasal_[um]': 'Retinal thickness in the inferonasal quadrant of the peripapillary region (in microns).',
    'retina_infero_temporal_[um]': 'Retinal thickness in the inferotemporal quadrant of the peripapillary region (in microns).',
    'retina_nasal_[um]': 'Retinal thickness in the nasal quadrant of the peripapillary region (in microns).',
    'retina_supero_nasal_[um]': 'Retinal thickness in the superonasal quadrant of the peripapillary region (in microns).',
    'retina_supero_temporal_[um]': 'Retinal thickness in the superotemporal quadrant of the peripapillary region (in microns).',
    'retina_temporal_[um]': 'Retinal thickness in the temporal quadrant of the peripapillary region (in microns).',
    
    'ILM_RNFL_All_[um]': 'Average retinal nerve fiber layer (RNFL) thickness from the inner limiting membrane (ILM) across the entire peripapillary region (in microns).',
    'ILM_RNFL_N/T': 'Ratio of nasal to temporal RNFL thickness in the peripapillary region (dimensionless).',
    'ILM_RNFL_PMB_[um]': 'RNFL thickness in the papillomacular bundle (PMB) region (in microns).',
    'ILM_RNFL_infero_nasal_[um]': 'RNFL thickness in the inferonasal quadrant of the peripapillary region (in microns).',
    'ILM_RNFL_infero_temporal_[um]': 'RNFL thickness in the inferotemporal quadrant of the peripapillary region (in microns).',
    'ILM_RNFL_nasal_[um]': 'RNFL thickness in the nasal quadrant of the peripapillary region (in microns).',
    'ILM_RNFL_supero_nasal_[um]': 'RNFL thickness in the superonasal quadrant of the peripapillary region (in microns).',
    'ILM_RNFL_supero_temporal_[um]': 'RNFL thickness in the superotemporal quadrant of the peripapillary region (in microns).',
    'ILM_RNFL_temporal_[um]': 'RNFL thickness in the temporal quadrant of the peripapillary region (in microns).',
    
    'RNFL_BM_All_[um]': 'Average thickness of the RNFL from the ILM to Bruch’s membrane (BM) across the entire peripapillary region (in microns).',
    'RNFL_BM_N/T': 'Ratio of nasal to temporal RNFL thickness (ILM to BM) in the peripapillary region (dimensionless).',
    'RNFL_BM_PMB_[um]': 'RNFL thickness (ILM to BM) in the papillomacular bundle (PMB) region (in microns).',
    'RNFL_BM_infero_nasal_[um]': 'RNFL thickness (ILM to BM) in the inferonasal quadrant of the peripapillary region (in microns).',
    'RNFL_BM_infero_temporal_[um]': 'RNFL thickness (ILM to BM) in the inferotemporal quadrant of the peripapillary region (in microns).',
    'RNFL_BM_nasal_[um]': 'RNFL thickness (ILM to BM) in the nasal quadrant of the peripapillary region (in microns).',
    'RNFL_BM_supero_nasal_[um]': 'RNFL thickness (ILM to BM) in the superonasal quadrant of the peripapillary region (in microns).',
    'RNFL_BM_supero_temporal_[um]': 'RNFL thickness (ILM to BM) in the superotemporal quadrant of the peripapillary region (in microns).',
    'RNFL_BM_temporal_[um]': 'RNFL thickness (ILM to BM) in the temporal quadrant of the peripapillary region (in microns).',
    
    'CHORupper_CHORlower_All_[um]': 'Average choroidal thickness across the entire peripapillary region (in microns).',
    'CHORupper_CHORlower_N/T': 'Ratio of nasal to temporal choroidal thickness in the peripapillary region (dimensionless).',
    'CHORupper_CHORlower_PMB_[um]': 'Choroidal thickness in the papillomacular bundle (PMB) region (in microns).',
    'CHORupper_CHORlower_infero_nasal_[um]': 'Choroidal thickness in the inferonasal quadrant of the peripapillary region (in microns).',
    'CHORupper_CHORlower_infero_temporal_[um]': 'Choroidal thickness in the inferotemporal quadrant of the peripapillary region (in microns).',
    'CHORupper_CHORlower_nasal_[um]': 'Choroidal thickness in the nasal quadrant of the peripapillary region (in microns).',
    'CHORupper_CHORlower_supero_nasal_[um]': 'Choroidal thickness in the superonasal quadrant of the peripapillary region (in microns).',
    'CHORupper_CHORlower_supero_temporal_[um]': 'Choroidal thickness in the superotemporal quadrant of the peripapillary region (in microns).',
    'CHORupper_CHORlower_temporal_[um]': 'Choroidal thickness in the temporal quadrant of the peripapillary region (in microns).',
}


ppole_cols = {
    'Filename': 'Name of the file for easy look-up with metadata sheet.',
    
    # Retinal thickness measurements (in microns)
    'all_retina_[um]': 'Average retinal thickness across the entire ETDRS region (in microns).',
    'central_retina_[um]': 'Retinal thickness at the central ETDRS subfield (in microns).',
    'inner_inferior_retina_[um]': 'Retinal thickness in the inner inferior ETDRS subfield (in microns).',
    'inner_nasal_retina_[um]': 'Retinal thickness in the inner nasal ETDRS subfield (in microns).',
    'inner_superior_retina_[um]': 'Retinal thickness in the inner superior ETDRS subfield (in microns).',
    'inner_temporal_retina_[um]': 'Retinal thickness in the inner temporal ETDRS subfield (in microns).',
    'outer_inferior_retina_[um]': 'Retinal thickness in the outer inferior ETDRS subfield (in microns).',
    'outer_nasal_retina_[um]': 'Retinal thickness in the outer nasal ETDRS subfield (in microns).',
    'outer_superior_retina_[um]': 'Retinal thickness in the outer superior ETDRS subfield (in microns).',
    'outer_temporal_retina_[um]': 'Retinal thickness in the outer temporal ETDRS subfield (in microns).',
    
    # Inner retina thickness measurements
    'all_inner_retina_[um]': 'Average inner retinal thickness across the ETDRS region (in microns).',
    'central_inner_retina_[um]': 'Inner retinal thickness at the central ETDRS region (in microns).',
    'inner_inferior_inner_retina_[um]': 'Inner retinal thickness in the inner inferior ETDRS subfield (in microns).',
    'inner_nasal_inner_retina_[um]': 'Inner retinal thickness in the inner nasal ETDRS subfield (in microns).',
    'inner_superior_inner_retina_[um]': 'Inner retinal thickness in the inner superior ETDRS subfield (in microns).',
    'inner_temporal_inner_retina_[um]': 'Inner retinal thickness in the inner temporal ETDRS subfield (in microns).',
    'outer_inferior_inner_retina_[um]': 'Inner retinal thickness in the outer inferior ETDRS subfield (in microns).',
    'outer_nasal_inner_retina_[um]': 'Inner retinal thickness in the outer nasal ETDRS subfield (in microns).',
    'outer_superior_inner_retina_[um]': 'Inner retinal thickness in the outer superior ETDRS subfield (in microns).',
    'outer_temporal_inner_retina_[um]': 'Inner retinal thickness in the outer temporal ETDRS subfield (in microns).',
    
    # Outer retina thickness measurements
    'all_outer_retina_[um]': 'Average outer retinal thickness across the ETDRS region (in microns).',
    'central_outer_retina_[um]': 'Outer retinal thickness at the central ETDRS region (in microns).',
    'inner_inferior_outer_retina_[um]': 'Outer retinal thickness in the inner inferior ETDRS subfield (in microns).',
    'inner_nasal_outer_retina_[um]': 'Outer retinal thickness in the inner nasal ETDRS subfield (in microns).',
    'inner_superior_outer_retina_[um]': 'Outer retinal thickness in the inner superior ETDRS subfield (in microns).',
    'inner_temporal_outer_retina_[um]': 'Outer retinal thickness in the inner temporal ETDRS subfield (in microns).',
    'outer_inferior_outer_retina_[um]': 'Outer retinal thickness in the outer inferior ETDRS subfield (in microns).',
    'outer_nasal_outer_retina_[um]': 'Outer retinal thickness in the outer nasal ETDRS subfield (in microns).',
    'outer_superior_outer_retina_[um]': 'Outer retinal thickness in the outer superior ETDRS subfield (in microns).',
    'outer_temporal_outer_retina_[um]': 'Outer retinal thickness in the outer temporal ETDRS subfield (in microns).',

    # Individual inner retinal layer thickness measurements
    'all_outer_{Layer1}_{Layer2}_[um]': 'Average outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness across the ETDRS region (in microns).',
    'central_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness at the central ETDRS region (in microns).',
    'inner_inferior_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness in the inner inferior ETDRS subfield (in microns).',
    'inner_nasal_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness in the inner nasal ETDRS subfield (in microns).',
    'inner_superior_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness in the inner superior ETDRS subfield (in microns).',
    'inner_temporal_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness in the inner temporal ETDRS subfield (in microns).',
    'outer_inferior_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness in the outer inferior ETDRS subfield (in microns).',
    'outer_nasal_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness in the outer nasal ETDRS subfield (in microns).',
    'outer_superior_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness in the outer superior ETDRS subfield (in microns).',
    'outer_temporal_outer_{Layer1}_{Layer2}_[um]': 'Outer {Layer2} (e.g. ILM_RNFL implies the retinal nerve fiber layer (RNFL)) thickness in the outer temporal ETDRS subfield (in microns).',
    
    # Choroidal measurements
    'all_choroid_[um]': "Average choroidal thickness across all ETDRS subfields (in microns).",
    'central_choroid_[um]': "Choroidal thickness at the central ETDRS region (in microns).",
    'inner_inferior_choroid_[um]': "Choroidal thickness in the inner inferior ETDRS subfield (in microns).",
    'inner_nasal_choroid_[um]': "Choroidal thickness in the inner nasal ETDRS subfield (in microns).",
    'inner_superior_choroid_[um]': "Choroidal thickness in the inner superior ETDRS subfield (in microns).",
    'inner_temporal_choroid_[um]': "Choroidal thickness in the inner temporal ETDRS subfield (in microns).",
    'outer_inferior_choroid_[um]': "Choroidal thickness in the outer inferior ETDRS subfield (in microns).",
    'outer_nasal_choroid_[um]': "Choroidal thickness in the outer nasal ETDRS subfield (in microns).",
    'outer_superior_choroid_[um]': "Choroidal thickness in the outer superior ETDRS subfield (in microns).",
    'outer_temporal_choroid_[um]': "Choroidal thickness in the outer temporal ETDRS subfield (in microns).",

    'all_choroid_vessel_[um2]': "Total choroidal vessel area across all ETDRS subfields (in square microns).",
    'central_choroid_vessel_[um2]': "Choroidal vessel area at the central ETDRS region (in square microns).",
    'inner_inferior_choroid_vessel_[um2]': "Choroidal vessel area in the inner inferior ETDRS subfield (in square microns).",
    'inner_nasal_choroid_vessel_[um2]': "Choroidal vessel area in the inner nasal ETDRS subfield (in square microns).",
    'inner_superior_choroid_vessel_[um2]': "Choroidal vessel area in the inner superior ETDRS subfield (in square microns).",
    'inner_temporal_choroid_vessel_[um2]': "Choroidal vessel area in the inner temporal ETDRS subfield (in square microns).",
    'outer_inferior_choroid_vessel_[um2]': "Choroidal vessel area in the outer inferior ETDRS subfield (in square microns).",
    'outer_nasal_choroid_vessel_[um2]': "Choroidal vessel area in the outer nasal ETDRS subfield (in square microns).",
    'outer_superior_choroid_vessel_[um2]': "Choroidal vessel area in the outer superior ETDRS subfield (in square microns).",
    'outer_temporal_choroid_vessel_[um2]': "Choroidal vessel area in the outer temporal ETDRS subfield (in square microns).",

    'all_choroid_CVI': "Choroidal vascularity index (CVI) across all ETDRS subfields (dimensionless).",
    'central_choroid_CVI': "Choroidal vascularity index (CVI) at the central ETDRS region (dimensionless).",
    'inner_inferior_choroid_CVI': "Choroidal vascularity index (CVI) in the inner inferior ETDRS subfield (dimensionless).",
    'inner_nasal_choroid_CVI': "Choroidal vascularity index (CVI) in the inner nasal ETDRS subfield (dimensionless).",
    'inner_superior_choroid_CVI': "Choroidal vascularity index (CVI) in the inner superior ETDRS subfield (dimensionless).",
    'inner_temporal_choroid_CVI': "Choroidal vascularity index (CVI) in the inner temporal ETDRS subfield (dimensionless).",
    'outer_inferior_choroid_CVI': "Choroidal vascularity index (CVI) in the outer inferior ETDRS subfield (dimensionless).",
    'outer_nasal_choroid_CVI': "Choroidal vascularity index (CVI) in the outer nasal ETDRS subfield (dimensionless).",
    'outer_superior_choroid_CVI': "Choroidal vascularity index (CVI) in the outer superior ETDRS subfield (dimensionless).",
    'outer_temporal_choroid_CVI': "Choroidal vascularity index (CVI) in the outer temporal ETDRS subfield (dimensionless).",

    'all_choroid_[mm3]': "Total choroidal volume across all ETDRS subfields (in cubic millimeters).",
    'central_choroid_[mm3]': "Choroidal volume at the central ETDRS region (in cubic millimeters).",
    'inner_inferior_choroid_[mm3]': "Choroidal volume in the inner inferior ETDRS subfield (in cubic millimeters).",
    'inner_nasal_choroid_[mm3]': "Choroidal volume in the inner nasal ETDRS subfield (in cubic millimeters).",
    'inner_superior_choroid_[mm3]': "Choroidal volume in the inner superior ETDRS subfield (in cubic millimeters).",
    'inner_temporal_choroid_[mm3]': "Choroidal volume in the inner temporal ETDRS subfield (in cubic millimeters).",
    'outer_inferior_choroid_[mm3]': "Choroidal volume in the outer inferior ETDRS subfield (in cubic millimeters).",
    'outer_nasal_choroid_[mm3]': "Choroidal volume in the outer nasal ETDRS subfield (in cubic millimeters).",
    'outer_superior_choroid_[mm3]': "Choroidal volume in the outer superior ETDRS subfield (in cubic millimeters).",
    'outer_temporal_choroid_[mm3]': "Choroidal volume in the outer temporal ETDRS subfield (in cubic millimeters).",

    'all_choroid_vessel_[mm3]': "Total choroidal vessel volume across all ETDRS subfields (in cubic millimeters).",
    'central_choroid_vessel_[mm3]': "Choroidal vessel volume at the central ETDRS region (in cubic millimeters).",
    'inner_inferior_choroid_vessel_[mm3]': "Choroidal vessel volume in the inner inferior ETDRS subfield (in cubic millimeters).",
    'inner_nasal_choroid_vessel_[mm3]': "Choroidal vessel volume in the inner nasal ETDRS subfield (in cubic millimeters).",
    'inner_superior_choroid_vessel_[mm3]': "Choroidal vessel volume in the inner superior ETDRS subfield (in cubic millimeters).",
    'inner_temporal_choroid_vessel_[mm3]': "Choroidal vessel volume in the inner temporal ETDRS subfield (in cubic millimeters).",
    'outer_inferior_choroid_vessel_[mm3]': "Choroidal vessel volume in the outer inferior ETDRS subfield (in cubic millimeters).",
    'outer_nasal_choroid_vessel_[mm3]': "Choroidal vessel volume in the outer nasal ETDRS subfield (in cubic millimeters).",
    'outer_superior_choroid_vessel_[mm3]': "Choroidal vessel volume in the outer superior ETDRS subfield (in cubic millimeters).",
    'outer_temporal_choroid_vessel_[mm3]': "Choroidal vessel volume in the outer temporal ETDRS subfield (in cubic millimeters).",

}


import pandas as pd
metakey_df = pd.DataFrame({'column':meta_cols.keys(), 'description':meta_cols.values()})
linescanradial_df = pd.DataFrame({'column':lsr_cols.keys(), 'description':lsr_cols.values()})
ppole_df = pd.DataFrame({'column':ppole_cols.keys(), 'description':ppole_cols.values()})
peripapillary_df = pd.DataFrame({'column':peri_cols.keys(), 'description':peri_cols.values()})
slo_df = pd.DataFrame({'column':slo_cols.keys(), 'description':slo_cols.values()})