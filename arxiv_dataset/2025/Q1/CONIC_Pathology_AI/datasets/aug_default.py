aug_params =    {"grid_distortion_num_steps": 5, #default: 5
                 "grid_distortion_distort_limit": 0.02, #default 0.3
                 "grid_dropout_ratio": 0.3, #default 0.5
                 "random_resizecrop_min": 0.8, #0.08
                 "random_resizecrop_ration_min": 0.9, # 0.75
                 "random_resizecrop_ration_max": 1.1, # 1.3333
                 "random_grid_shuffle_grid": 2, # 3
                 "elastic_transform_sigma": 10, # 50
                 "elastic_transform_alpha_affine": 10, #50 
                 "piecewise_affine_scale_min": 0.01, # 0.03
                 "piecewise_affine_scale_max": 0.03, # 0.05 jitter how far each point is moved
                 "piecewise_affine_nb_rows": 4, # 4 >=2, larger number for larger image
                 "piecewise_affine_nb_cols": 4, # 4 >=2
                 "clahe_clip_limit": 2, # 4 upper threshold value for contrast limiting
                 "color_jitter_brightness": 0.2, # 0.2
                 "color_jitter_contrast": 0.25, # 0.2
                 "color_jitter_saturation": 0.2, # 0.2
                 "color_jitter_hue": 0.05, # 0.2
                 "down_scale_min": 0.5, # 0.25 lower bound on the image scale
                 "down_scale_max": 0.99, # 0.25 upper bound on the down scale
                 "emboss_alpha_min": 0, # 0.2 range of the visibility of the embossed image
                 "emboss_alpha_max": 0.3, # 0.5 0.0: only the original image 1.0: only the embossed image
                 "emboss_strength_min": 0.0, # 0.2 strength range of the embossing 
                 "emboss_strength_max": 0.2, # 0.7 as above
                 "Gauss_noise_var_min": 0.0, # 10.0 variance range for noise
                 "Gauss_noise_var_max": 10.0, # 50.0 as above
                 "Gauss_blur_limit": 3, # (3, 7)
                 "glass_blur_sigma": 0.3, # 0.7 standard deviation for Gaussian kernel
                 "hsv_hue": 30, # 20 
                 "hsv_sat": 30, # 30
                 "hsv_val": 30, # 20
                 "iso_color_shift_min": 0.01, # 0.01 variance range for color hue change
                 "iso_color_shift_max": 0.03, # 0.05 as above
                 "iso_intensity_min": 0.01, # 0.1 multiplicative factor that control strength of color and luminace noise
                 "iso_intensity_max": 0.1, # 0.5 as above
                 "img_compression_quality_lower": 90, #99 >=0 <=100, lower bound on the image quality
                 "img_compression_quality_upper": 100, # 100 as above
                 "median_blur_limit": 3, # 7 must be odd, >=3 <inf
                 "motion_blur_limit":  3, # (3, 7) >=3, <inf
                 "multiplicative_noise_multiplier_min": 0.9, # 0.9 
                 "multiplicative_noise_multiplier_max": 1.1, # 1.1
                 "posterrize_num_bits": 4, # 4, number of hight bits >=0 <=8
                 "random_brightness_contrast_brightness": 0.2, # 0.2
                 "random_brightness_contrast_contrast": 0.2, #0.2
                 "random_gamma_limit_min": 80, # 80
                 "random_gamma_limit_max": 120, # 120
                 "random_fog_coef_lower": 0.3, # 0.3 lower limit for fog intensity coefficient >=0 <=1
                 "random_fog_coef_upper": 1, # 1 as above
                 "random_fog_alpha_coef": 0.08, # 0.08 transparency of the fog circles
                 "random_rain_slant_lower": -10, # -10 >=-20 <=20 
                 "random_rain_slant_upper": 10, # 10 >=-20 <=20
                 "random_rain_drop_length": 20, # 20 >=0 <=100
                 "random_shadow_num_shadows_upper": 3, # 2 upper limit of the possible shadows
                 "random_snow_point_upper": 0.3, # 0.3 
                 "random_tone_curve_scale": 0.1, # 0.1 standard deviation of the normal distribution >=0 <=1
                 "superpixel_n_segments": 500, # 100 rough targer number of superpixels
                 }


aug_this_try = {"Compose": [["horizontal_flip", "vertical_flip", "random_rotate90", "transpose", "color_jitter", "center_crop"]],
                "Someof": [],
                "Oneof": ["Gaussian_blur", "median_blur", "motion_blur"]}

aug_None = {"Compose": [["center_crop"]],
                "Someof": [],
                "Oneof": []}

aug_flip = {"Compose": [["horizontal_flip", "vertical_flip", "random_rotate90", "center_crop"]],
                "Someof": [],
                "Oneof": []}

aug_color_jitter = {"Compose": [["color_jitter", "center_crop"]],
                "Someof": [],
                "Oneof": []}

aug_blur = {"Compose": [["Gaussian_blur", "median_blur", "motion_blur", "center_crop"]],
                "Someof": [],
                "Oneof": []}

aug_distort = {"Compose": [["grid_distortion", "elastic_transform", "center_crop"]],
                "Someof": [],
                "Oneof": []}

