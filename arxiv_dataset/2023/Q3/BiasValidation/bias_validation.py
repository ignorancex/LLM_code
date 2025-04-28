#!/usr/bin/env python
import multiprocessing
from multiprocessing.pool import Pool
import argparse
import logging
import coloredlogs
import tomli as tml
from tqdm import tqdm
from pathlib import Path
import difflib
import shutil
import subprocess
import sys
import os
from files.scripts.analysis import *
import pickle

"""
Default config options
"""
config = {
    "logfile": "log.txt",
    "base": "/project2/rkessler/SURVEYS/DES/USERS/parmstrong/dev/bias_validation/",
    "clean": True
}

"""
Prepare logging to STDOUT and a file
"""
def setup_logging(logging_filename, verbose):
    level = logging.DEBUG if verbose else logging.INFO

    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s" if verbose else "%(message)s"
    handlers = [logging.StreamHandler()]
    handlers[0].setLevel(level)
    handlers.append(logging.FileHandler(logging_filename, mode="w"))
    handlers[-1].setLevel(logging.DEBUG)
    logging.basicConfig(level=level, format=fmt, handlers=handlers)

    coloredlogs.install(
        level = level,
        fmt = fmt,
        reconfigure = True,
        level_styles = coloredlogs.parse_encoded_styles("debug=8;info=green;warning=yellow;error=red,bold;critical=red,inverse"),
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

"""
Read in arguments. By default, bias_validation will check whether your Pippin job has been run before (but not whether it has succeeded), so to force your Pippin job to be rerun, pass -n
"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("toml", help="the name of the toml config file to run.", type=str)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-n", "--no_skip", help="Rerun pippin jobs", action="store_true")

    args = parser.parse_args()
    return args

"""
Setup validation cosmology Pippin runs. Mostly, this reads in options and then replaces keys in input_files / validation_master.yml
"""
def validation_setup(options, config, logger):
    if options is None:
        logger.error("No validation cosmology specified, quitting")
        return None
    name = list(options.keys())[0]
    options = options[name]
    Om_list = options["OMEGA_MATTER"]
    w0_list = options["W0_LAMBDA"]
    input_path = config.get("pippin_inputs") / f"BV_V_{name}.yml"
    master = config.get("input_files") / "validation_master.yml"
    reference_sim_path = config["pippin_outputs"][[key for key in config["pippin_outputs"].keys() if "BV_FR" in key][0]][0] / "1_SIM" / "DESBIASCOR"
    reference_lcfit_path = config["pippin_outputs"][[key for key in config["pippin_outputs"].keys() if "BV_FR" in key][0]][0] / "2_LCFIT" / "DESFIT_DESBIASCOR"
    if options.get("share_biascor", False):
        biascor = f"""EXTERNAL: {reference_sim_path}
        """
        EXTERNAL_FIT = f"EXTERNAL: {reference_lcfit_path}"
    else:
        EXTERNAL_FIT = ""
        biascor = """IA_G10_DES3YR:
            BASE: /project2/rkessler/SURVEYS/DES/USERS/parmstrong/dev/bias_validation/files/base_files/sn_ia_salt2_g10_des3yr.input
            GENSIGMA_SALT2ALPHA: 1E8  1E8
            GENRANGE_SALT2ALPHA: 0.12 0.20
            GENGRID_SALT2ALPHA: 2
            GENSIGMA_SALT2BETA: 1E8  1E8
            GENRANGE_SALT2BETA: 2.6  3.6
            GENGRID_SALT2BETA: 2
        GLOBAL:
            W0_LAMBDA: {W0_LAMBDA}
            OMEGA_MATTER: {OMEGA_MATTER}
            OMEGA_LAMBDA: {OMEGA_LAMBDA}
            NGEN_UNIT: 50.0
            RANSEED_REPEAT: 50 12345
        """
    # Edit the master yml file
    SIM_MASTER = """DESSIM_{i}:
        IA_G10_DES3YR:
            BASE: /project2/rkessler/SURVEYS/DES/USERS/parmstrong/dev/bias_validation/files/base_files/sn_ia_salt2_g10_des3yr.input
        GLOBAL:
            W0_LAMBDA: {W0_LAMBDA}
            OMEGA_MATTER: {OMEGA_MATTER}
            OMEGA_LAMBDA: {OMEGA_LAMBDA}
            NGEN_UNIT: 5.0
            RANSEED_CHANGE: 150 12345
    DESBIASCOR_{i}:
        {biascor}
    """
    REPLACE_SIM = ""
    for i in range(len(Om_list)):
        Om = Om_list[i]
        w0 = w0_list[i]
        bcor = biascor.format(**{"OMEGA_MATTER": Om, "OMEGA_LAMBDA": 1-Om, "W0_LAMBDA": w0})
        SIM = SIM_MASTER.format(**{"biascor": bcor, "i": i, "OMEGA_MATTER": Om, "OMEGA_LAMBDA": 1-Om, "W0_LAMBDA": w0})
        REPLACE_SIM += SIM
    BIASCOR_MASTER = """BCOR_OMW_NO_OMPRI_{i}:
        BASE: /project2/rkessler/SURVEYS/DES/USERS/parmstrong/dev/bias_validation/files/base_files/SALT2mu_no_ompri.input
        DATA: [DESFITSYS_DESSIM_{i}]
        SIMFILE_BIASCOR: [DESFIT_DESBIASCOR_{i}]
        NMAX: 190
        OPTS:
            BATCH_INFO: sbatch $SBATCH_TEMPLATES/SBATCH_Midway2b.TEMPLATE 50
    """
    REPLACE_BIASCOR = ""
    for i in range(len(Om_list)):
        Om = Om_list[i]
        w0 = w0_list[i]
        BIASCOR = BIASCOR_MASTER.format(**{"i": i, "OMEGA_MATTER": Om, "W0_LAMBDA": w0})
        REPLACE_BIASCOR += BIASCOR 
    COV_MASTER = """COV_OMW_NO_OMPRI_{i}:
        MASK: BCOR_OMW_NO_OMPRI_{i}
    """
    REPLACE_COV = ""
    for i in range(len(Om_list)):
        COV = COV_MASTER.format(**{"i": i})
        REPLACE_COV += COV
    WFIT_MASTER = """   SN_OMW_NO_OMPRI_{i}:
            MASK: COV_OMW_NO_OMPRI_{i}
            OPTS:
                BATCH_INFO: sbatch $SBATCH_TEMPLATES/SBATCH_Midway2b.TEMPLATE 50
                WFITOPTS:
                    - /om_no_pri/ -ompri {Om} -dompri 50 -outfile_chi2grid chi2.txt -minchi2 -omsteps 201 -ommin {Om_min} -ommax {Om_max} -w0steps 301 -w0min {w0_min} -w0max {w0_max}
                WFITAVG:
                    - COV_OMW_NO_OMPRI_{i}_BCOR_OMW_NO_OMPRI_{i}
    """
    REPLACE_WFIT = ""
    for i in range(len(Om_list)):
        Om = Om_list[i]
        Om_min = Om - 1
        Om_max = Om + 1
        w0 = w0_list[i]
        w0_min = w0 - 2
        w0_max = w0 + 2
        WFIT = WFIT_MASTER.format(**{"i": i, "Om": Om, "w0": w0, "Om_min": Om_min, "Om_max": Om_max, "w0_min": w0_min, "w0_max": w0_max})
        REPLACE_WFIT += WFIT
    with open(master, 'r') as f:
        tmp = f.read()
    prepend = f"# Generated via:\n# [ validation.{name} ]\n# OMEGA_MATTER = {Om_list}\n# W0_LAMBDA = {w0_list}"
    tmp = tmp.format(**{"PREPEND": prepend, "EXTERNAL_FIT": EXTERNAL_FIT, "REPLACE_SIM": REPLACE_SIM, "REPLACE_BIASCOR": REPLACE_BIASCOR, "REPLACE_COV": REPLACE_COV, "REPLACE_WFIT": REPLACE_WFIT})
    skip = False
    if input_path.exists():
        logger.info(f"Found another input file under {name}, comparing them now")
        with open(input_path, 'r') as f:
            tmp2 = f.read()
        diffs = list(difflib.unified_diff(tmp.split('\n'), tmp2.split('\n'), fromfile="new_input", tofile="old_input", lineterm=''))
        if len(diffs) == 0:
            logger.info("No differences found, will not rerun")
            skip = True
        else:
            logger.warning(f"Found the following differences:")
            for line in diffs:
                logger.warning(line)
            logger.warning("If you continue, you will overwrite previously defined inputs. Would you like to continue (y/n)?")
            response = input().upper()
            if response == "Y":
                logger.warning("You have chosen to continue, overwriting previous input files.")
            else:
                logger.warning("Continuing with previous input, will not rerun")
                skip = True
    else:
        # Make sure all subdirs exist
        input_path.touch(exist_ok=True)
    with open(input_path, 'w') as f:
        f.write(tmp)
    return input_path, skip, Om_list, w0_list

"""
Setup reference cosmology Pippin runs. Mostly, this reads in options and then replaces keys in input_files / reference_master.yml
"""
def reference_setup(options, config, logger):
    if options is None:
        logger.error("No reference cosmology specified, quitting")
        return None
    name = list(options.keys())[0]
    options = options[name]
    Om = options["OMEGA_MATTER"]
    w0 = options["W0_LAMBDA"]
    input_path = config.get("pippin_inputs") / f"BV_FR_{name}.yml"
    master = config.get("input_files") / "reference_master.yml"
    Om_min = Om - 1
    Om_max = Om + 1
    w0_min = w0 - 2
    w0_max = w0 + 2
    # Edit the master yml file
    with open(master, 'r') as f:
        tmp = f.read()
    prepend = f"# Generated via:\n# [ reference.{name} ]\n# OMEGA_MATTER = {Om}\n# W0_LAMBDA = {w0}"
    tmp = tmp.format(**{"PREPEND": prepend, "W0_LAMBDA": w0, "OMEGA_MATTER": Om, "OMEGA_LAMBDA": 1 - Om, "Om_min": Om_min, "Om_max": Om_max, "w0_min": w0_min, "w0_max": w0_max})
    # Check if a file already exists, and whether you should overwrite that file
    skip = False
    if input_path.exists():
        logger.info(f"Found another input file under {name}, comparing them now")
        with open(input_path, 'r') as f:
            tmp2 = f.read()
        diffs = list(difflib.unified_diff(tmp.split('\n'), tmp2.split('\n'), fromfile="new_input", tofile="old_input", lineterm=''))
        if len(diffs) == 0:
            logger.info("No differences found, will not rerun")
            skip = True
        else:
            logger.warning(f"Found the following differences:")
            for line in diffs:
                logger.warning(line)
            logger.warning("If you continue, you will overwrite previously defined inputs. Would you like to continue (y/n)?")
            response = input().upper()
            if response == "Y":
                logger.warning("You have chosen to continue, overwriting previous input file.")
            else:
                logger.warning("Continuing with previous input, will not rerun")
                skip = True
    else:
        # Make sure all subdirs exist
        input_path.touch(exist_ok=True)
    with open(input_path, 'w') as f:
        f.write(tmp)
    return input_path, skip, Om, w0

"""
Read in all of the defined Pippin outputs, then run all the plotting and analysis
"""
def analyse(pippin_outputs, options, config, logger):
    if len(options.keys()) == 0:
        logger.warning("No analysis options chosen, quitting")
        return None
    #num_cpu = multiprocessing.cpu_count()
    if config["wfits"].exists():
        wfits = pickle.load(open(config["wfits"], "rb"))
    else:
        num_cpu = 8
        logger.info(f"Num CPU: {num_cpu}")
        wfits = {
            "FR": {},
            "V": {}
        }
        nom_Om = None
        nom_w0 = None
        logger.info(f"PO: {pippin_outputs}")
        mask = options.get("mask", "")
        for (name, (path, Om_l, w0_l)) in pippin_outputs.items():
            if "_FR_" in name:
                logger.info("Loading in full runthrough wfit output")
                d = wfits["FR"]
            else:
                logger.info("Loading in validation wfit output")
                d = wfits["V"]
            wfit_dirs = get_wfit_dirs(path)
            wfit_file_dict = {wfit_dir: get_wfit_files(wfit_dir) for wfit_dir in wfit_dirs}
            for wfit_dir in wfit_file_dict.keys():
                wfit_name = wfit_dir.parts[-1]
                if not mask in wfit_name:
                    logger.info(f"Skipping {wfit_name}")
                    continue
                if "_FR_" in name:
                    i = 0
                    Om = Om_l
                    w0 = w0_l
                    nom_Om = Om
                    nom_w0 = w0
                else:
                    i = int(str(wfit_dir)[-1])
                    Om = Om_l[i]
                    w0 = w0_l[i]
                logger.info(f"{wfit_dir}, {i}, {Om}, {w0}")
                wfit_files = wfit_file_dict[wfit_dir]
                d[str(wfit_dir)] = {}
                with Pool(num_cpu) as pool:
                    for (f, data, best) in tqdm(pool.imap(read_wfit, wfit_files), total=len(wfit_files)):
                        d[str(wfit_dir)][str(f)] = {}
                        d[str(wfit_dir)][str(f)]["data"] = data
                        d[str(wfit_dir)][str(f)]["best"] = best
                        d[str(wfit_dir)][str(f)]["Om"] = Om
                        d[str(wfit_dir)][str(f)]["w0"] = w0
        pickle.dump(wfits, open(config["wfits"], "wb"))

    plot_contour_options = options.get("plot_contour", None)
    if plot_contour_options is not None:
        plot_contour(wfits, plot_contour_options, config["plot_output"], logger)
    plot_percentile_options = options.get("plot_percentile", None)
    if plot_percentile_options is not None:
        plot_percentile(wfits, plot_percentile_options, config["plot_output"], logger)
    plot_nominal_options = options.get("plot_nominal", None)
    if plot_nominal_options is not None:
        plot_nominal(wfits, plot_nominal_options, config["plot_output"], logger)
    plot_GPE_options = options.get("plot_GPE", None)
    if plot_GPE_options is not None:
        plot_GPE(wfits, plot_GPE_options, config["plot_output"], logger)
    plot_KDE_options = options.get("plot_KDE", None)
    if plot_KDE_options is not None:
        plot_KDE(wfits, plot_KDE_options, config["plot_output"], logger)
    plot_ellipse_options = options.get("plot_ellipse", None)
    if plot_ellipse_options is not None:
        plot_ellipse(wfits, plot_ellipse_options, config["plot_output"], logger)
    plot_comparison_options = options.get("plot_comparison", None)
    if plot_comparison_options is not None:
        plot_comparison(wfits, plot_comparison_options, config["plot_output"], logger)
    plot_final_options = options.get("plot_final", None)
    if plot_final_options is not None:
        plot_final(wfits, plot_final_options, config["plot_output"], logger)
    plot_all_options = options.get("plot_all", None)
    if plot_all_options is not None:
        plot_all(wfits, plot_all_options, config["plot_output"], logger)

"""
Entry point function, reads in toml input file and runs bias_validation
"""
def run():
    args = get_args()
    toml_path = Path(args.toml).resolve()
    with open(toml_path, 'rb') as f:
        toml = tml.load(f)
    # Setup paths
    config["base"] = Path(config["base"])
    config["files"] = config["base"] / "files"
    config["base_files"] = config["files"] / "base_files"
    config["input_files"] = config["files"] / "input_files"
    config["inputs"] = config["base"] / "inputs"
    config["pippin_inputs"] = config["inputs"] / "Pippin"
    config["outputs"] = config["base"] / "outputs"
    config["wfits"] = config["outputs"] / "wfits.p"
    config["toml_input"] = toml_path.parents
    config["output"] = config["outputs"] / toml_path.stem
    config["plot_output"] = config["output"] / "plots"
    config["pippin_output"] = Path(os.environ["PIPPIN_OUTPUT"])
    config["pippin_outputs"] = {}
    output = config["output"]
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)
    if not config["plot_output"].exists():
        config["plot_output"].mkdir(parents=True, exist_ok=True)
    logfile = output / config.get("logfile")
    if not logfile.exists():
        logfile.touch(exist_ok=True)
    setup_logging(logfile, args.verbose)
    logger = logging.getLogger("bias_validation")
    logger.info(f"Config complete, output folder set to {output}, logging to {logfile})")
    logger.info("Preparing reference cosmology pippin input")

    # Start of reference cosmology
    reference_options = toml.get("reference", None)
    if reference_options is not None:
        reference_path, skip_reference, reference_Om, reference_w0 = reference_setup(reference_options, config, logger)
        if reference_path is None:
            logger.warning("No full runthrough paths defined")
        else:
            if skip_reference and not args.no_skip:
                logging.info(f"Skipping {reference_path}")
            else:
                logging.info(f"Reference setup complete, running {reference_path}")
                cmd = ["pippin.sh", "-v", str(reference_path)]
                logger.debug(f"Running {cmd}")
                rtn = subprocess.run(cmd).returncode
                
                logger.debug(f"Completed with exit code {rtn}")
                if rtn != 0:
                    logger.warning(f"Non-zero return code {rtn}. The reference job likely failed, will attempt to continue anyway")
            job_name = reference_path.stem.upper()
            output_dir = config["pippin_output"] / job_name
            if output_dir.exists():
                config["pippin_outputs"][job_name] = (output_dir, reference_Om, reference_w0)
                logger.debug(f"Added {output_dir} to analysis list")
            logger.info("Completed reference cosmology")

    # Validation jobs
    validation_options = toml.get("validation", None)
    if validation_options is not None:
        validation_path, skip_validation, validation_Om, validation_w0 = validation_setup(validation_options, config, logger) 
        if validation_path is None:
            logger.warning("No validation paths defined")
        else:
            if skip_validation and not args.no_skip:
                logging.info(f"Skipping {validation_path}")
            else:
                logging.info(f"validation setup complete, running {validation_path}")
                cmd = ["pippin.sh", "-v", str(validation_path)]
                logger.debug(f"Running {cmd}")
                rtn = subprocess.run(cmd).returncode
                
                logger.debug(f"Completed with exit code {rtn}")
                if rtn != 0:
                    logger.warning(f"Non-zero return code {rtn}. The validation job likely failed, will attempt to continue anyway")
            job_name = validation_path.stem.upper()
            output_dir = config["pippin_output"] / job_name
            if output_dir.exists():
                config["pippin_outputs"][job_name] = (output_dir, validation_Om, validation_w0)
                logger.debug(f"Added {output_dir} to analysis list")
            logger.info("Completed validation cosmology")

    # Analysis
    if toml.get("analyse", None) is not None:
        logger.debug(config["pippin_outputs"])
        for options in toml["analyse"]:
            analyse(config["pippin_outputs"], options, config, logger)

if __name__ == "__main__":
    run()
