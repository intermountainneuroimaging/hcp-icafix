from pathlib import Path
import os, logging
import subprocess as sp
import pandas as pd
import math
import numpy as np
import json
import shutil

log = logging.getLogger(__name__)

def execute_shell(cmd, dryrun=False, cwd=os.getcwd()):
    log.info("\n %s", cmd)
    if not dryrun:
        terminal = sp.Popen(
            cmd,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
            cwd=cwd
        )
        stdout, stderr = terminal.communicate()
        log.info("\n %s", stdout)
        log.info("\n %s", stderr)

        return stdout.strip('\n')


def build_lookup(analysis, fw):
    subject = fw.get_subject(analysis.parents["subject"])
    session = fw.get_session(analysis.parents["session"])
    gearname = analysis.gear_info["name"]

    lookup_table = {"PIPELINE": gearname, "SUBJECT": subject.label, "SESSION": session.label}

    return lookup_table


def apply_lookup(text, lookup_table):
    if '{' in text and '}' in text:
        for lookup in lookup_table:
            text = text.replace('{' + lookup + '}', lookup_table[lookup])
    return text


def motion_to_fmripreplike(filepath):
    os.makedirs(os.path.join(filepath.parent, "mc"), exist_ok=True)

    data = pd.read_csv(filepath, header=None, delim_whitespace=True)

    data.columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'trans_x_derivative1',
                    'trans_y_derivative1', 'trans_z_derivative1', 'rot_x_derivative1', 'rot_y_derivative1',
                    'rot_z_derivative1']

    # convert rotation to radians (consistent with fmriprep)
    data2 = data.copy()

    radians_ = np.vectorize(math.radians)
    cols_to_convert = ["rot_x", "rot_y", "rot_z", "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1"]
    for col in cols_to_convert:
        data2[col] = radians_(data[col])

    # save output as tsv (fmriprep format)
    outpath = os.path.join(filepath.parent, "mc", "confounds_timeseries.tsv")
    data2.to_csv(outpath, sep='\t', header=True, index=False, float_format='%.5f')
    log.info("motion to fmriprep format: %s", outpath)


def motion_to_fsllike(filepath):
    cwd = os.getcwd()
    os.chdir(filepath.parent)
    os.makedirs(os.path.join(filepath.parent, "mc"), exist_ok=True)

    # reorder outputs and convert rotation units to radians
    cmd = """cat Movement_Regressors.txt | awk '{ print $4/57.29 " " $5/57.29 " " $6/57.29 " " $1 " " $2 " " $3}' > mc/prefiltered_func_data_mcf.par"""
    execute_shell(cmd, cwd=filepath.parent)
    log.info("motion to fsl format: %s", os.path.join(filepath.parent, "mc", "prefiltered_func_data_mcf.par"))
    os.chdir(cwd)


def copy_hcp_to_fmripreplike(root_dir, bidspath, source, dest):
    os.chdir(os.path.join(root_dir, bidspath))
    if os.path.islink(dest):
        os.unlink(dest)
    if not os.path.exists(source):
        log.warning("source file does not exist.")
        return
    log.info("copy... %s -> %s", source, os.path.join(bidspath, dest))
    shutil.copy(source, dest)


def symlink_hcp_to_fmripreplike(root_dir, bidspath, source, dest):
    os.chdir(os.path.join(root_dir, bidspath))
    if os.path.islink(dest):
        os.unlink(dest)
    if not os.path.exists(source):
        log.warning("source file does not exist.")
        return
    log.info("linking... %s -> %s", source, os.path.join(bidspath, dest))
    os.symlink(source, dest)


def main(root_dir, anlys_id, fw, dryrun= False, fix_trainingfile=''):
    """
    file mapper is used to arrange human connectome minimal preprocesisng pipeline (HCPPipe and ICAFIX) into a bids-derivateive format.
    All outputs are symbolically linked to reduce excess file storage costs. Always retain the original HCPPipe directory.
    Args:
        root_dir: Parent directory containing "HCPPipe" and "ICAFIX" results
        anlys_id: flywheel analysis id
        dryrun: test functionality without running
        fix_trainingfile: training file name used to differentiate ICA cleaned timeseries.

    Returns:

    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "hcp_mapper.json")) as f:
        data = json.load(f)

    # ------
    analysis = fw.get_analysis(anlys_id)
    lookup_table = build_lookup(analysis, fw)
    lookup_table["PIPELINE"] = "bids-hcp"

    # relevant when running after icafix
    lookup_table["TRAININGFILE"] = fix_trainingfile

    for k in data.keys():
        modality = data[k]

        bidspath = apply_lookup(modality["bidspath"], lookup_table)
        os.makedirs(os.path.join(root_dir, bidspath), exist_ok=True)

        # functional modalities are treated different because covariates are also generated from motion file
        if k == "func":
            # grab all functional bold acquisitions
            acqs = fw.get_session(analysis.parent["id"]).acquisitions.find('label=~^func-bold')

            for idx, x in enumerate(acqs):

                #skip sbref files
                if "sbref" in x.label.lower():
                    continue

                # build lookup table with acquisition information
                lookup_table_itr = lookup_table
                lookup_table_itr["ACQ"] = acqs[idx].label.replace("func-bold_", "")

                # create movement files to match fsl and fmriprep formats (not sure which is better to use generically)
                motion_file_pattern = os.path.join(str(root_dir), "HCPPipe/sub-{SUBJECT}/ses-{SESSION}/MNINonLinear/Results/ses-{SESSION}_{ACQ}_bold/Movement_Regressors.txt")
                motion_file = Path(apply_lookup(motion_file_pattern, lookup_table_itr))

                if not dryrun and motion_file.exists():
                    motion_to_fsllike(motion_file)
                    motion_to_fmripreplike(motion_file)

                # apply symbolic linking
                for s in modality["files"].keys():

                    source = apply_lookup(s, lookup_table_itr)
                    dest = apply_lookup(modality["files"][s], lookup_table_itr)

                    if not dryrun:
                        symlink_hcp_to_fmripreplike(root_dir, bidspath, source, dest)

        # all other acquisition modalities
        else:

            # apply symbolic linking
            for s in modality["files"].keys():

                source = apply_lookup(s, lookup_table)
                dest = apply_lookup(modality["files"][s], lookup_table)

                if not dryrun:
                    symlink_hcp_to_fmripreplike(root_dir, bidspath, source, dest)

