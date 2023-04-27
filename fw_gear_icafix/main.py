"""Main module."""

import logging
import os
import os.path as op
import subprocess as sp
import sys
import pandas as pd
import shutil
from collections import OrderedDict
from zipfile import ZIP_DEFLATED, ZipFile

from tempfile import NamedTemporaryFile

from flywheel_gear_toolkit import GearToolkitContext
from flywheel_gear_toolkit.interfaces.command_line import (
    build_command_list,
    exec_command,
)

from fw_gear_icafix import metadata

log = logging.getLogger(__name__)


def run(gear_args):
    """Main script to launch hcp_fix using configuration set by user and zip results for storage.

    Returns:
        file: hcpfix_results.zip
    """
    log.info("This is the beginning of the run file")

    # pull list of func files for analysis (at this time only setup for single run mode **not multirun**)
    try:
        funcfiles = check_input_files(gear_args.work_dir, gear_args.hcp_zipfile)
    except Exception as e:
        log.exception(e)
        log.fatal('Unable to locate correct functional file')
        sys.exit(1)

    # add a loop here**
    for f in funcfiles:

        # fetch dummy volumes (use hard coded value if present, else grab from mriqc)
        gear_args.config['AcqDummyVolumes'] = fetch_dummy_volumes(f, gear_args)

        # #remove specified numner of inital volumes
        # temp_file = drop_initial_volumes(f, gear_args)
        #
        # # generate the hcp_fix command options from gear contex
        # generate_ica_command(f, gear_args)
        #
        # # execute hcp_fix command (inside this method checks for gear-dry-run)
        # execute(gear_args)
        #
        # # add dummy vols back to keep output same as input:
        # ica_files = searchfiles(os.path.join(os.path.dirname(f),"*hp*.nii.gz"), dryrun=False)
        # for ica_file in ica_files:
        #     cleanup_volume_files(ica_file, temp_file, gear_args)
        #
        # ica_files_surface = searchfiles(os.path.join(os.path.dirname(f),"*Atlas*hp*.dtseries.nii"), dryrun=False)
        # for ica_file in ica_files_surface:
        #     cleanup_surface_files(ica_file, temp_file, gear_args)
        #
        # # remove all tmp files
        # cmd = "rm -Rf tmp*"
        # execute_shell(cmd, dryrun=gear_args.config["dry-run"], cwd=os.path.dirname(f))

        # store metadata at the acquisition level
        labels_file = searchfiles(os.path.join(os.path.dirname(f),"*hp*.ica","fix4melview*.txt"), dryrun=False, find_first=True)
        icstats_file = searchfiles(os.path.join(os.path.dirname(f), "*hp*.ica", "filtered_func_data.ica","melodic_ICstats"), dryrun=False,
                                  find_first=True)
        metrics = store_metadata(labels_file, icstats_file, f, gear_args)


    # cleanup gear and store outputs and logs...
    cleanup(gear_args)

    return 0


def check_input_files(workdir, zip_files):
    # Look for tasks in HCP preprocessed file list
    taskdirs = sp.Popen(
        "ls -d " + workdir.absolute().as_posix() + "/*/HCPPipe/sub-*/ses-*/MNINonLinear/Results/*task*", shell=True, stdout=sp.PIPE,
        stderr=sp.PIPE, universal_newlines=True
    )

    stdout, _ = taskdirs.communicate()
    log.info("Running HCP Fix for the following directories: ")
    log.info("\n %s", stdout)

    # quick manipulation to pull the task name (same as preprocessed image name)
    matches = []
    for f in stdout.splitlines():
        pp = f.split('/')
        pp.append(pp[-1])
        matches.append("/" + os.path.join(*pp) + ".nii.gz")

    return matches


def fetch_dummy_volumes(taskname, context):
    # Function generates number of dummy volumes from config or mriqc stored IQMs
    if context.config["DropNonSteadyState"] is False:
        return 0

    bids_name = "_".join(taskname.split("/")[-1].split(".")[0].split("_")[1:-1])

    acq, f = metadata.find_matching_acq(bids_name, context)

    if "DummyVolumes" in context.config:
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        log.info("Set by user....Using %s dummy volumes", context.config['DummyVolumes'])
        return context.config['DummyVolumes']

    if f:
        IQMs = f.info["IQM"]
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs"])

        return IQMs["dummy_trs"]

    # if we reach this point there is a problem! return error and exit
    log.error("Option to drop non-steady state volumes selected, no value passed or could be interpreted from session metadata. Quitting...")


def drop_initial_volumes(input_file, context):
    # if dummy volumes are passed in config, remove them from working file before running ica
    # ** stitch it back together at the end**

    if context.config['AcqDummyVolumes'] == 0:
        # do nothing
        return

    log.info("Removing dummy volumes for ICA-FIX component creation...")
    log.info("Removing %s volumes. ", str(context.config['AcqDummyVolumes']))
    dummyvars = context.config['AcqDummyVolumes']

    f = NamedTemporaryFile(delete=False, dir=os.path.dirname(input_file))
    store_original_filename = f.name + "_" + os.path.basename(input_file)
    shutil.copyfile(input_file, store_original_filename)

    # create trimmed nifti
    cmd = os.environ["FSLDIR"] + "/bin/fslroi " + store_original_filename + " " + input_file + " " + str(dummyvars) + " -1"
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(input_file))

    # create trimmed Movement_Regressors.txt
    motion_file = os.path.join(os.path.dirname(input_file),"Movement_Regressors.txt")
    store_original_motionfile = f.name + "_" + os.path.basename(motion_file)
    shutil.copyfile(motion_file, store_original_motionfile)

    cmd = "tail -n +" + str(dummyvars+1) + " < " + store_original_motionfile + " > " + motion_file
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(input_file))

    # create trimmed cifti (there may be a more direct way to do this...)
    cifti_file = input_file.replace(".nii.gz", "_Atlas.dtseries.nii")
    store_original_ciftifile = f.name + "_" + os.path.basename(cifti_file)

    # retrieve step-interval-size (TR)
    cmd = os.environ["FSL_FIX_WBC"] + " -file-information " + cifti_file + " -only-step-interval "
    stepsize = execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(input_file))

    # make copy of original file
    shutil.copyfile(cifti_file, store_original_ciftifile)

    # convert cifti to nifti
    cmd = os.environ["FSL_FIX_WBC"] + " -cifti-convert -to-nifti " + cifti_file + " tmp_cifti2nfiti.nii.gz"
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(input_file))

    # trim cifti2nifti file
    cmd = os.environ["FSLDIR"] + "/bin/fslroi " + "tmp_cifti2nfiti.nii.gz" + " " + "tmp_cifti2nfiti_trimmed.nii.gz" + " " + str(
        dummyvars) + " -1"
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(input_file))

    # convert back to cifti
    cmd = os.environ["FSL_FIX_WBC"] + " -cifti-convert -from-nifti " + "tmp_cifti2nfiti_trimmed.nii.gz" + " " + store_original_ciftifile + " " + cifti_file + " -reset-timepoints " + str(stepsize) + " 0"
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(input_file))

    return store_original_filename


def cleanup_volume_files(ica_file, temp_file, context):
    # stitch dummy volumes back in at the end (keep total scan length the same!)

    if context.config['AcqDummyVolumes'] == 0:
        # do nothing
        return

    log.info("Adding dummy frames back to ICA cleaned output%s!", os.path.basename(ica_file))
    dummyvars = context.config['AcqDummyVolumes']

    # get the original dummy volumes
    f = NamedTemporaryFile(delete=False, dir=os.path.dirname(temp_file))
    dummyvols_filename = f.name + "_" + os.path.basename(temp_file)
    cmd = os.environ["FSLDIR"] + "/bin/fslroi " + temp_file + " " + dummyvols_filename + " 0 " + str(dummyvars)
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.getcwd())

    # add original dummy vols back to cleaned (and filtered ica outputs)
    cmd = os.environ["FSLDIR"] + "/bin/fslmerge -t " + ica_file + " " + dummyvols_filename + " " + ica_file
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.getcwd())


def cleanup_surface_files(cifti_file, temp_file, context):
    # stitch dummy volumes back in at the end (keep total scan length the same!)

    if context.config['AcqDummyVolumes'] == 0:
        # do nothing
        return

    log.info("Adding dummy frames back to ICA cleaned output %s!", os.path.basename(cifti_file))
    dummyvars = context.config['AcqDummyVolumes']

    # get the original dummy volumes
    f = NamedTemporaryFile(delete=False, dir=os.path.dirname(temp_file))
    temp_cifti2nifti_filename = f.name + "_" + os.path.basename(cifti_file)

    # pull the original initial surface frames
    cmd = os.environ["FSLDIR"] + "/bin/fslroi " + "tmp_cifti2nfiti.nii.gz" + " " + "tmp_cifti2nfiti_initalvols.nii.gz 0 " + str(dummyvars)
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(cifti_file))

    # merge original frames with cleaned cifti2nifti
    cmd = os.environ["FSL_FIX_WBC"] + " -cifti-convert -to-nifti " + cifti_file + " " + temp_cifti2nifti_filename
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(cifti_file))

    # add original dummy frames back to cleaned (and filtered ica outputs)
    cmd = os.environ["FSLDIR"] + "/bin/fslmerge -t " + "tmp_output" + " " + "tmp_cifti2nfiti_initalvols.nii.gz" + " " + temp_cifti2nifti_filename
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(cifti_file))

    # get original stepsize
    cmd = os.environ["FSL_FIX_WBC"] + " -file-information " + cifti_file + " -only-step-interval "
    stepsize = execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(cifti_file))

    # finally, return to cifti format
    cmd = os.environ["FSL_FIX_WBC"] + " -cifti-convert -from-nifti " + "tmp_output.nii.gz" + " " + cifti_file + " " + cifti_file + " -reset-timepoints " + str(stepsize) + " 0"
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(cifti_file))



def generate_ica_command(input_file, context):
    training_file = context.config['TrainingFile']
    highpass = context.config['HighPassFilter']
    mot_reg = context.config['do_motion_regression']
    fix_threshold = context.config['FixThreshold']
    del_intermediates = context.config['DeleteIntermediates']

    context.icafix["params"] = OrderedDict(
        [('input', input_file), ('highpass', highpass), ('mot_reg', str(mot_reg).upper()),
         ('training_file', training_file), ('fix_threshold', fix_threshold),
         ('del_intermediate', str(del_intermediates).upper())])


def execute(gear_args):
    command = []
    command.append(gear_args.icafix["common_command"])
    command = build_command_list(command, gear_args.icafix["params"], include_keys=False)

    stdout_msg = (
            "hcp_fix logs (stdout, stderr) will be available "
            + 'in the file "pipeline_logs.zip" upon completion.'
    )
    if gear_args.config["dry-run"]:
        log.info("hcp_fix command:\n{command}")
    try:
        stdout, stderr, returncode = exec_command(
            command,
            dry_run=gear_args.config["dry-run"],
            environ=gear_args.environ,
            stdout_msg=stdout_msg,
        )
        if "error" in stderr.lower() or returncode != 0:
            gear_args["errors"].append(
                {"message": "hcp_fix failed. Check log", "exception": stderr}
            )
    except Exception as e:
        if gear_args.config["dry_run"]:
            # Error thrown due to non-iterable stdout, stderr, returncode
            pass
        else:
            log.exception(e)
            log.fatal('Unable to run hcp_fix')
            sys.exit(1)


def cleanup(gear_args: GearToolkitContext):
    """
    Execute a series of steps to store outputs on the proper containers.

    Args:
        gear_args: The gear context object
            containing the 'gear_dict' dictionary attribute with keys/values
            utilized in the called helper functions.
    """
    # look for output files...
    # Following HCP directory structure, input fMRI should be preprocessed and in the MNINonLinear/Results directory
    # Look for tasks in HCP preprocessed file list
    searchfiles = sp.Popen(
        "cd " + gear_args.work_dir.absolute().as_posix() + "; ls -d  */HCPPipe/sub-*/ses-*/MNINonLinear/Results/*task*/*task*hp*nii*", shell=True,
        stdout=sp.PIPE,
        stderr=sp.PIPE, universal_newlines=True
    )
    stdout, _ = searchfiles.communicate()

    # quick manipulation to pull the task name (same as preprocessed image name)
    outfiles = []
    for f in stdout.splitlines():
        outfiles.append(f)

    # check if intermediate files should be saved (recommended)
    if not gear_args.config["DeleteIntermediates"]:
        # add the ica folders to zipped output...
        searchfiles = sp.Popen(
            "cd " + gear_args.work_dir.absolute().as_posix() + "; ls -d */HCPPipe/sub-*/ses-*/MNINonLinear/Results/*task*/*task*.ica", shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE, universal_newlines=True
        )
        stdout, _ = searchfiles.communicate()

        # quick manipulation to pull the task name (same as preprocessed image name)
        for f in stdout.splitlines():
            outfiles.append(f)
    newline = "\n"
    log.info("The following output files will be saved: \n %s", newline.join(outfiles))

    # zip output files
    os.chdir(gear_args.work_dir)
    output_zipname = gear_args.output_dir.absolute().as_posix() + "/hcpfix_results_"+gear_args.gtk_context.destination["id"]+".zip"
    outzip = ZipFile(output_zipname, "w", ZIP_DEFLATED)

    for fl in outfiles:
        if os.path.isdir(fl):
            for root, _, files in os.walk(fl):
                for ff in files:
                    ff_path = op.join(root, ff)
                    outzip.write(ff_path)
        else:
            outzip.write(fl)

    outzip.close()

    # log final results size
    os.chdir(gear_args.output_dir)
    duResults = sp.Popen(
        "du -hs *", shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
    )
    stdout, _ = duResults.communicate()
    log.info("\n %s", stdout)

    return 0


def store_metadata(labels_file, icstats_file, taskname, context):
    # after successful completion of the gear, generate simple metadata on ica component classification
    # metadata:
    #   classification [total, signal, unknown, unclassified noise]
    #      : prc_explained_variance
    #      : prc_total_variance
    #      : components

    df1 = metadata.ingest_labels(labels_file)
    df2 = metadata.ingest_icstats(icstats_file)

    dfstats = pd.concat([df1, df2], axis=1)

    metrics = metadata.report_metrics(dfstats)

    metrics = metrics.set_index(metrics.columns[0])

    info_obj = metrics.to_dict()
    info_obj["job"] = context.gtk_context.destination["id"]

    trainingfile = context.config['TrainingFile'].split(".")[0]
    info_obj = {trainingfile: info_obj}

    bids_name = "_".join(taskname.split("/")[-1].split(".")[0].split("_")[1:-1])
    acq, fw_file = metadata.find_matching_acq(bids_name, context)

    if fw_file:
        # B/c of 'info' being a flywheel.models.info_list_output.InfoListOutput,
        # deep_merge in `update_file` doesn't work.

        fw_file.update_info({"ICAFIX": info_obj})
        # log.info(f"Updated metadata file: {fw_file.name}")



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
        log.debug("\n %s", stdout)
        log.debug("\n %s", stderr)

        return stdout.strip('\n')


def searchfiles(path, dryrun=False, find_first=False) -> list[str]:
    cmd = "ls -d " + path

    log.debug("\n %s", cmd)

    if not dryrun:
        terminal = sp.Popen(
            cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
        )
        stdout, stderr = terminal.communicate()
        log.debug("\n %s", stdout)
        log.debug("\n %s", stderr)

        files = stdout.strip("\n").split("\n")

        if find_first:
            files = files[0]

        return files
