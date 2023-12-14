"""Main module."""

import logging
import os
import os.path as op
import subprocess as sp
import sys
import pandas as pd
import shutil
from collections import OrderedDict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tempfile import NamedTemporaryFile

from flywheel_gear_toolkit import GearToolkitContext
from flywheel_gear_toolkit.interfaces.command_line import (
    build_command_list,
    exec_command,
)

from fw_gear_icafix import metadata
import utils.filemapper as filemapper
from utils.report.report import report
from utils.zip_htmls import zip_htmls

log = logging.getLogger(__name__)


def run(gear_args):
    """Main script to launch hcp_fix using configuration set by user and zip results for storage.

    Returns:
        file: hcpfix_results.zip
    """
    log.info("This is the beginning of the run file")

    # add a loop here**
    for index, row in gear_args.files.iterrows():

        if not os.path.exists(row["preprocessed_files"]):
            log.fatal('Unable to locate correct functional file')
            sys.exit(1)

        # fetch dummy volumes (use hard coded value if present, else grab from mriqc)
        gear_args.config['AcqDummyVolumes'] = fetch_dummy_volumes(row["preprocessed_files"], gear_args)

        #remove specified numner of inital volumes
        temp_file = drop_initial_volumes(row, gear_args)

        # generate the hcp_fix command options from gear contex
        if gear_args.mode == "hcpfix":
            generate_icafix_command(row["preprocessed_files"], gear_args,"hcpfix")

            # execute hcp_fix command (inside this method checks for gear-dry-run)
            fix_command = execute(gear_args)

        elif gear_args.mode == "fix cleanup":
            # # first apply new training model
            icadir = searchfiles(os.path.join(row["taskdir"],"*hp*.ica"), dryrun=False, find_first=True)
            generate_icafix_command(icadir, gear_args, "classify")
            fix_command = execute(gear_args)

            # generate new clean dataset
            hpfile = os.path.basename(icadir).replace(".ica",".nii.gz")
            cmd = "ln -s ../" + hpfile + " " + "filtered_func_data.nii.gz"
            execute_shell(cmd, dryrun=gear_args.config["dry-run"], cwd=icadir)

            labels_file = searchfiles(os.path.join(row["taskdir"], "*hp*.ica", "fix4melview*.txt"), dryrun=False,
                                      find_recent=True)
            generate_icafix_command(labels_file, gear_args, "apply cleanup")
            fix_command = execute(gear_args)

            # unlink filtered func file
            cmd = "unlink " + os.path.join(icadir, "filtered_func_data.nii.gz")
            execute_shell(cmd, dryrun=gear_args.config["dry-run"])

            # create output cleaned directory
            cmd = "mv " + os.path.join(icadir,"filtered_func_data_clean.nii.gz") + " " + os.path.join(os.path.dirname(icadir),os.path.basename(icadir).replace(".ica","_"+Path(gear_args.config['TrainingFilePath']).stem+"_clean.nii.gz"))
            execute_shell(cmd, dryrun=gear_args.config["dry-run"], cwd=icadir)

        elif gear_args.mode == "hand labeled":

            # identify the hand labels for current acquisition
            handlabels = fetch_noise_labels(row["preprocessed_files"], gear_args)

            # write hand_labels_noise.txt
            icadir = searchfiles(os.path.join(row["taskdir"], "*hp*.ica"), dryrun=False, find_first=True)
            handlabels_file = op.join(icadir,"hand_label_noise.txt")
            with open(handlabels_file,'w') as fid:
                fid.write(" ,".join(handlabels))

            # generate new clean dataset
            hpfile = os.path.basename(icadir).replace(".ica", ".nii.gz")
            cmd = "ln -s ../" + hpfile + " " + "filtered_func_data.nii.gz"
            execute_shell(cmd, dryrun=gear_args.config["dry-run"], cwd=icadir)

            generate_icafix_command(handlabels_file, gear_args, "apply cleanup")
            fix_command = execute(gear_args)

            # unlink filtered func file
            cmd = "unlink " + os.path.join(icadir, "filtered_func_data.nii.gz")
            execute_shell(cmd, dryrun=gear_args.config["dry-run"])

            # create output cleaned directory
            cmd = "mv " + os.path.join(icadir, "filtered_func_data_clean.nii.gz") + " " + os.path.join(
                os.path.dirname(icadir), os.path.basename(icadir).replace(".ica", "_" + "handlabel" + "_clean.nii.gz"))
            execute_shell(cmd, dryrun=gear_args.config["dry-run"], cwd=icadir)


        # add dummy vols back to keep output same as input:
        ica_files = searchfiles(os.path.join(row["taskdir"],"*hp*.nii.gz"), dryrun=False)
        for ica_file in ica_files:
            cleanup_volume_files(ica_file, temp_file, gear_args)

        if row["surface_files"]:
            ica_files_surface = searchfiles(os.path.join(row["taskdir"],"*Atlas*hp*.dtseries.nii"), dryrun=False)
            for ica_file in ica_files_surface:
                cleanup_surface_files(ica_file, temp_file, gear_args)

        # remove all tmp files
        cmd = "rm -Rf tmp*"
        execute_shell(cmd, dryrun=gear_args.config["dry-run"], cwd=row["taskdir"])

        # store metadata at the acquisition level
        labels_file = searchfiles(os.path.join(row["taskdir"],"*hp*.ica","fix4melview*.txt"), dryrun=False, find_recent=True)
        icstats_file = searchfiles(os.path.join(row["taskdir"], "*hp*.ica", "filtered_func_data.ica","melodic_ICstats"), dryrun=False,
                                  find_first=True)
        metrics = store_metadata(labels_file, icstats_file, row["preprocessed_files"], gear_args)

        # generate report for ica classification
        reportdir = report(row["taskdir"],fix_command)

        zip_htmls(gear_args.output_dir, gear_args.dest_id,reportdir)

    # cleanup gear and store outputs and logs...
    cleanup(gear_args)

    return 0


def check_input_files(workdir, suffix):
    # Look for tasks in HCP preprocessed file list
    taskdirs = sp.Popen(
        "ls -d " + workdir.absolute().as_posix() + "/HCPPipe/sub-*/ses-*/MNINonLinear/Results/*task*", shell=True, stdout=sp.PIPE,
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
        matches.append("/" + os.path.join(*pp) + suffix)

    return matches


def fetch_dummy_volumes(taskname, context):
    # Function generates number of dummy volumes from config or mriqc stored IQMs
    if context.config["DropNonSteadyState"] is False:
        return 0

    bids_name = fetch_acq_name(taskname)

    acq, f = metadata.find_matching_acq(bids_name, context)

    if "DummyVolumes" in context.config:
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        log.info("Set by user....Using %s dummy volumes", context.config['DummyVolumes'])
        return context.config['DummyVolumes']

    if f:
        IQMs = f.info["IQM"]
        log.info("Extracting dummy volumes from acquisition: %s", acq.label)
        if "dummy_trs_custom" in IQMs:
            log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs_custom"])
            return IQMs["dummy_trs_custom"]
        else:
            log.info("Set by mriqc....Using %s dummy volumes", IQMs["dummy_trs"])
            return IQMs["dummy_trs"]

    # if we reach this point there is a problem! return error and exit
    log.error("Option to drop non-steady state volumes selected, no value passed or could be interpreted from session metadata. Quitting...")


def fetch_acq_name(taskname):
    taskname_split = taskname.split("/")[-1].split(".")[0].split("_")
    index = taskname_split.index("bold")
    bids_name = "_".join(taskname_split[1:index])
    return bids_name


def drop_initial_volumes(input_files, context):
    # if dummy volumes are passed in config, remove them from working file before running ica
    # ** stitch it back together at the end**

    if context.config['AcqDummyVolumes'] == 0:
        # do nothing
        return

    log.info("Removing dummy volumes for ICA-FIX component creation...")
    log.info("Removing %s volumes. ", str(context.config['AcqDummyVolumes']))
    dummyvars = context.config['AcqDummyVolumes']

    f = NamedTemporaryFile(delete=False, dir=input_files["taskdir"])
    store_original_filename = f.name + "_" + os.path.basename(input_files["preprocessed_files"])
    shutil.copyfile(input_files["preprocessed_files"], store_original_filename)

    # create trimmed nifti
    cmd = os.environ["FSLDIR"] + "/bin/fslroi " + store_original_filename + " " + input_files["preprocessed_files"] + " " + str(dummyvars) + " -1"
    execute_shell(cmd, dryrun=context.config["dry-run"], cwd=input_files["taskdir"])

    # create trimmed Movement_Regressors.txt
    if input_files["motion_files"]:
        store_original_motionfile = f.name + "_" + os.path.basename(input_files["motion_files"])
        shutil.copyfile(input_files["motion_files"], store_original_motionfile)

        cmd = "tail -n +" + str(dummyvars+1) + " < " + store_original_motionfile + " > " + input_files["motion_files"]
        execute_shell(cmd, dryrun=context.config["dry-run"], cwd=os.path.dirname(input_files["preprocessed_files"]))

    # create trimmed cifti (there may be a more direct way to do this...)
    if input_files["surface_files"]:
        store_original_ciftifile = f.name + "_" + os.path.basename(input_files["surface_files"])

        # retrieve step-interval-size (TR)
        cmd = os.environ["FSL_FIX_WBC"] + " -file-information " + input_files["surface_files"] + " -only-step-interval "
        stepsize = execute_shell(cmd, dryrun=context.config["dry-run"], cwd=input_files["taskdir"])

        # make copy of original file
        shutil.copyfile(input_files["surface_files"], store_original_ciftifile)

        # convert cifti to nifti
        cmd = os.environ["FSL_FIX_WBC"] + " -cifti-convert -to-nifti " + input_files["surface_files"] + " tmp_cifti2nfiti.nii.gz"
        execute_shell(cmd, dryrun=context.config["dry-run"], cwd=input_files["taskdir"])

        # trim cifti2nifti file
        cmd = os.environ["FSLDIR"] + "/bin/fslroi " + "tmp_cifti2nfiti.nii.gz" + " " + "tmp_cifti2nfiti_trimmed.nii.gz" + " " + str(
            dummyvars) + " -1"
        execute_shell(cmd, dryrun=context.config["dry-run"], cwd=input_files["taskdir"])

        # convert back to cifti
        cmd = os.environ["FSL_FIX_WBC"] + " -cifti-convert -from-nifti " + "tmp_cifti2nfiti_trimmed.nii.gz" + " " + store_original_ciftifile + " " + input_files["surface_files"] + " -reset-timepoints " + str(stepsize) + " 0"
        execute_shell(cmd, dryrun=context.config["dry-run"], cwd=input_files["taskdir"])

    return store_original_filename


def cleanup_volume_files(ica_file, temp_file, context):
    # stitch dummy volumes back in at the end (keep total scan length the same!)

    if context.config['AcqDummyVolumes'] == 0:
        # do nothing
        return

    log.info("Adding dummy frames back to ICA cleaned output %s!", os.path.basename(ica_file))
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



def generate_icafix_command(input_file, context, stage):
    training_file = context.config['TrainingFilePath']
    highpass = context.config['HighPassFilter']
    mot_reg = context.config['do_motion_regression']
    fix_threshold = context.config['FixThreshold']
    del_intermediates = context.config['DeleteIntermediates']

    if stage == "hcpfix":
        context.icafix["params"] = OrderedDict(
            [('input', input_file), ('highpass', highpass), ('mot_reg', str(mot_reg).upper()),
             ('training_file', training_file), ('fix_threshold', fix_threshold),
             ('del_intermediate', str(del_intermediates).upper())])
        context.icafix["common_command"]= "/opt/HCP-Pipelines/ICAFIX/hcp_fix"
    if stage == "classify":
        context.icafix["params"] = OrderedDict(
            [('flag', '-c'), ('input', input_file), ('training_file', training_file), ('fix_threshold', fix_threshold)]
        )
        context.icafix["common_command"] = "/opt/fix/fix"

    if stage == "apply cleanup":
        context.icafix["params"] = OrderedDict(
            [('flag', '-a'), ('input', input_file)]
        )
        if mot_reg:
            context.icafix["params"].update({'mot_reg', "-m "+str(mot_reg).upper()})
            if highpass:
                context.icafix["params"].update({'highpass', "-h " + highpass})

        context.icafix["common_command"] = "/opt/fix/fix"


def fetch_noise_labels(taskname, context):

    taskname_split=taskname.split("/")[-1].split(".")[0].split("_")
    index = taskname_split.index("bold")
    bids_name = "_".join(taskname_split[1:index])

    acq, f = metadata.find_matching_acq(bids_name, context)

    row = context.noiselabels.loc[context.noiselabels['acquisition'] == acq.label]
    noiselabels = row["noiselabels"]

    return noiselabels


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
        log.info("\n %s", stdout)
        log.info("\n %s", stderr)

        if "error" in stderr.lower() or returncode != 0:
            gear_args["errors"].append(
                {"message": "hcp_fix failed. Check log", "exception": stderr}
            )
    except Exception as e:
        log.exception(e)
        log.fatal('Unable to run hcp_fix')
        sys.exit(1)
    return command


def cleanup(gear_args: GearToolkitContext):

    # create bids-derivative naming scheme
    if gear_args.mode == "fix cleanup":
        trainingname = "_"+Path(gear_args.config['TrainingFilePath']).stem
        filemapper.main(gear_args.analysis_dir, gear_args.dest_id, gear_args.client, fix_trainingfile=trainingname)
    elif gear_args.mode == "hand labeled":
        trainingname = "_" + "handlabel"
        filemapper.main(gear_args.analysis_dir, gear_args.dest_id, gear_args.client, fix_trainingfile=trainingname)
    else:
        filemapper.main(gear_args.analysis_dir, gear_args.dest_id, gear_args.client)

    # locate new files from analysis
    new = []
    for path, subdirs, files in os.walk(gear_args.work_dir):
        for name in files:
            if "tmp" in name or "temp" in name:
                continue  ## ignore temporary files...
            new.append(os.path.join(path, name))

    outfiles = [x for x in new if x not in set(gear_args.unzipped_files)]

    outfiles_rel = [i.replace(str(gear_args.work_dir) + os.sep, '') for i in outfiles]

    with open(op.join(str(gear_args.work_dir),"files.txt"), 'w') as f:
        f.write("\n".join(map(str, outfiles_rel)))

    # use new list for final output
    # zip output files
    os.chdir(gear_args.work_dir)
    output_zipname = gear_args.output_dir.absolute().as_posix() + "/hcpfix_results_" + \
                     gear_args.gtk_context.destination["id"] + ".zip"

    # NEW method to zip working directory using 'zip --symlinks -r outzip.zip data/'
    cmd = "zip --symlinks -r " + output_zipname + " -@ < files.txt "
    execute_shell(cmd, cwd=str(gear_args.work_dir))

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

    taskname_split = taskname.split("/")[-1].split(".")[0].split("_")
    index = taskname_split.index("bold")
    bids_name = "_".join(taskname_split[1:index])
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


def searchfiles(path, dryrun=False, find_first=False, find_recent=False) -> list[str]:

    if find_recent:
        options=" -dt "
    else:
        options=" -d "
    cmd = "ls" + options + path

    log.debug("\n %s", cmd)

    if not dryrun:
        terminal = sp.Popen(
            cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
        )
        stdout, stderr = terminal.communicate()
        log.debug("\n %s", stdout)
        log.debug("\n %s", stderr)

        files = stdout.strip("\n").split("\n")

        if find_first or find_recent:
            files = files[0]

        return files


