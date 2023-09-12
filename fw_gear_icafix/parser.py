"""Parser module to parse gear config.json."""
from typing import Tuple
from zipfile import ZipFile
from flywheel_gear_toolkit import GearToolkitContext
import os
import logging
import json
from fw_gear_icafix.main import execute_shell
import errorhandler
import pandas as pd

from pathlib import Path

log = logging.getLogger(__name__)

# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()


# This function mainly parses gear_context's config.json file and returns relevant
# inputs and options.
class GearArgs:
    def __init__(
            self, gtk_context: GearToolkitContext, env_file=None
    ):
        """[Summary]

        Returns:
            [type]: [description]
        """

        # setup enviornment for python system commands - all enviornment variables should be
        #    defined in the manifest and attached to the docker image via flywheel engine
        #    if a static ENV is desired, an env file can be generated and attached to project
        if env_file:
            with open(env_file, "r") as f:
                self.environ = json.load(f)
        else:
            self.environ = os.environ

        # pull input filepaths
        self.debug = gtk_context.config.get("debug")
        hcp_zipfile = gtk_context.get_input_path("hcp_zip")
        previous_results_zipfile = gtk_context.get_input_path("previous-results")
        hand_labeled_noise_file = gtk_context.get_input_path("hand-labeled-noise-components")

        if gtk_context.get_input_path("custom_training_file"):
            self.custom_training_file = gtk_context.get_input_path("custom_training_file")
            log.info("Custom training file path, %s", self.custom_training_file)

            # check config matches input...
            if gtk_context.config["TrainingFile"] != "User Defined":
                log.error(
                    "Custom training file passed as input, but TrainingFile option set to: %s. Not sure how to handle!",
                    self.config["TrainingFile"])

            # set training file config parameter to custom path
            gtk_context.config["TrainingFilePath"] = gtk_context.get_input_path("custom_training_file")
        else:
            gtk_context.config["TrainingFilePath"] = gtk_context.config["TrainingFile"]

        # pull config settings
        self.icafix = {
            "common_command": "",
            "params": ""
        }
        self.config = gtk_context.config
        self.gtk_context = gtk_context
        self.client = gtk_context.client
        self.work_dir = gtk_context.work_dir
        self.analysis_dir = Path(os.path.join(gtk_context.work_dir, self.gtk_context.destination["id"]))
        self.output_dir = gtk_context.output_dir
        self.dest_id = self.gtk_context.destination["id"]

        os.makedirs(self.analysis_dir, exist_ok=True)

        if hcp_zipfile and not previous_results_zipfile:
            log.info("Gear mode: Full Analysis")
            self.mode = "hcpfix"
            self.input_zip = hcp_zipfile

        elif not hcp_zipfile and previous_results_zipfile:
            log.info("Gear mode: Apply Denoising to Existing Dataset")
            self.mode = "fix cleanup"
            self.input_zip = previous_results_zipfile

            if hand_labeled_noise_file:
                self.mode = "hand labeled"
                self.input_zip = previous_results_zipfile

                if self.check_hand_label_spreadsheet(hand_labeled_noise_file):
                    analys = self.client.get_container(self.dest_id)
                    log.info("Using hand labeled noise for session %s", self.client.get_container(analys.parents["session"]).id)
                else:
                    log.error("Hand labeled noise not correctly organized")

        else:
            log.error("Ambiguous Inputs passed, unable to determine gear mode. Please try again.")
            return

        log.info("Inputs file path, %s", self.input_zip)
        self.unzip_inputs(self.input_zip)

        # pull original file structure
        orig = []
        for path, subdirs, files in os.walk(self.work_dir):
            for name in files:
                orig.append(os.path.join(path, name))

        self.unzipped_files = orig

    def unzip_inputs(self, zip_filename):
        """
        unzip_inputs unzips the contents of zipped gear output into the working
        directory.
        Args:
            self: The gear context object
                containing the 'gear_dict' dictionary attribute with key/value,
                'gear-dry-run': boolean to enact a dry run for debugging
            zip_filename (string): The file to be unzipped
        """
        rc = 0
        outpath = []
        # use linux "unzip" methods in shell in case symbolic links exist
        log.info("Unzipping file, %s", zip_filename)
        cmd = "unzip -o " + zip_filename + " -d " + str(self.analysis_dir)
        execute_shell(cmd, cwd=self.analysis_dir)

        # if unzipped directory is a destination id - move all outputs one level up
        with ZipFile(zip_filename, "r") as f:
            top = [item.split('/')[0] for item in f.namelist()]
            top1 = [item.split('/')[1] for item in f.namelist()]

        log.info("Done unzipping.")

        if len(top[0]) == 24:
            # directory starts with flywheel destination id - obscure this for now...
            cmd = "mv " + top[0] + '/* . ; rm -R ' + top[0]
            execute_shell(cmd, cwd=self.analysis_dir)
            for i in set(top1):
                outpath.append(os.path.join(self.analysis_dir, i))

            # get previous gear info
            self.preproc_gear = self.client.get(top[0])
        else:
            outpath = os.path.join(self.analysis_dir, top[0])

        if error_handler.fired:
            log.critical('Failure: exiting with code 1 due to logged errors')
            run_error = 1
            return run_error

        return rc, outpath

    def check_hand_label_spreadsheet(self, file):

        # pull flywheel sdk client
        analys = self.client.get_container(self.dest_id)

        # read spreadsheet
        df = pd.read_csv(file, sep='\t',dtype={'subject': str, 'session': str, 'flywheel session id': str, 'noiselabels': str})
        columns = df.columns

        if ("subject" in columns) and ("session" in columns) and ("acquisition" in columns) and (
                "noiselabels" in columns):

            # check that the correct subject and session exist
            subject = self.client.get_container(analys.parents["subject"])
            session = self.client.get_container(analys.parents["session"])

            itr = df.loc[(df['subject'] == subject.label) & (df['session'] == session.label) ]
            if not itr.empty:
                self.noiselabels = itr
                return True

        elif ("flywheel session id" in columns) and ("acquisition" in columns) and ("noiselabels" in columns):
            session = self.client.get_container(analys.parents["session"])
            itr = df.loc[df['flywheel session id'] == session.id]
            if not itr.empty:
                self.noiselabels = itr
                return True

        else:
            return False
