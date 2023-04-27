"""Parser module to parse gear config.json."""
from typing import Tuple
from zipfile import ZipFile
from flywheel_gear_toolkit import GearToolkitContext
import os
import logging
import json
from fw_gear_icafix.main import execute_shell
from pathlib import Path
log = logging.getLogger(__name__)

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
        self.hcp_zipfile = gtk_context.get_input_path("hcp_zip")
        log.info("Inputs file path, %s", self.hcp_zipfile)
        if gtk_context.get_input_path("custom_training_file"):
            self.custom_training_file = gtk_context.get_input_path("custom_training_file")
            log.info("Custom training file path, %s", self.custom_training_file)

            # check config matches input...
            if self.config["TrainingFile"] != "User Defined":
                log.error("Custom training file passed as input, but TrainingFile option set to: %s. Not sure how to handle!", self.config["TrainingFile"])

            # set training file config parameter to custom path
            self.config["TrainingFile"] = gtk_context.get_input_path("custom_training_file")

        # pull config settings
        self.icafix = {
            "common_command": "/opt/HCP-Pipelines/ICAFIX/hcp_fix",
            "params": ""
        }
        self.config = gtk_context.config
        self.gtk_context = gtk_context
        self.work_dir = gtk_context.work_dir
        self.output_dir = gtk_context.output_dir

        with ZipFile(self.hcp_zipfile, "r") as f:
            hcp_anlys_id = [item.split('/')[0] for item in f.namelist()]

        # unzip HCPpipeline files
        # self.unzip_hcp(self.hcp_zipfile)

        # get current analysis (new) destination id
        dest_id = self.gtk_context.destination["id"]

        # move HCP results one level up..
        cmd = "mkdir " + str(dest_id) + "; mv "+hcp_anlys_id[0]+"/* "+str(dest_id) + "; rm -rf "+hcp_anlys_id[0]
        execute_shell(cmd, dryrun=False, cwd=gtk_context.work_dir)


    def unzip_hcp(self, zip_filename):
        """
        unzip_hcp unzips the contents of zipped gear output into the working
        directory.  All of the files extracted are tracked from the
        above process_hcp_zip.
        Args:
            self: The gear context object
                containing the 'gear_dict' dictionary attribute with key/value,
                'dry-run': boolean to enact a dry run for debugging
            zip_filename (string): The file to be unzipped
n        """
        hcp_zip = ZipFile(zip_filename, "r")
        log.info("Unzipping hcp outputs, %s", zip_filename)
        if not self.config.get("dry_run"):
            hcp_zip.extractall(self.work_dir)
            log.debug(f'Unzipped the file to {self.work_dir}')