#!/usr/bin/env python
"""The run script"""
import logging
import sys, os
import shutil
from pathlib import Path
from flywheel_gear_toolkit import GearToolkitContext

# This design with a separate main and parser module
# allows the gear to be publishable and the main interfaces
# to it can then be imported in another project which enables
# chaining multiple gears together.
from fw_gear_icafix.main import run
from fw_gear_icafix.parser import GearArgs
from utils.singularity import run_in_tmp_dir
import errorhandler
# The run.py should be as minimal as possible.
# The gear is split up into 2 main components. The run.py file which is executed
# when the container runs. The run.py file then imports the rest of the gear as a
# module.


# Track if message gets logged with severity of error or greater
error_handler = errorhandler.ErrorHandler()


os.chdir('/flywheel/v0/')
log = logging.getLogger(__name__)

## TODO - Feature updates
#  1. Accept single HCP Pipelines result dir (001_hcp.zip)  (done)
#  2. Accept custom training file (project level upload)    (done)
#  3. Pull non-steady state detector from mriqc metadata (make bool) -- duplicate this logic in feat (done)
#  4. retain all hp2000 files (highpass files) (done)
#  5. remove all temp files (done)
#  6. add metadata to output variance explained, count? (done)
#  7. accept already generated ica directory (done)
#  8. make bids compatible derivatives (done)
#  9. accept hand label lists and apply denoising
#  10. multirun mode!


def main(context: GearToolkitContext):  # pragma: no cover
    """Parses config and run"""

    FWV0 = Path.cwd()
    log.info("Running gear in %s", FWV0)
    log.info("output_dir is %s", context.output_dir)
    log.info("work_dir is %s", context.work_dir)

    # initiat return_code
    return_code = 0

    #Parse inputs to extract the args, kwargs from the context
    # (e.g. config.json).
    log.info("Populating gear arguments")
    gear_args = GearArgs(context)

    if error_handler.fired:
        log.critical('Failure: exiting with code 1 due to logged errors')
        run_error = 1
        return run_error

    # Pass the args, kwargs to fw_gear_skeleton.main.run function to execute
    # the main functionality of the gear.
    e_code = run(gear_args)

    # Exit the python script (and thus the container) with the exit
    # code returned by example_gear.main.run function.
    # sys.exit(e_code)
    return_code = e_code
    return return_code

# Only execute if file is run as main, not when imported by another module
if __name__ == "__main__":  # pragma: no cover
    # Get access to gear config, inputs, and sdk client if enabled.
    with GearToolkitContext() as gear_context:
        scratch_dir = run_in_tmp_dir(gear_context.config["gear-writable-dir"])
    # Has to be instantiated twice here, since parent directories might have
    # changed
    with GearToolkitContext() as gear_context:

        # # Initialize logging, set logging level based on `debug` configuration
        # # key in gear config.
        gear_context.init_logging()

        # Pass the gear context into main function defined above.
        return_code = main(gear_context)

    # clean up (might be necessary when running in a shared computing environment)
    if scratch_dir:
        log.debug("Removing scratch directory")
        for thing in scratch_dir.glob("*"):
            if thing.is_symlink():
                thing.unlink()  # don't remove anything links point to
                log.debug("unlinked %s", thing.name)
        shutil.rmtree(scratch_dir)
        log.debug("Removed %s", scratch_dir)

    sys.exit(return_code)
