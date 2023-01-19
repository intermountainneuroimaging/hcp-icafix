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

# The run.py should be as minimal as possible.
# The gear is split up into 2 main components. The run.py file which is executed
# when the container runs. The run.py file then imports the rest of the gear as a
# module.

# os.chdir('/flywheel/v0/')
log = logging.getLogger(__name__)


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
