import io
import logging
import os
import socket
from pathlib import Path

import click
import colorama
from click import option
from colored import attr, fg, stylize
from dotenv import find_dotenv, load_dotenv
from phd_qmclib.util.win32 import enable_virtual_terminal_processing
from phd_qmclib.utils import now

from . import config
from .dmc_exec import (
    CLIApp, config as dmc_exec_config
)

# NOTE 1: We can use enable_virtual_terminal_processing function
#   from phd_qmclib.util.win32 module to enable ANSI escape codes on
#   Windows.
enable_virtual_terminal_processing()

# Load environment variables
load_dotenv(find_dotenv(), verbose=True)

BANNER = '''
#####################################################################

    Diffusion Monte Carlo simulation for an interacting Bose gas
    within multi-rods with a contact interaction.

#####################################################################
'''

# TODO: Improve this pattern.
BASE_PATH = os.path.dirname(os.path.abspath(__file__))


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


config_path_type = click.Path(exists=True)
output_path_type = click.Path(exists=False)


# noinspection PyUnusedLocal
def gen_filename(ext='yml') -> str:
    """

    :param ext:
    :return:
    """
    # Current time.
    now_ = now()
    date_id = now_.strftime('%Y-%m-%d')
    time_id = now_.strftime('%H-%M-%S.%fus')
    hostname = socket.gethostname()
    hist_filename = \
        f"mrbp-dmc-conf_{date_id}_{time_id}@{hostname}.yml"

    return hist_filename


@click.group()
def cli():
    """CLI to execute a DMC calculation for a 1D Bose Gas in a Multi-Rod
    Lattice.
    """
    pass


@cli.command()
@click.argument('template', type=config_path_type)
@click.option('-o', '--output', type=output_path_type, default=None)
@click.option('-r', '--replace', is_flag=True, default=False)
def proc_template(template: str,
                  output: str = None,
                  replace: bool = False):
    """Process a template and generates a configuration file."""
    # Disable colorama processing (again...).
    colorama.deinit()

    tpl_path = Path(template).absolute()

    assert tpl_path.is_file()

    if output is None:
        output_path = Path('.').absolute()
    else:
        output_path = Path(output).absolute()

    if output_path.is_dir():
        output_path /= gen_filename()

    styled_path = stylize(f"{tpl_path}", attr("bold"))
    print(f"Template path:")
    print(f"    {styled_path}")
    styled_output = stylize(output_path, attr("bold"))
    print("Path to output configuration file:")
    print(f"    {styled_output}")

    if output_path.exists():
        if not replace:
            raise IOError(f"file {output_path} exists")
        else:
            output_msg = "W: Output file already exists. It will be replaced"
            styled_output = stylize(output_msg, fg(208) + attr("bold"))
            print(f"{styled_output}")

    # Create the output directory.
    os.makedirs(output_path.parent, exist_ok=True)

    config_template = config.Template(tpl_path)
    config_template.save(output_path)

    print(f"Output file successfully saved")


# noinspection PyUnusedLocal
@cli.command()
@click.argument('config-path', type=config_path_type)
@click.option('-y', '--assume-yes', is_flag=True, default=False)
@click.option('-v', '--verbose', is_flag=True, default=False)
@click.option('-S', '--silent', is_flag=True, default=False)
@option('--dry-run', is_flag=True, default=False)
def start(config_path: str,
          assume_yes: bool,
          verbose: bool,
          silent: bool,
          dry_run: bool):
    """Start a DMC simulation"""
    # Disable colorama processing (again...).
    colorama.deinit()

    print(BANNER)

    config_path = Path(config_path).absolute()

    #
    assert config_path.is_file()

    # Create the CLI object and validate.
    config_data = dmc_exec_config.loader.load(config_path)
    proc_cli = CLIApp.from_config(config_data)

    proc_cli.exec()

    print('Execution completed')
