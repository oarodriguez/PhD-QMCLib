import io
import logging
import os
import socket
from pathlib import Path

import click
from click import option
from dotenv import find_dotenv, load_dotenv
from my_research_libs.utils import now

from . import config
from .dmc_exec import (
    CLIApp, config as dmc_exec_config
)

load_dotenv(find_dotenv(), verbose=True)
UNIX_NEWLINE = '\n'

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

    tpl_path = Path(template).absolute()

    assert tpl_path.is_file()

    if output is None:
        output_path = Path('.').absolute()
    else:
        output_path = Path(output).absolute()

    if output_path.is_dir():
        output_path /= gen_filename()

    if output_path.exists() and not replace:
        raise IOError(f"file {output_path} exists")

    print(output_path)

    # Create the output directory.
    os.makedirs(output_path.parent, exist_ok=True)

    config_template = config.Template(tpl_path)
    config_template.save(output_path)


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
    print(BANNER)

    config_path = Path(config_path).absolute()

    #
    assert config_path.is_file()

    # Create the CLI object and validate.
    config_data = dmc_exec_config.loader.load(config_path)
    proc_cli = CLIApp.from_config(config_data)

    proc_cli.exec()

    print('Execution completed')
