from pathlib import Path

import attr
import yaml

from my_research_libs.mrbp_qmc import dmc_exec
from my_research_libs.mrbp_qmc.dmc_exec import load_cli_app_config


def test_app_spec():
    """Testing the DMCProcCLI."""
    # TODO: Fix config file.

    conf_location = Path('./dmc-cli-app-spec.yml')
    config = load_cli_app_config(conf_location)
    data = config['app_spec'][0]

    app_spec = dmc_exec.cli_app.AppSpec.from_config(data)
    proc_cli_conf = attr.asdict(app_spec)

    # Execute
    app_input = app_spec.build_input()
    app_result = app_spec.exec(app_input)
    app_spec.dump_output(proc_result=app_result)

    print(yaml.dump(proc_cli_conf, indent=4, allow_unicode=True))


if __name__ == '__main__':
    test_app_spec()
