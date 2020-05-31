import attr
import yaml

from phd_qmclib.mrbp_qmc import dmc_exec
from phd_qmclib.mrbp_qmc.dmc_exec import CLIApp, config


def test_app_spec():
    """Testing the DMCProcCLI."""
    # TODO: Fix config file.

    config_data = config.loader.load('./dmc-cli-app-spec.yml')
    data = config_data['app_spec'][0]

    app_spec = dmc_exec.cli_app.AppSpec.from_config(data)
    proc_cli_conf = attr.asdict(app_spec)

    # Execute
    app_result = app_spec.exec(dump_output=False)
    print(app_result)

    print(yaml.dump(proc_cli_conf, indent=4, allow_unicode=True))


def test_cli_app():
    """"""
    config_data = config.loader.load('./dmc-cli-app-spec.yml')

    app_cli = CLIApp.from_config(config_data)
    app_cli.exec()


if __name__ == '__main__':
    test_app_spec()
