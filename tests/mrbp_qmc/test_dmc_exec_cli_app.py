import attr
import yaml

from my_research_libs.mrbp_qmc import dmc_exec
from my_research_libs.mrbp_qmc.dmc_exec import CLIApp, load_cli_app_config


def test_app_spec():
    """Testing the DMCProcCLI."""
    # TODO: Fix config file.

    config = load_cli_app_config('./dmc-cli-app-spec.yml')
    data = config['app_spec'][0]

    app_spec = dmc_exec.cli_app.AppSpec.from_config(data)
    proc_cli_conf = attr.asdict(app_spec)

    # Execute
    app_result = app_spec.exec(dump_output=False)
    print(app_result)

    print(yaml.dump(proc_cli_conf, indent=4, allow_unicode=True))


def test_cli_app():
    """"""
    config = load_cli_app_config('./dmc-cli-app-spec.yml')

    app_cli = CLIApp.from_config(config)
    app_cli.exec()


if __name__ == '__main__':
    test_app_spec()
