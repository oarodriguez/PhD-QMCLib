import attr
import yaml

from my_research_libs.mrbp_qmc import dmc_exec


def test_app_spec():
    """Testing the DMCProcCLI."""
    # TODO: Fix config file.

    with open('./dmc-cli-app-spec.yml', 'r') as fp:
        data = yaml.safe_load(fp)

    proc_cli = dmc_exec.cli_app.AppSpec.from_config(data['app_spec'])
    proc_cli_conf = attr.asdict(proc_cli)

    print(yaml.dump(proc_cli_conf, indent=4, allow_unicode=True))


if __name__ == '__main__':
    test_app_spec()
