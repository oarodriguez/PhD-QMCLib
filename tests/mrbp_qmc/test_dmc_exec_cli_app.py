import attr
import yaml

from my_research_libs.mrbp_qmc import dmc_exec


def test_proc_cli():
    """Testing the DMCProcCLI."""
    # TODO: Fix config file.

    with open('./dmc-conf.tpl.yml', 'r') as fp:
        data = yaml.safe_load(fp)

    proc_cli = dmc_exec.cli_app.CLIApp.from_config(data)
    proc_cli_conf = attr.asdict(proc_cli)

    print(yaml.dump(data, indent=4, allow_unicode=True))


if __name__ == '__main__':
    test_proc_cli()
