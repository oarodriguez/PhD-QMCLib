import attr
import yaml

from my_research_libs.mrbp_qmc import dmc_exec


def test_proc_cli():
    """Testing the DMCProcCLI."""
    # TODO: Fix config file.

    with open('./dmc-conf.tpl.yml', 'r') as fp:
        data = yaml.safe_load(fp)

    proc_cli = dmc_exec.cli_app.ProcCLI.from_config(data)
    proc_cli_conf = attr.asdict(proc_cli)

    proc_cli_clone = dmc_exec.cli_app.ProcCLI.from_config(proc_cli_conf)
    assert proc_cli == proc_cli_clone

    print(yaml.dump(proc_cli_conf, indent=4))


if __name__ == '__main__':
    test_proc_cli()
