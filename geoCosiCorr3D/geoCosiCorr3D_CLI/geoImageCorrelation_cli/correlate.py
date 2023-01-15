import click
from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate
from geoCosiCorr3D.geoCore.constants import CORRELATION
from geoCosiCorr3D.geoCore.geoCosiCorrBaseCfg.BaseReadConfig import load_config
from geoCosiCorr3D.geoCosiCorr3D_CLI.geoImageCorrelation_cli.cli_utils import validatePath, validatePositives, validateWindowSizes, \
    rangeValidator, call_with_conf
from geoCosiCorr3D.geoCore.core_cli import Exclusive, Opt, CorrOpt, OptArg


# the cli entrypoint
@click.group(context_settings=dict(help_option_names=["-help", "-h"]))
def cli():
    pass


# pairs of argument/option name to name/path in configuration file
# A '$' before name indicates it is an argument and not an option
corrArgToConfig = {
    "$base_image": "base_image_path",
    "$target_image": "target_image_path",
    "base_band": "base_band",
    "target_band": "target_band",
    "output": "output_path",
    "correlator": "correlator_name",
    "window_sizes": "correlator_params.window_size",
    "step": "correlator_params.step",
    "grid": "correlator_params.grid",
    "mask_thresh": "correlator_params.mask_th",
    "iters": "correlator_params.nb_iters",
    "search_range": "correlator_params.search_range"
}

# run this with the specified json between the definition of each command
# the json should be a maximal config file with each value being [default value, "help message"]
Opt.parameter_data = load_config(argToConfig=corrArgToConfig, config_path=CORRELATION.CORR_PARAMS_CONFIG)


@cli.command()
@click.argument('base_image', type=click.File('r'), callback=validatePath)
@click.argument('target_image', type=click.File('r'), callback=validatePath, cls=OptArg)
@click.option('-base_band', '-bb', cls=CorrOpt)
@click.option('-target_band', '-tb', cls=CorrOpt)
@click.option('-output', '-o', cls=CorrOpt, type=str)
@click.option('-correlator', '-c', cls=CorrOpt, type=click.Choice(['frequency', 'spatial']), is_eager=True, help="")
@click.option('-window_sizes', '-ws', cls=CorrOpt, nargs=4, callback=validateWindowSizes)
@click.option('-step', '-s', cls=CorrOpt, nargs=2, callback=validatePositives)
@click.option('-grid', '-g', cls=CorrOpt, type=bool)
@click.option('-mask_thresh', '-mt', cls=Exclusive, incompatible_with=('correlator', 'spatial'),
              callback=rangeValidator(min=0, max=1))
@click.option('-iters', '-i', cls=Exclusive, incompatible_with=('correlator', 'spatial'),
              callback=rangeValidator(min=0))
@click.option('-search_range', '-sr', cls=Exclusive, incompatible_with=('correlator', 'frequency'),
              callback=validatePositives, nargs=2)
def correlate(base_image, target_image, base_band, target_band, output, correlator, window_sizes, step, grid,
              mask_thresh, iters, search_range):
    """Generates a displacement map between two overlapping orthorectified images.

    Alt Usage: cc.py correlate CONFIGURATION_FILE"""

    # Check if only one argument, and if so, load json and re-invoke command with parameters as options
    if target_image is None:
        if CorrOpt.options_used:
            raise click.UsageError("Options cannot be used with 'correlate CONFIGURATION_FILE'.")
        call_with_conf(correlate, corrArgToConfig, base_image)
        return

    # Create corr_config file with all required information
    config = {}
    for arg in corrArgToConfig:
        isArg = arg[0] == '$'
        fullPath = corrArgToConfig[arg].split('.')
        current = config
        for dir in fullPath[:-1]:
            current.setdefault(dir, {})
            current = current[dir]
        current[fullPath[-1]] = eval(arg[1:] if isArg else arg)

    Correlate.from_config(config)


if __name__ == '__main__':
    cli()
