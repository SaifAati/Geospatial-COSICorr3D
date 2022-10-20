import click
import json
from geoCosiCorr3D.geoImageCorrelation.geoCorr_utils import pow2


# class Opt(click.Option):
#     """Wrapper classes for options that provide default value filling from configuration file."""
#     parameter_data = {}
#
#     def __init__(self, *params, **kargs):
#         name = params
#         while not isinstance(name, str): name = name[0]
#         name = name[1:]
#         if name in self.parameter_data:
#             value, help = self.parameter_data[name]
#             kargs.setdefault('default', value)
#             kargs.setdefault('help', help)
#             kargs.setdefault('show_default', True)
#
#         super(Opt, self).__init__(*params, **kargs)
#
# class CorrOpt(Opt):
#     """Options for the correlate command are not allowed if used as 'correlate CONFIGURATION_FILE', so this marks a flag if they are used."""
#     options_used = False
#     def handle_parse_result(self, ctx, opts, args):
#         CorrOpt.options_used = CorrOpt.options_used or self.name in opts
#         return super(CorrOpt, self).handle_parse_result(ctx, opts, args)
#
# class OptArg(click.Argument):
#     """Argument that appears required in help but is actually optional."""
#     def handle_parse_result(self, ctx, opts, args):
#         self.required = False
#         return super(OptArg, self).handle_parse_result(ctx, opts, args)
#
# class Exclusive(Opt):
#     """Represents an option which is incompatible with the selection of a different option/argument's value.
#     The target argument should be marked as eager to ensure it is processed before the exclusive command."""
#     def __init__(self, *args, **kwargs):
#         self.incompatible_with = kwargs.pop('incompatible_with')
#         assert self.incompatible_with, "'incompatible_with' parameter required"
#         super(Exclusive, self).__init__(*args, **kwargs)
#         #TODO: add disclaimer to help message?
#
#     def handle_parse_result(self, ctx, opts, args):
#         CorrOpt.options_used = CorrOpt.options_used or self.name in opts
#         otherName = self.incompatible_with[0]
#         otherValue = self.incompatible_with[1]
#
#         isPresent = self.name in opts
#         missing = otherValue == None and otherName not in ctx.params
#         isIncompatible = missing or ctx.params[otherName] == otherValue
#
#         if isIncompatible:
#             if isPresent:
#                 raise click.UsageError( f"Illegal usage: {self.name} is not allowed with {otherValue} {otherName}")
#             else:
#                 self.prompt = None
#
#         return super(Exclusive, self).handle_parse_result(ctx, opts, args)

# region validators

def validateWindowSizes(ctx, param, values):
    """Validates window sizes are powers of 2 and > 0"""
    if not all(map(lambda x: x > 0, values)):
        raise click.BadParameter("all window sizes must be positive")
    if not all(map(pow2, values)):
        raise click.BadParameter("all window sizes must be powers of 2")
    return values


def validatePositives(ctx, param, values):
    if not all(map(lambda x: x > 0, values)):
        raise click.BadParameter("all values must be positive")
    return values


def rangeValidator(min=None, max=None):
    def validator(ctx, param, value):
        if (min == None or value >= min) and (max == None or value <= max):
            return value
        mn = "" if min == None else f"{min}<="
        mx = "" if max == None else f"<={max}"
        raise click.BadParameter(f"value must be {mn}x{mx}")

    return validator


def validatePath(ctx, param, file):
    return file and file.name


# endregion validators

def call_with_conf(command, argToConfig, config_path):
    """Uses the argToConfig map to invoke a command using values loaded from the configuration file."""
    config = load_config(argToConfig, config_path)

    options = []
    for arg, value in config.items():
        isArg = arg[0] == '$'
        if not isArg: options.append("-" + arg)
        if isinstance(value, list):
            options.extend(map(str, value))
        else:
            options.append(str(value))
    command.main(args=options)


def load_config(argToConfig, config_path):
    """Loads a configuration file into a dictionary."""
    f = open(config_path, )
    config = json.load(f)
    f.close()

    options = {}
    for arg in argToConfig:
        isArg = arg[0] == '$'
        fullPath = argToConfig[arg].split('.')
        current = config
        for dir in fullPath:
            if dir not in current:
                if isArg: raise click.UsageError(f'required argument "{arg[1:]}" is missing from configuration file.')
                break
            current = current[dir]
        else:
            options[arg] = current
    return options
