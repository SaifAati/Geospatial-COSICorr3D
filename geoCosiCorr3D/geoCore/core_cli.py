import json

import click
from geoCosiCorr3D.geoCore.base.base_cli import BaseOptArg, Opt
from geoCosiCorr3D.geoImageCorrelation.geoCorr_utils import pow2


class OptArg(BaseOptArg):
    pass

class CorrOpt(Opt):
    """Options for the correlate command are not allowed if used as 'correlate CONFIGURATION_FILE', so this marks a flag if they are used."""
    options_used = False

    def handle_parse_result(self, ctx, opts, args):
        CorrOpt.options_used = CorrOpt.options_used or self.name in opts
        return super(CorrOpt, self).handle_parse_result(ctx, opts, args)

class Exclusive(Opt):
    """Represents an option which is incompatible with the selection of a different option/argument's value.
    The target argument should be marked as eager to ensure it is processed before the exclusive command."""

    def __init__(self, *args, **kwargs):
        self.incompatible_with = kwargs.pop('incompatible_with')
        assert self.incompatible_with, "'incompatible_with' parameter required"
        super(Exclusive, self).__init__(*args, **kwargs)
        # TODO: add disclaimer to help message?

    def handle_parse_result(self, ctx, opts, args):
        CorrOpt.options_used = CorrOpt.options_used or self.name in opts
        otherName = self.incompatible_with[0]
        otherValue = self.incompatible_with[1]

        isPresent = self.name in opts
        missing = otherValue == None and otherName not in ctx.params
        isIncompatible = missing or ctx.params[otherName] == otherValue

        if isIncompatible:
            if isPresent:
                raise click.UsageError(f"Illegal usage: {self.name} is not allowed with {otherValue} {otherName}")
            else:
                self.prompt = None

        return super(Exclusive, self).handle_parse_result(ctx, opts, args)
