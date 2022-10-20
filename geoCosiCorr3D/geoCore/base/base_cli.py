import click


class Opt(click.Option):
    """Wrapper classes for options that provide default value filling from configuration file."""
    parameter_data = {}

    def __init__(self, *params, **kargs):
        name = params
        while not isinstance(name, str): name = name[0]
        name = name[1:]
        if name in self.parameter_data:
            value, help = self.parameter_data[name]
            kargs.setdefault('default', value)
            kargs.setdefault('help', help)
            kargs.setdefault('show_default', True)

        super(Opt, self).__init__(*params, **kargs)


class BaseOptArg(click.Argument):
    """Argument that appears required in help but is actually optional."""

    def handle_parse_result(self, ctx, opts, args):
        self.required = False
        return super(BaseOptArg, self).handle_parse_result(ctx, opts, args)
