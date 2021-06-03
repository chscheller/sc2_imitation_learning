from absl import flags
from absl.flags import ListParser, CsvListSerializer


class IntListParser(ListParser):
    def parse(self, argument):
        parsed_list = super().parse(argument)
        return [int(x) for x in parsed_list]


# noinspection PyPep8Naming
def DEFINE_int_list(name, default, help, flag_values=flags.FLAGS, **args):
    parser = IntListParser()
    serializer = CsvListSerializer(',')
    flags.DEFINE(parser, name, default, help, flag_values, serializer, **args)
