from argparse import ArgumentParser


class CompareModelOptions:

    def __init__(self):
        self._init_parser()

    def _init_parser(self):
        usage = 'bin/project'
        self.parser = ArgumentParser(usage=usage)
        self.parser.add_argument('-l',
                                 '--log_dir',
                                 default='./compare_models_logs',
                                 dest='log_dir',
                                 help='This is where the log files end up')

    def parse(self, args=None):
        return self.parser.parse_args(args)