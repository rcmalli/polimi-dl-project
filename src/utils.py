import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-i', '--input',
        metavar='C',
        default='None',
        help='The Input file')
    args = argparser.parse_args()
    return args
