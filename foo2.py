import argparse
import sys

# parser = argparse.ArgumentParser(description='Process some integers')
# parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
# args = parser.parse_args()
# print(args.accumulate(args.integers))


parser = argparse.ArgumentParser(description='Run an FFT on a sample.')
parser.add_argument('in', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='sample to process')
parser.add_argument('-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='file to output results to')
args = parser.parse_args()
print(args)