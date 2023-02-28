import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vae', default=False)
args = vars(parser.parse_args())

print(args)
#print(args['vae'])