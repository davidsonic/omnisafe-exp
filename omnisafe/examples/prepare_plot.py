from pathlib import Path
import os

folders = os.listdir('runs/')
methods = [x.split('-', 1)[0] for x in folders]
env_names = [x.split('-', 1)[1] for x in folders]
env_names = list(set(env_names))

save_path = Path('exps')
for env_name in env_names:
    for method in methods:
        if method == 'NaturalPG' and method == 'PolicyGradient':
            continue
        src = f'runs/{method}-{env_name}'
        env = env_name[1:-1]
        dst_root = f'exps/{env}'
        if not os.path.exists(dst_root): 
            os.makedirs(dst_root, exist_ok=True)
        dst = f'exps/{env}/{method}'
        # print(f'{src}, {dst}')
        if os.path.exists(dst): 
            continue
        else:
            os.system(f'cp -R {src} {dst}')

                                

import argparse
from omnisafe.utils.plotter2 import Plotter

def plot(logdir:str): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='Steps')
    parser.add_argument('--value', '-y', default='Rewards', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--estimator', default='mean')
    args = parser.parse_args()

    args.logdir = logdir
    plotter = Plotter()
    plotter.make_plots(
        args.logdir,
        args.legend,
        args.xaxis,
        args.value,
        args.count,
        smooth=args.smooth,
        select=args.select,
        exclude=args.exclude,
        estimator=args.estimator,
    )



if __name__ == '__main__':
    for env_name in env_names:
        env = env_name[1:-1]
        plot([f'exps/{env}'])