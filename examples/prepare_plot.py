from pathlib import Path
import os

folders = os.listdir('runs/')
methods = [x.split('-', 1)[0] for x in folders]
env_names = [x.split('-', 1)[1] for x in folders]
env_names = list(set(env_names))

save_path = Path('exps')
for env_name in env_names:
    for method in methods:
        if method == 'NaturalPG':
            continue
        src = f'runs/{method}-{env_name}'
        env = env_name[1:-1]
        dst_root = f'exps/{env}'
        if not os.path.exists(dst_root): 
            os.makedirs(dst_root, exist_ok=True)
        dst = f'exps/{env}/{method}'

        print(f'{src}, {dst}')
        if os.path.exists(dst): 
            continue
        else:
            os.system(f'cp -R {src} {dst}')


# run plot
print(env_names)
for env_name in env_names:
    env = env_name[1:-1]
    os.system(f'python3 plot.py --logdir=exps/{env}')
                                
