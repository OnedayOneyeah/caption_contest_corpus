from utils import dict_to_markdown, mkdirp
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=int, help='0,1,2,3,4 are cross-val splits 5 is leaderboard')
    parser.add_argument('task', type=str, choices=['matching', 'ranking'])

    parser.add_argument('--clip_model',
                        default='ViT-L/14@336px',
                        choices=['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'RN50x16', 'RN50x64', 'ViT-L/14@336px', 'ViT-L/14'])

    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    parser.add_argument('--n_epochs',
                        type=int,
                        default=10)

    parser.add_argument('--lr',
                        type=float,
                        default=.00001)

    parser.add_argument('--result_dir',
                        type=str,
                        default="results/")

    parser.add_argument('--dataset_path',
                        type=str,
                        default="dataset/")

    parser.add_argument('--use_accelerate',
                        type=int,
                        default=0,
                        help='if this flag is set, we will use huggingface accelerate intsead of dataparallel')

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='how many steps for gradient accumulation')

    parser.add_argument('--pad',
                        type=int,
                        default=1,
                        help='if 0 we will do standard center crop, if 1 we will do pad.')

    parser.add_argument('--warmup',
                        type=int,
                        default=200,
                        help='how many steps of warmup should we use?')

    parser.add_argument('--force_run',
                        type=int,
                        default=0,
                        help='if 1, we will force the run even if the output already exists.')

    parser.add_argument('--prefix',
                        type=str,
                        default=None,
                        help='if this prefix is set, it will be appended to the input.')

    parser.add_argument('--mode',
                        type=str,
                        default='basic',
                        help='if this option is set (except for basic), adding extra answer candidate or text augmentation will be performed.',
                        choices=['basic', 'rephrase', 'keywords', 'Antonym'])

    parser.add_argument("--debug",
                        action="store_true",
                        help="debug (fast) mode, break all loops, do not load all data into memory.")

    args = parser.parse_args()

    if args.prefix and ('+' in args.prefix or '~' in args.prefix):
        print('We dont support plus signs or tildes in prefixes.')
        quit()

    args.result_dir = f'results/{time.strftime("%Y_%m_%d_%H_%M_%S")}/'
    mkdirp(args.result_dir)
    # args.output_path = args.result_dir + 'model_best.pt'

    args.output_path = (args.result_dir +
                        'task={}'.format(args.task) +
                        '~split={}'.format(args.split) +
                        '~val{}'.format('acc') + '={:.5f}' + '~pad={}'.format(args.pad) +
                        '~model=' + '{}'.format(args.clip_model.replace('/', '*')) +
                        ('' if not args.prefix else 'prefix=' + '+'.join(args.prefix.strip().split())) +
                        '~lr={}.pt'.format(args.lr))

    if not args.prefix:
        args.prefix = ''
    else:
        args.prefix = args.prefix.strip() + ' '

    if not args.force_run:
        toks = args.output_path.split('/')
        outdir = '/'.join(toks[:-1]) if len(toks) > 1 else '.'
        def fnameparse(x):
            return (x.split('~val')[0], '~'.join(x.split('~val')[1].split('~')[1:]))
        if fnameparse(args.output_path) in set([fnameparse(x) for x in os.listdir(outdir) if '.pt' in x]):
            print('{} done already, run with --force_run to run.'.format(args.output_path))
            quit()

    print(dict_to_markdown(vars(args), max_str_len=120))

    return args