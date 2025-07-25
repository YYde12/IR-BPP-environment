import argparse
import torch
import numpy as np
from tools import load_shape_dict, shotInfoPre, shapeProcessing

def get_args():
    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description='Rainbow for IR BPP')

    # Parameters for the reinforcement learning agent
    parser.add_argument('--hidden-size', type=int, default=128, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=31, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-1, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=8, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--target-update', type=int, default=int(1e3), metavar='τ',
                        help='Number of steps after which to update target network')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Batch size')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY',
                        help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                        help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=1.0, metavar='β',
                        help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                        help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=1, metavar='T',
                        help='Number of consecutive states processed')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'dataset-efficient'],
                        metavar='ARCH', help='Network architecture')
    parser.add_argument('--load-model', action='store_true', help='Load the trained model')
    parser.add_argument('--learn-start', type=int, default=int(5e2), metavar='STEPS',
                        help='Number of steps before starting training')

    parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS',
                        help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes-training', type=int, default=100, metavar='N',
                        help='Number of evaluation episodes to average over')
    # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
                        help='Number of transitions to use for validating Q')

    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')

    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--save-interval', type=int, default=1000, help='How often to save the model.')
    parser.add_argument('--model-save-path',type=str, default='./logs/experiment', help='The path to save the trained model')

    parser.add_argument('--disable-bzip-memory', action='store_true',
                        help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
    parser.add_argument('--print-log-interval',     type=int,   default=10, help='How often to print training logs')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')

    # Parameters for irregular shape packing
    parser.add_argument('--envName', type=str, default='Physics-v0', help='The environment name for packing policy training and testing')
    parser.add_argument('--dataset', type=str, default='blockout', help='The organized dataset folder for training') # blockout general kitchen abc
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--custom', type=str, default=None, help='Customized label for the experiment')
    parser.add_argument('--hierachical', action='store_true', help='Use hierachical policy')
    parser.add_argument('--bufferSize', type=int, default=1, help='Object buffer size') # 1 3 5 10
    parser.add_argument('--num_processes', type=int, default=2, help='How many parallel processes used for training') # 16 1
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--samplePointsNum', type=int, default=1024, help='How many points to sample from the object surface')
    parser.add_argument('--selectedAction', type=int, default=500, help='How many actions to select from the action space')
    parser.add_argument('--maxBatch', type=int, default=2, help='How many batches for simulation')
    parser.add_argument('--visual', action='store_true', help='Render the scene')
    parser.add_argument('--resolutionA', type=float, default = 0.02, help='The resolution for the action space')
    parser.add_argument('--resolutionH', type=float, default = 0.01, help='The resolution for the heightmap')
    parser.add_argument('--resolutionZ', type=float, default = 0.01, help='The resolution for the z axis')
    parser.add_argument('--resolutionRot', type=int, default = 8, help='The resolution for the rotation, 2 for cube, 4 for blockout, and 8 for the rest')

    parser.add_argument('--locmodel', type=str, default=None, help='The path to load the trained location model to select location candidate.')
    parser.add_argument('--ordmodel', type=str, default=None, help='(Optional) The path to load the trained order model to select object from the buffer')

    parser.add_argument('--only-simulate-current', action='store_true', help='Only simulate the current item')
    parser.add_argument('--non_blocking', action='store_true', help='Train actor and critic in non-blocking mode')
    parser.add_argument('--time_limit', type=float, default = 0.01, help='Time limit for each simulation step when non_blocking is True')

    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-episodes-test', type=int, default=2000, help='Number of evaluation episodes to average over')
    parser.add_argument('--use_heuristic', action='store_true', help='If set, use heuristic policy instead of RL')
    parser.add_argument('--heuristic_method', type=str, default='MINZ', choices=['MINZ', 'DBLF', 'FIRSTFIT', 'HM'], help='Heuristic method to use when --use_heuristic is set')


    args = parser.parse_args()
    print('first hierachical',args.hierachical)
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))


    args.objPath = './dataset/{}/shape_vhacd'.format(args.dataset)
    args.pointCloud = './dataset/{}/pointCloud'.format(args.dataset)
    args.dicPath = './dataset/{}/id2shape.pt'.format(args.dataset)

    if  args.dataset == 'kitchen':
        args.dataSample = 'category'
    else:
        args.dataSample = 'instance'

    args.categories = len(torch.load(args.dicPath))
    args.bin_dimension = np.round([0.32, 0.32, 0.30], decimals=6)

    args.ZRotNum = args.resolutionRot  # Max: 4/8

    args.objVecLen = 9
    args.model = None
    args.load_memory_path = None
    args.save_memory_path = None
    args.scale =  [100, 100, 100] # fix it! don't change it!
    args.meshScale = 1
    args.heightResolution = args.resolutionZ
    args.shapeDict, args.infoDict = load_shape_dict(args, True, scale=args.meshScale)
    args.physics = True
    args.heightMap = True
    args.useHeightMap = True
    args.globalView = True if args.evaluate else False
    args.shotInfo = shotInfoPre(args, args.meshScale)
    args.simulation = True
    args.distributed = True
    args.test_name = './dataset/{}/test_sequence.pt'.format(args.dataset)
    args.shapeArray = shapeProcessing(args.shapeDict, args)

    if args.evaluate:
        args.num_processes = 1

    # temp setting
    # args.evaluate = True

    return args
