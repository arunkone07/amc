import os
import numpy as np
import argparse
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True

from env.quantization_env import QuantizationEnv
from lib.agent import DDPG
from lib.utils import get_output_folder

from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='Quantization search script')

    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    parser.add_argument('--model', default='mobilenet', type=str, help='model to quantize')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    parser.add_argument('--reward', default='acc_flops_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--n_calibration_batches', default=60, type=int, help='n_calibration_batches')
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for critic')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--init_delta', default=0.5, type=float, help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float, help='delta decay during exploration')
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=300, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')
    parser.add_argument('--bits', default=None, type=str, help='bits for quantizing')
    parser.add_argument('--channels', default=None, type=str, help='channels after quantizing')
    parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    return parser.parse_args()

def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
    if model == 'mobilenet' and dataset == 'imagenet':
        from models.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
    elif model == 'mobilenetv2' and dataset == 'imagenet':
        from models.mobilenet_v2 import MobileNetV2
        net = MobileNetV2(n_class=1000)
    else:
        raise NotImplementedError
    sd = torch.load(checkpoint_path)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net = net.cuda()
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, range(n_gpu))

    return net, deepcopy(net.state_dict())

def train(num_episode, agent, env, output):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermediate model
        if episode % int(num_episode / 3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            print('#{}: episode_reward:{:.4f} acc: {:.4f}'.format(episode, episode_reward, info['accuracy']))
            text_writer.write('#{}: episode_reward:{:.4f} acc: {:.4f}\n'.format(episode, episode_reward, info['accuracy']))
            final_reward = T[-1][0]
            # agent observe and update policy
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    agent.update_policy()

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', env.best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)
            # record the quantization bitwidth for each layer
            for i, bitwidth in enumerate(env.strategy):
                tfwriter.add_scalar('bitwidth/{}'.format(i), bitwidth, episode)

            text_writer.write('best reward: {}\n'.format(env.best_reward))
            text_writer.write('best policy: {}\n'.format(env.best_strategy))

    # Save the model with the best quantization strategy
    print("Saving model with best quantization strategy...")
    env.reset()  # Reset the environment to load the original model
    env._quantize_model(env.best_strategy)

    model_path = os.path.join(output, 'model_best_quantized.pth')
    torch.save(env.model.state_dict(), model_path)
    print("Best quantized model saved to {}".format(model_path))
    text_writer.close()


def export_model(env, args):
    assert args.bits is not None, 'Please provide a valid bit-width list'
    assert args.export_path is not None, 'Please provide a valid export path'
    env.set_export_path(args.export_path) #This is not used in the quant env but is here for consistency.

    print('=> Quantizing with bit-widths: {}'.format(args.bits))

    bits = args.bits.split(',')
    bits = [int(b) for b in bits]

    assert len(bits) == len(env.quantizable_idx), "Number of bit-widths must match the number of quantizable layers"

    for b in bits:
        env.step(b) # call step to set the strategy

    env._quantize_model(env.strategy) #apply the strategy.

    torch.save(env.model.state_dict(), args.export_path) #export the model.

    print("=> Quantized model exported to: {}".format(args.export_path))

    return

if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    model, checkpoint = get_model_and_checkpoint(args.model, args.dataset, checkpoint_path=args.ckpt_path, n_gpu=args.n_gpu)

    env = QuantizationEnv(model, checkpoint, args.dataset, args=args, n_data_worker=args.n_worker, 
                          batch_size=args.data_bsize)

    if args.job == 'train':
        # build folder and logs
        base_folder_name = '{}_{}_q_search'.format(args.model, args.dataset)
        if args.suffix is not None:
            base_folder_name = base_folder_name + '_' + args.suffix
        args.output = get_output_folder(args.output, base_folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tfwriter = SummaryWriter(logdir=args.output)
        text_writer = open(os.path.join(args.output, 'log.txt'), 'w')
        print('=> Output path: {}...'.format(args.output))

        nb_states = env.layer_embedding.shape[1]
        nb_actions = 1  # just 1 action here (quantization bitwidth)

        args.rmsize = args.rmsize * len(env.quantizable_idx)  # for each layer
        print('** Actual replay buffer size: {}'.format(args.rmsize))

        agent = DDPG(nb_states, nb_actions, args)
        train(args.train_episode, agent, env, args.output)
    elif args.job == 'export':
        export_model(env, args) 
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
