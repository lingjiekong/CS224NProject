import os
from datetime import datetime
import sys
import math

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *

ex = Experiment('train_transcriber')


@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    # logdir = 'runs/transcriber-190304-193700'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000

    batch_size = 8
    sequence_length = 327680
    model_complexity = 16

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Halving batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    clip_gradient_norm = 3

    validation_length = 320000
    validation_interval = 500

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, clip_gradient_norm,
          validation_length, validation_interval):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    dataset = MAESTRO(sequence_length=sequence_length)
    # dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm'], 
                  # sequence_length=sequence_length)
    # dataset = MAPS(groups=['AkPnBcht'], 
    #               sequence_length=sequence_length)

    loader = DataLoader(dataset, batch_size, shuffle=True)

    # iterations = math.ceil(len(dataset) * 1000 / batch_size / checkpoint_interval) * checkpoint_interval # make sure smaller datasets have less iterations
    # print(iterations)


    validation_dataset = MAESTRO(groups=['validation'], sequence_length=validation_length)
    # validation_dataset = MAPS(groups=['SptkBGCl', 'StbgTGd2'],
                              # sequence_length=validation_length)
    # validation_dataset = MAPS(groups=['SptkBGCl'],
    #                           sequence_length=validation_length)


    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=[0, 2])
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    hist_valid_scores = []
    patience, num_trial = 0, 0
    max_patience, max_trial = 5, 5 

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        scheduler.step()

        mel = melspectrogram(batch['audio'].reshape(-1, batch['audio'].shape[-1])[:, :-1]).transpose(-1, -2)
        predictions, losses = model.module.run_on_batch(batch, mel)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if clip_gradient_norm:
            for parameter in model.parameters():
                clip_grad_norm_([parameter], clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            
            writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            print ("running validation\n")
            model.eval()
            with torch.no_grad():
                count, values = 0.0, 0.0
                for key, value in evaluate(validation_dataset, model).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
                    count += 1.0
                    values += np.mean(value)
                dev_value = values/count
                print('training loss: %.8f;  validation loss: %.8f' % (loss[0], dev_value))
            is_better = len(hist_valid_scores) == 0 or dev_value > max(hist_valid_scores)
            hist_valid_scores.append(dev_value)
            model.train()

            if is_better:
                patience = 0
                torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            else:
                patience += 1
                print('hit patience %d' % patience, file=sys.stderr)
                if patience == max_patience:
                    num_trial += 1
                    print('hit #%d trial' % num_trial, file=sys.stderr)
                    if num_trial == max_trial:
                        print('early stop!', file=sys.stderr)
                        exit(0)
                    # reset patience
                    patience = 0
