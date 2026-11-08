from tensorboard.backend.event_processing import event_accumulator
import sys

import glob
import os

if len(sys.argv) > 1:
    log_file = sys.argv[1]
else:
    candidates = glob.glob("spdrbot3_direct_project/logs/rsl_rl/zoe_hexapod_rough/*/events.out.tfevents.*")
    log_file = candidates[0] if candidates else "logs/rsl_rl/zoe_hexapod_rough/events.out.tfevents"

ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

tags = ea.Tags()['scalars']
print(f"Found tags: {tags}")

if 'Train/mean_episode_length' in tags:
    events = ea.Scalars('Train/mean_episode_length')
    if len(events) > 0:
        print(f"Mean episode length at end of training: {events[-1].value}")

if 'Train/mean_reward' in tags:
    events = ea.Scalars('Train/mean_reward')
    if len(events) > 0:
        print(f"Mean reward at end of training: {events[-1].value}")
