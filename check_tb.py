from tensorboard.backend.event_processing import event_accumulator
import sys

log_file = "/home/rithwik/SpdrBot-main/spdrbot3_direct_project/logs/rsl_rl/zoe_hexapod_rough/2026-05-30_08-39-25/events.out.tfevents.1780110573.rithwik-Legion-Pro-5.36838.0"

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
