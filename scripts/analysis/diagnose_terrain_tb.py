# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Extract and summarize all scalar metrics from TensorBoard events for Stage 2 diagnosis."""

import os
import sys
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def analyze_event_file(event_file_path):
    print(f"Reading: {event_file_path}")
    ea = event_accumulator.EventAccumulator(event_file_path)
    ea.Reload()

    tags = sorted(ea.Tags().get("scalars", []))
    print(f"Found {len(tags)} scalar tags.\n")

    results = []
    for t in tags:
        events = ea.Scalars(t)
        steps = [e.step for e in events]
        vals = np.array([e.value for e in events])
        first_step, last_step = steps[0], steps[-1]
        first_val = vals[0]
        mid_val = vals[len(vals) // 2]
        last_val = vals[-1]
        min_val = np.min(vals)
        max_val = np.max(vals)
        mean_val = np.mean(vals)
        
        # Simple trend
        first_q = np.mean(vals[:max(1, len(vals) // 4)])
        last_q = np.mean(vals[-max(1, len(vals) // 4):])
        if last_q > first_q * 1.05:
            trend = "INCREASING"
        elif last_q < first_q * 0.95:
            trend = "DECREASING"
        else:
            trend = "STABLE"

        results.append({
            "tag": t,
            "count": len(vals),
            "first_step": first_step,
            "last_step": last_step,
            "first": first_val,
            "mid": mid_val,
            "last": last_val,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "trend": trend,
        })

    # Print markdown table
    print(f"{'Scalar Tag':<45} | {'First':<10} | {'Mid':<10} | {'Last':<10} | {'Min':<10} | {'Max':<10} | {'Trend'}")
    print("-" * 115)
    for r in results:
        print(f"{r['tag']:<45} | {r['first']:<10.4f} | {r['mid']:<10.4f} | {r['last']:<10.4f} | {r['min']:<10.4f} | {r['max']:<10.4f} | {r['trend']}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/rithwik/SpdrBot-main/logs/rsl_rl/zoe_hexapod_terrain/2026-11-06_22-18-05/events.out.tfevents.1793983689.rithwik-Legion-Pro-5.160402.0"
    analyze_event_file(path)

