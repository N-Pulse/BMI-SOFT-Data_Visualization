"""Pretend to be the stimulus PC: replays the LSL codes of eeg_emg_v3.py
following the acquisition protocol (DSD section 3.3).

Standard session for ONE arm: 6 movements = 6 blocks in random order, 3
consecutive repetitions per block, one baseline orientation per run.

    python scripts/fake_stimulus.py                       # full session, right arm
    python scripts/fake_stimulus.py --arm left --seed 42  # left arm, reproducible order
    python scripts/fake_stimulus.py --blocks 2 --rest 0 --delay 10   # quick test
"""
import argparse
import random
import time

from pylsl import StreamInfo, StreamOutlet

FAST = {10, 26}  # 1 s videos; every other movement lasts 3 s

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=None)
p.add_argument("--arm", choices=["left", "right"], default="right",
               help="arm used for the whole recording (the one wearing the EMG)")
p.add_argument("--blocks", type=int, default=6,
               help="number of blocks (max = number of movements)")
p.add_argument("--reps", type=int, default=3, help="consecutive repetitions per block")
p.add_argument("--movements", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
               help="movement codes (default: the 6 pre-ticked ones)")
p.add_argument("--orientation", type=int, choices=[1, 2, 3], default=2,
               help="1 up, 2 side (default), 3 down")
p.add_argument("--delay", type=float, default=15,
               help="countdown before the rest phase: time to click Listen")
p.add_argument("--rest", type=float, default=60,
               help="seconds per resting-state condition (0 = skip)")
p.add_argument("--iti", type=float, default=2, help="seconds between repetitions")
p.add_argument("--block-gap", type=float, default=5, help="seconds between blocks")
args = p.parse_args()

rng = random.Random(args.seed)
ARM = 1 if args.arm == "left" else 2                               # 1 left, 2 right
combos = [(ARM, mov) for mov in args.movements]
rng.shuffle(combos)                                                # random block order
blocks = combos[:args.blocks]

outlet = StreamOutlet(StreamInfo("stimulus_stream", "Markers", 1, 0, "int32", "fake-stim"))


def code(phase, arm, movement):
    return 10000 * phase + 1000 * arm + 100 * args.orientation + movement


def send(value, twice=False):
    outlet.push_sample([value])
    if twice:
        outlet.push_sample([value])


print(f"arm={args.arm}, orientation={args.orientation}, "
      f"{len(blocks)} blocks x {args.reps} reps")
print("Expected markers:")
n = 0
for b, (arm, mov) in enumerate(blocks, 1):
    for r in range(1, args.reps + 1):
        n += 1
        print(f"  {n:2d}  block {b:2d}  rep {r}  movement={mov:2d}  "
              f"arm={args.arm}  duration={'1' if mov in FAST else '3'} s")

print(f"Start: click 'Listen to stimulus' within {args.delay:.0f} s.")
for _ in range(5):                      # LSL test pings
    send(9999); time.sleep(0.5)
send(8888); time.sleep(1.5)             # ready
time.sleep(args.delay)                  # countdown before the first cue

if args.rest > 0:
    send(9701); time.sleep(args.rest)   # resting state, eyes open
    send(9702); time.sleep(args.rest)   # resting state, eyes closed

for b, (arm, mov) in enumerate(blocks, 1):
    video = 1.0 if mov in FAST else 3.0
    print(f"block {b}/{len(blocks)}: movement {mov}, arm {args.arm}")
    time.sleep(4 + 2 + 1)                           # announcement, pre-cue text, fixation
    send(code(1, arm, mov), twice=True); time.sleep(video)   # cue: observation only
    for _ in range(args.reps):                      # consecutive repetitions
        time.sleep(1.5)                             # instruction (no code)
        send(code(3, arm, mov), twice=True); time.sleep(3)             # prep
        send(code(5, arm, mov), twice=True); time.sleep(video)         # move
        send(code(7, arm, mov)); send(code(5, arm, mov)); time.sleep(2)  # return + extra move
        send(code(9, arm, mov)); send(code(5, arm, mov)); time.sleep(args.iti)  # ITI + extra move
    if b < len(blocks):
        time.sleep(args.block_gap)                  # fixation between blocks

send(8899)                                          # end
print("Done.")