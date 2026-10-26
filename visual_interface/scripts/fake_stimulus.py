"""Pretend to be the stimulus PC: replays the LSL codes of eeg_emg_v3.py
following the acquisition protocol (DSD sections 2.3 and 4.1).

Standard run for ONE arm: 6 movements = 6 blocks in random order, 3
consecutive repetitions per block, one baseline per run (1-3 open hand,
4-6 closed hand). Every video lasts 3 s.

    python scripts/fake_stimulus.py                          # full run, right arm
    python scripts/fake_stimulus.py --baseline 5 --seed 42   # closed hand, side
    python scripts/fake_stimulus.py --single                 # one code per phase
    python scripts/fake_stimulus.py --blocks 2 --rest 0 --delay 10   # quick test
"""
import argparse
import random
import time

from pylsl import StreamInfo, StreamOutlet

VIDEO = 3.0  # every video of the updated list lasts 3 s

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=None)
p.add_argument("--arm", choices=["left", "right"], default="right",
               help="arm used for the whole recording (the one wearing the EMG)")
p.add_argument("--baseline", type=int, choices=range(1, 7), default=2,
               help="1-3 palm up/side/down (open hand), 4-6 fist up/side/down (closed hand)")
p.add_argument("--blocks", type=int, default=6,
               help="number of blocks (max = number of movements)")
p.add_argument("--reps", type=int, default=3, help="consecutive repetitions per block")
p.add_argument("--movements", type=int, nargs="+", default=None,
               help="movement codes (default: 6 that fit the baseline)")
p.add_argument("--single", action="store_true",
               help="send exactly one code per phase (no duplicates, no extra move codes)")
p.add_argument("--delay", type=float, default=15,
               help="countdown before the rest phase: time to click Listen")
p.add_argument("--rest", type=float, default=60,
               help="seconds per resting-state condition (0 = skip)")
p.add_argument("--iti", type=float, default=2, help="seconds between repetitions")
p.add_argument("--block-gap", type=float, default=5, help="seconds between blocks")
args = p.parse_args()

if args.movements is None:
    # open-hand baselines start with closing movements, closed-hand ones with opening movements
    args.movements = [1, 2, 3, 4, 5, 6] if args.baseline <= 3 else [18, 19, 20, 24, 35, 37]

rng = random.Random(args.seed)
ARM = 1 if args.arm == "left" else 2                               # 1 left, 2 right
combos = [(ARM, mov) for mov in args.movements]
rng.shuffle(combos)                                                # random block order
blocks = combos[:args.blocks]

outlet = StreamOutlet(StreamInfo("stimulus_stream", "Markers", 1, 0, "int32", "fake-stim"))


def code(phase, arm, movement):
    return 10000 * phase + 1000 * arm + 100 * args.baseline + movement


def send(value, twice=False):
    outlet.push_sample([value])
    if twice and not args.single:
        outlet.push_sample([value])


def extra_move(arm, movement):
    """The previous script version also sent a move code after return and ITI."""
    if not args.single:
        send(code(5, arm, movement))


print(f"arm={args.arm}, baseline={args.baseline}, "
      f"{len(blocks)} blocks x {args.reps} reps, single={args.single}")
print("Expected markers:")
n = 0
for b, (arm, mov) in enumerate(blocks, 1):
    for r in range(1, args.reps + 1):
        n += 1
        print(f"  {n:2d}  block {b:2d}  rep {r}  movement={mov:2d}  "
              f"arm={args.arm}  duration={VIDEO:.0f} s")

print(f"Start: click 'Listen to stimulus' within {args.delay:.0f} s.")
for _ in range(5):                      # LSL test pings
    send(9999); time.sleep(0.5)
send(8888); time.sleep(1.5)             # ready
time.sleep(args.delay)                  # countdown before the first cue

if args.rest > 0:
    send(9701); time.sleep(args.rest)   # resting state, eyes open
    send(9702); time.sleep(args.rest)   # resting state, eyes closed

for b, (arm, mov) in enumerate(blocks, 1):
    print(f"block {b}/{len(blocks)}: movement {mov}, arm {args.arm}")
    time.sleep(4 + 2 + 1)                           # announcement, pre-cue text, fixation
    send(code(1, arm, mov), twice=True); time.sleep(VIDEO)    # cue: observation only
    for _ in range(args.reps):                      # consecutive repetitions
        time.sleep(1.5)                             # instruction (no code)
        send(code(3, arm, mov), twice=True); time.sleep(3)              # prep
        send(code(5, arm, mov), twice=True); time.sleep(VIDEO)          # move
        send(code(7, arm, mov)); extra_move(arm, mov); time.sleep(2)    # return
        send(code(9, arm, mov)); extra_move(arm, mov); time.sleep(args.iti)  # ITI
    if b < len(blocks):
        time.sleep(args.block_gap)                  # fixation between blocks

send(8899)                                          # end
print("Done.")