#!/usr/bin/env python3
"""Regression tests for the recording path.

The recording path is where a silent bug is most expensive: you only find out
after a session with a participant that the data is unusable, and you cannot
ask them to come back. These tests cover the parts that decide whether a
recording can be trained on.

Run them from the visual_interface folder:

    python tests/test_recording.py

No pytest needed, but `pytest tests/` works too.
"""

import os
import sys
import time
import tempfile

# Import app.py from the parent folder, and keep every file this test writes
# inside a temporary directory so a real bids_output/ is never touched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(tempfile.mkdtemp(prefix="npulse-test-"))

import app  # noqa: E402


def test_open_marker_has_no_fake_duration():
    """A marker that was never closed must be 'n/a', never 0.

    Writing 0 would tell the Decoding team the gesture was instantaneous,
    which is a wrong answer rather than a missing one.
    """
    tsv = app.create_events_tsv([{"offset": 1.0, "label": "grip"}])
    row = tsv.strip().split("\n")[1].split("\t")
    assert row[1] == "n/a", f"expected 'n/a' duration, got {row[1]!r}"


def test_closed_marker_records_real_duration():
    tsv = app.create_events_tsv([
        {"offset": 2.5, "label": "pinch", "description": "", "duration": 1.75}
    ])
    row = tsv.strip().split("\n")[1].split("\t")
    assert row[0] == "2.500", row[0]
    assert row[1] == "1.750", row[1]
    assert row[2] == "pinch", row[2]


def test_events_tsv_has_bids_header():
    header = app.create_events_tsv([]).strip().split("\t")
    assert header == ["onset", "duration", "trial_type", "description"], header


def test_software_filters_reports_what_was_applied():
    """The sidecar must not claim data is unprocessed when it is not."""
    original = (app.bandpass_enabled, app.smoothing_enabled, app.downsample_factor)
    try:
        app.bandpass_enabled = True
        app.smoothing_enabled = True
        app.downsample_factor = 4
        described = app.describe_software_filters("emg")
        assert isinstance(described, dict), described
        assert "Bandpass filter" in described
        assert described["Bandpass filter"]["lower cutoff (Hz)"] == app.lowcut
        assert "Smoothing" in described
        assert described["Downsampling"]["factor"] == 4
    finally:
        app.bandpass_enabled, app.smoothing_enabled, app.downsample_factor = original


def test_emg_recording_is_not_notched():
    """The power-line notch is EEG-only; notching EMG removes muscle content."""
    described = app.describe_software_filters("emg")
    if isinstance(described, dict):
        assert "Notch filter" not in described, described
    assert "Notch filter" in app.describe_software_filters("eeg")


def test_dataset_description_is_written():
    """Without this file the folder is not a valid BIDS dataset."""
    import json
    path = app.ensure_dataset_description()
    assert os.path.exists(path), path
    with open(path) as handle:
        content = json.load(handle)
    assert content["Name"]
    assert content["BIDSVersion"]


def test_marker_open_and_close_roundtrip():
    app.recording_active = True
    app.recording_start_time = time.time()
    with app.event_lock:
        app.event_markers.clear()

    client = app.app.test_client()
    assert client.post("/add-marker", json={"label": "grip"}).status_code == 200
    time.sleep(0.05)

    closed = client.post("/end-marker", json={"label": "grip"}).get_json()
    assert closed["status"] == "marker_closed", closed
    assert closed["marker"]["duration"] > 0

    # Closing something that was never opened is an error, not a silent success.
    missing = client.post("/end-marker", json={"label": "never_opened"})
    assert missing.status_code == 404, missing.status_code


def test_marker_rejects_nonsense_duration():
    app.recording_active = True
    app.recording_start_time = time.time()
    client = app.app.test_client()
    result = client.post("/add-marker", json={"label": "x", "duration": "soon"}).get_json()
    # An unparseable duration leaves the marker open rather than inventing a number.
    assert result["marker"]["duration"] is None, result


def test_sidecar_matches_the_recording_it_describes():
    """Stale Metadata Editor rows must not contradict the file next to them.

    The editor rows are filled in when the page loads. If the operator then
    changes Subject / Run or the channel count (as happens with the 6-channel
    bracelet, whose UI default is 8), the sidecar used to keep the old values.
    """
    import json
    import random

    # Rows exactly as the page would have loaded them: subject 01, 8 channels.
    stale_rows = app.get_default_metadata_fields(
        "emg", 500.0, 8, task_label="resting", subject="01", session="01", run="01")

    channels, samples = 6, 1000
    data = [[random.gauss(0, 50) for _ in range(samples)] for _ in range(channels)]

    paths = app.save_bids_recording(
        data, [], "98", "01", "grasp", "02", 500.0, "emg", stale_rows)

    with open(paths["json"]) as handle:
        sidecar = json.load(handle)
    with open(paths["channels"]) as handle:
        channel_rows = handle.read().strip().split("\n")[1:]

    assert len(channel_rows) == channels, len(channel_rows)
    assert sidecar["EMGChannelCount"] == channels, sidecar["EMGChannelCount"]
    assert sidecar["NumberOfChannels"] == channels, sidecar["NumberOfChannels"]
    assert sidecar["Subject"] == "98", sidecar["Subject"]
    assert sidecar["Run"] == "02", sidecar["Run"]
    assert sidecar["TaskName"] == "grasp", sidecar["TaskName"]
    assert sidecar["SamplingFrequency"] == 500.0, sidecar["SamplingFrequency"]
    # The keys BIDS requires for the emg data type are present, and no EEG keys leaked in.
    assert sidecar["EMGReference"] and sidecar["EMGPlacementScheme"]
    assert not [k for k in sidecar if k.startswith("EEG")], [k for k in sidecar if k.startswith("EEG")]


def test_a_row_the_operator_switched_off_stays_off():
    """Correcting stale rows must never re-add a row that was deliberately removed."""
    import json
    import random

    rows = app.get_default_metadata_fields(
        "emg", 500.0, 8, task_label="resting", subject="01", session="01", run="01")
    for row in rows:
        if row["key"] in ("Subject", "NumberOfChannels"):
            row["include"] = False

    data = [[random.gauss(0, 50) for _ in range(1000)] for _ in range(6)]
    paths = app.save_bids_recording(data, [], "97", "01", "grasp", "01", 500.0, "emg", rows)
    with open(paths["json"]) as handle:
        sidecar = json.load(handle)
    assert "Subject" not in sidecar, sidecar.get("Subject")
    assert "NumberOfChannels" not in sidecar, sidecar.get("NumberOfChannels")


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures.append((test.__name__, exc))
            print(f"  FAIL  {test.__name__}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
