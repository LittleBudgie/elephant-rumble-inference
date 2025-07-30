#!/usr/bin/env python
import argparse
import os
import tempfile
import time
import torch
from .aves_torchaudio_wrapper import AvesTorchaudioWrapper
from .elephant_rumble_classifier import ElephantRumbleClassifier
from .audio_file_processor import AudioFileProcessor
from .audio_file_visualizer import AudioFileVisualizer
from .raven_file_helper import RavenFileHelper
from .raven_file_helper import RavenLabel
from pydub import AudioSegment

# consider: https://www.youtube.com/watch?v=Qw9TmrAIS6E for demos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TODO - try Apple MPS

print(f"using {DEVICE}")
if DEVICE == "cpu":
    print(
        "WARNING - this can be extremely slow on the CPU - prepare to wait a long time"
    )
    print("   Recommend running with --limit-audio-hours=1 when on CPU.")

# Windows workarounds
if os.name == 'nt':  # Check if running on Windows
    # touch.hub.get_dir() returns a unix-like path
    print("""########### ATTEMPTING WINDOWS WORKAROUNDS ########""")
    cachedir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "torch",
        "hub"
    )
    torch.hub.set_dir(cachedir)
    print(torch.hub.get_dir())

def parse_args():

    usage = r"""

        elephant-rumble-inference --save-raven data/*.wav 

        elephant-rumble-inference \
            --visualizations-per-audio-file=5 --duration-of-visualizaton=60 \
            --load-labels ~/proj/elephantlistening/data/Rumble \
            --save-dir /tmp \
            ~/proj/elephantlistening/data/Rumble/Training/Sounds/*.wav 

        elephant-rumble-inference --help

    """
    parser = argparse.ArgumentParser(
        description="Find elephant rumbles in an audio clip", 
        usage=usage,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter

    )
    parser.add_argument("--model", type=str, help="Specify the model name. Recommended '2024(full list here: http://0ape.com/pretrained_models/ )")
    parser.add_argument("input_files", nargs="*", help="List of input files")
    parser.add_argument(
        "-o", "--save-dir",
        type=str,
        default="output",
        help="directory to save outputs",
    )
    parser.add_argument(
        "-v", "--visualizations-per-audio-file",
        type=int,
        default=1,
        help="visualiztions are slow so be patient if you pick more than 1",
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default='2024-07-03.pth',
        help="Name of the model.  Recommended 2024-06-29.pth, 2024-06-30.pth, 2024-07-03.pth, or 2024-07-09.pth. Full list of available models here: http://0ape.com/pretrained_models/ )",
    )
    parser.add_argument(
        "-d", "--duration-of-visualizations",
        type=int,
        default=1,
        help="Minutes of audio for a visualization. 15 is nice for wide monitor, 60 is interesting if you don't mind horizontal scrolling",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        default=True,
        help="Save classification scores to a file",
    )
    parser.add_argument(
        "-r", "--save-raven",
        action="store_true",
        default=True,
        help="Save a raven file with found labels",
    )
    parser.add_argument(
        "-l", "--load-labels-from-raven-file-folder",
        type=str,
        help="show labels from existing raven files",
    )
    parser.add_argument(
        "--limit-audio-hours",
        type=int,
        default=24,
        help="Limit audio hours (Recommend setting to 1 for CPU).",
    )
    #added
    parser.add_argument(
        "--merge-files-in-dir",
        type=bool,
        default=False,
        help="Merges all audio files in each subdirectory of input path in lexographical order. Outputs merged audio file in subdirectory and raven selection table for merged audio file in output."
    )

    args = parser.parse_args()

    if len(args.input_files) == 0:
        print("Need at least one audio file.\n\n\n",usage)
    return args


def initialize_models(model_name):
    # model_name = 'elephant_rumble_classifier_500_192_2024-06-29T22:51:14.720487_valloss=5.83.pth'
    # model_name = 'elephant_rumble_classifier_500_192_2024-06-30T02:01:33.715741_valloss=6.76.pth'
    # model_name = 'elephant_rumble_classifier_500_192_2024-06-30T02:22:33.598037_valloss=6.55.pth'
    # model_name = "elephant_rumble_classifier_500_192_2024-07-03T01:27:40.424353_from_train_folder_valloss=5.55.pth"
    # model_name = "best.pth" # windows likes short names
    atw = AvesTorchaudioWrapper().to(DEVICE)
    erc = ElephantRumbleClassifier().to("cpu")
    erc.load_pretrained_weights(model_name)
    print(f"Using weights from {erc.model_name}")
    atw.eval()
    erc.eval()
    return atw, erc

def get_audio_paths_from_dir(input_dirs):
    audio_paths = []
    for dir in input_dirs:
        for sub_dir in os.walk(dir):
            files_in_sub_dir = sub_dir[2]
            for file in files_in_sub_dir:
                file_type = os.path.splitext(file)[-1].lower()
                if (file_type == ".wav"):
                    audio_paths.append(sub_dir[0] + "/" + file)
    return audio_paths

def get_audio_paths_merged(input_dirs):
    audio_paths = []
    for dir in input_dirs:
        for sub_dir in os.walk(dir):
            files_in_sub_dir = sub_dir[2]
            for file in files_in_sub_dir:
                file_type = os.path.splitext(file)[-1].lower()
                merged_audio_file = None
                if (file_type == ".wav"):
                    curr_file_path = sub_dir[0] + "/" + file
                    audio_file = AudioSegment.from_file(curr_file_path, format = "wav")
                    if (merged_audio_file is None):
                        merged_audio_file = audio_file
                    else:
                        merged_audio_file = merged_audio_file + audio_file
            merged_file_path = sub_dir[0] + "/MERGED.wav"
            merged_audio = merged_audio_file.export(merged_file_path, format = "wav")
            audio_paths.append(merged_file_path)
    return audio_paths



def classify_audio_file(afp, audio_file, limit_audio_hours, save_file_path):
    with torch.inference_mode():
        t0 = time.time()
        scores = afp.classify_wave_file_for_rumbles(
            audio_file, limit_audio_hours=limit_audio_hours
        )
        if save_file_path:
            torch.save(scores, save_file_path)
        t1 = time.time()
        print(f"{t1-t0} to classify {audio_file} [limited to {limit_audio_hours} hours]")
        return scores


def save_raven_file(audio_file, scores, raven_file, afp):
    rfh = RavenFileHelper()
    continuous_segments = rfh.find_continuous_segments(scores[:, 1] - scores[:, 0] > 0)
    long_enough_segments = rfh.find_long_enough_segments(continuous_segments, n=3)
    print(
        f"of the {len(continuous_segments)} segments classified as rumbles ",
        f"only {len(long_enough_segments)} were over a second long.",
    )
    raven_labels = []
    for s0, s1 in long_enough_segments:
        bt = afp.score_index_to_time(s0)
        et = afp.score_index_to_time(s1)
        lf, hf = 5, 250
        duration = et - bt
        tag1 = tag2 = tag3 = notes = "generated_by_classifier"
        score = 0.9  # TODO get the score from the model
        ravenfile = "classifier_generated_raven_file.raven"
        rl = RavenLabel(
            bt,
            et,
            lf,
            hf,
            duration,
            audio_file,
            tag1,
            tag2,
            tag3,
            notes,
            score,
            ravenfile,
        )
        raven_labels.append(rl)
    rfh.write_raven_file(raven_labels, raven_file)


def choose_save_locations(args, audio_file):
    audio_file_without_path = os.path.basename(audio_file)
    save_dir = args.save_dir
    score_file = raven_file = visualization_dir = None
    os.makedirs(save_dir, exist_ok=True)
    if args.save_scores:
        score_file = os.path.join(save_dir, audio_file_without_path + ".scores.pt")
    if args.save_raven:
        raven_file = os.path.join(save_dir, audio_file_without_path + ".raven.txt")
    if args.visualizations_per_audio_file > 0:
        visualization_dir = save_dir
    return score_file, raven_file, visualization_dir



def get_windows_torch_hub_dir():
    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "torch", "hub")



def main():
    args = parse_args()
    
    atw, erc = initialize_models(args.model_name)
    afp = AudioFileProcessor(atw, erc, device=DEVICE)

    audio_paths = []
    if args.merge_files_in_dir:
        audio_paths = get_audio_paths_merged(args.input_files)
    else:
        audio_paths = get_audio_paths_from_dir(args.input_files)
    
    print(f"Input files: {audio_paths}")
    for audio_file in audio_paths:
        audio_file_without_path = os.path.basename(audio_file)

        score_file, raven_file, visualization_dir = choose_save_locations(
            args, audio_file
        )
        print(score_file, raven_file, visualization_dir)
        if raven_file and os.path.exists(raven_file):
            print(f"skipping {raven_file} -- already exists")
            print(f"(delete {raven_file} if you want to re-process it")
            continue

        t0 = time.time()
        scores = classify_audio_file(
            afp, audio_file, args.limit_audio_hours, score_file
        )
        t1 = time.time()

        if args.save_raven:
            save_raven_file(audio_file_without_path, scores, raven_file, afp)

        t2 = time.time()

        if visualization_dir:
            duration_of_visualizations_min  = args.duration_of_visualizations
            duration_of_visualizations_secs = args.duration_of_visualizations * 60

            print("Rendering visualizations...")
            if args.load_labels_from_raven_file_folder:
                rfh = RavenFileHelper(args.load_labels_from_raven_file_folder)
                lbls = rfh.get_all_labels_for_wav_file(audio_file_without_path)
            else:
                lbls = []

            rfh = RavenFileHelper()
            continuous_segments = rfh.find_continuous_segments(scores[:, 1] - scores[:, 0] > 0)
            long_enough_segments = rfh.find_long_enough_segments(continuous_segments, n=3)
            interesting_seconds = [afp.score_index_to_time(bt) for bt,et in long_enough_segments]
            from collections import Counter
            # 5 minute spectrograms are easier to handle than hour long ones.
            interesting_times = Counter([int(sec/duration_of_visualizations_secs)*duration_of_visualizations_secs for sec in interesting_seconds])
            #for element, count in interesting_times.most_common():
            #    print(f"{element}: {count}")
            num_vis =0
            with torch.inference_mode():
                for interesting_time, count in interesting_times.most_common():
                    hour   = (interesting_time) // 60 // 60
                    minute = (interesting_time // 60) % 60
                    dttm   = f"{hour:02}:{minute:02}:00"
                    vis_filename = f"{audio_file_without_path}_{dttm}.png"
                    vis_path = os.path.join(visualization_dir,vis_filename)
                    if os.name == 'nt': # windows doesn't allow a filename to have an iso time in it?
                        vis_path = os.path.join(visualization_dir,f"{audio_file_without_path}_{hour:02}_{minute:02}_00.png")
                    AudioFileVisualizer().visualize_audio_file_fragment(
                        f"{audio_file_without_path}, Starting at {dttm}, Classified by {erc.model_name}",
                        vis_path,
                        audio_file,
                        scores[:, 1],
                        scores[:, 0],
                        afp,
                        start_time=interesting_time,
                        end_time=interesting_time+duration_of_visualizations_secs,
                        width = 4 * duration_of_visualizations_min,
                        height = 4,
                        colormap="clean",
                        labels=lbls,
                    )
                    num_vis += 1
                    if num_vis >= args.visualizations_per_audio_file:
                        print(f"only doing {num_vis} visualization per file")
                        break

main()