from utils import process_sequence, classifier_model, detector_model
import numpy as np
import argparse


def predict_riboswitch(seq):
    seq = process_sequence(seq)
    p = detector_model.predict(seq)
    d_pred = np.argmax(p)
    if not d_pred:
        return {
            'is_riboswitch': False,
            'annotations': None,
            'confidence': f"{p[0][d_pred]*100:.2f}",
        }

    seq = process_sequence(seq, for_classifier=True)
    p = classifier_model.predict(seq)
    c_pred = np.argmax(p)

    return {
        'is_riboswitch': True,
        'annotations': f"Ribo-{c_pred}",
        'confidence': f"{p[0][c_pred]*100:.2f}",
    }


parser = argparse.ArgumentParser(
    prog='RiboAnnot CLI',
    description=
    'Detect, classify, and annotate riboswitch in a given genomic sequence.')

parser.add_argument(
    '-w',
    '--width',
    default=250,
    type=int,
    help="Specifies the sequence length to consider for detecting riboswitches."
)
parser.add_argument(
    '-s',
    '--skip',
    default=50,
    type=int,
    help=
    "Specifies the skip when tiling the sequence. Eg: skip 50 will tile, 1 - 250, 50 - 300, etc."
)
parser.add_argument(
    '-b',
    '--begin',
    default=0,
    type=int,
    help="Specified the starting index (starts with 0) position to tile from.")
parser.add_argument(
    '-e',
    '--end',
    default=None,
    type=int,
    help="Specifies the ending index (starts with 0) position to tile upto.")

parser.add_argument("sequence",
                    type=str,
                    help="The genome sequencec to annotate for riboswitch")

args = parser.parse_args()

for i in range(args.begin, args.end or len(args.sequence), args.skip):
    seq = args.sequence[i:i + args.skip]
    pred = predict_riboswitch(seq)
    if pred['is_riboswitch']:
        print("Riboswitch found at:", i, "to:", i + args.skip)
        print("Riboswitch details -",
              f"Type:{pred['annotations']}, {pred['confidence']}%")
