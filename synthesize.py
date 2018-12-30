#!/usr/bin/env python3
import argparse
import os
import sys

from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer

synthesizer = Synthesizer()


def main(args):
    hparams.parse(args.hparams)
    print(hparams_debug_string(), file=sys.stderr)
    with open(args.text, 'r') as f:
        sentences = f.readlines()
        for sentence in sentences:
            sentence = sentence.strip()
    synthesizer.load(args.checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)
    for idx, sentence in enumerate(sentences):
        print(idx, sentence, file=sys.stderr)
        wav = synthesizer.synthesize(sentence)
        path = os.path.join(args.output_dir, '%d.wav' % idx)
        # wavfile.write(path, hparams.sample_rate, wav)
        with open(path, 'wb') as f:
            f.write(wav)
    print('Finish synthesis.', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
    parser.add_argument('--output_dir', default='output/', help='The output directory.')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--text', required=True, help='The text list which you want to synthesis.')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main(args)
