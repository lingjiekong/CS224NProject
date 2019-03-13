import json
import os
from onsets-and-frames/dataset import PianoRollAudioDataset

Class ExtractNoteSpectra(PianoRollAudioDataset):

	def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE)
