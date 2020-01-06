import os
import glob
import torch
import pickle
import music21
import numpy as np

from torch.utils.data import DataLoader

class MusicData():
    
    def __init__(self, data_folder, build=True, seq_len=32, intervals=range(1)):
        self.build = build
        self.data_folder = data_folder
        self.intervals = intervals
        self.seq_len = seq_len
        
        # Get music file list
        self.music_list, self.parser = self.get_music_list()
        
        # Build dataset
        self.notes, self.durations = self.build_dataset()
        
        # Length of sequences
        self.n_notes = len(sorted(set(self.notes)))
        self.n_durations = len(sorted(set(self.durations)))
        
        # Create lookups for notes
        self.idx2note, self.note2idx = self.create_lookups(self.notes)
        
        # Create lookups for durations
        self.idx2duration, self.duration2idx = self.create_lookups(self.durations)
        
        # Get distinct notes and durations
        self.distincts = [self.get_distinct(self.notes), 
                         self.get_distinct(self.durations)]
        
        # Prepare sequences for model
        (self.pitch_input, self.duration_input), (
                self.pitch_output, self.duration_output) = self.prepare_sequences(
                self.distincts, seq_len=32)    
        
        # Convert sequences to Torch Tensors
        self.pitch_input = self.to_tensor(self.pitch_input)
        self.duration_input = self.to_tensor(self.duration_input)
        self.pitch_output = self.to_tensor(self.pitch_output)
        self.duration_output = self.to_tensor(self.duration_output)
        
    def data_loader(self, batch_size=32):
        pitch_input = DataLoader(self.pitch_input, batch_size=batch_size)
        duration_input = DataLoader(self.duration_input, batch_size=batch_size)
        pitch_output = DataLoader(self.pitch_output, batch_size=batch_size)
        duration_output = DataLoader(self.duration_output, batch_size=batch_size)
        
        return pitch_input, duration_input, pitch_output, duration_output
    
    def to_tensor(self, data):
        """
        Function that converts given data to torch tensor
        """
        return torch.tensor(data).long()
        
    def get_music_list(self):
        """
        Function that returns list of music object file available
        and parser for it
        """
        file_list = glob.glob(os.path.join(self.data_folder, "*.mid"))
        parser = music21.converter
        
        return file_list, parser
    
    def build_dataset(self):
        """
        Function that builds dataset if it is needed and dumps files to pickle loader
        or
        it loads already generated data into lists and returns them
        """
        notes = []
        durations = []
        
        # If it is necessary to build data from scratch
        if self.build == True:
            for i, file in enumerate(self.music_list):
                print(i+1, "Parsing {}".format(file))
                original_score = self.parser.parse(file).chordify()
                
                for interval in self.intervals:
                    
                    score = original_score.transpose(interval)
                    
                    notes.extend(['START'] * self.seq_len)
                    durations.extend([0] * self.seq_len)
                    
                    for element in score.flat:
                        
                        if isinstance(element, music21.note.Note):
                            if element.isRest:
                                notes.append(str(element.name))
                                durations.append(element.duration.quarterLength)
                            else:
                                notes.append(str(element.nameWithOctave))
                                durations.append(element.duration.quarterLength)
                                
                        if isinstance(element, music21.chord.Chord):
                            notes.append(".".join(
                                    n.nameWithOctave for n in element.pitches))
                            durations.append(element.duration.quarterLength)
            with open(os.path.join(self.data_folder, 'notes'), 'wb') as f:
                pickle.dump(notes, f)
            with open(os.path.join(self.data_folder, 'durations'), 'wb') as f:
                pickle.dump(durations, f)
        # To load alreay built data
        else:
            print("Data Successfully loaded!")
            with open(os.path.join(self.data_folder, 'notes'), 'rb') as f:
                notes = pickle.load(f)
            with open(os.path.join(self.data_folder, 'durations'), 'rb') as f:
                durations = pickle.load(f)
        
        return notes, durations
    
    def create_lookups(self, data):
        """
        Function that creates dictionary for lookups
        index to data and vice-versa
        """
        idx2data = {i: d for i, d in enumerate(data)}
        data2idx = {}
        for i, d in idx2data.items():
            if d not in data2idx:
                data2idx[d] = i
        return idx2data, data2idx

    def get_distinct(self, elements):
        """
        Function that returns distinct elements
        and length of it
        """
        names = sorted(set(elements))
        n_elements = len(names)
        return (names, n_elements)

    def prepare_sequences(self, distincts, seq_len=32):
        """
        Function for sequence preparation, where lists
        are encoded and transformed to array of integers
        """
        (note_names, n_notes), (duration_names, n_durations) = distincts
        
        notes_input = []
        notes_output = []
        
        durations_input = []
        durations_output = []
        
        for i in range(len(self.notes) - seq_len):
            # extract input and output from notes
            notes_in = self.notes[i:i + seq_len]
            notes_out = self.notes[i + seq_len]
            # convert extracted notes into indexes
            notes_input.append([self.note2idx[note] for note in notes_in])
            notes_output.append(self.note2idx[notes_out])
            
            # extract input and output from durations
            durations_in = self.durations[i:i + seq_len]
            durations_out = self.durations[i + seq_len]
            # convert durations into indexes
            durations_input.append([self.duration2idx[dur] for dur in durations_in])
            durations_output.append(self.duration2idx[durations_out])        
            
        n_patterns = len(notes_input)
        
        # Reshape input 
        notes_input = np.reshape(notes_input, (n_patterns, seq_len))
        durations_input = np.reshape(durations_input, (n_patterns, seq_len))
        
        model_input = [notes_input, durations_input]
        
        # extract unique values from notes and durations
        notes_set = list(set(notes_output))
        durations_set = list(set(durations_output))
        
        # one hot encode notes_output
        notes_output_ohe = []
        for el in notes_output:
            row = [0 for _ in range(n_notes)]
            row[notes_set.index(el)] = 1
            notes_output_ohe.append(row)
            
        # one hot encode durations_output
        durations_output_ohe = []
        for el in durations_output:
            row = [0 for _ in range(n_durations)]
            row[durations_set.index(el)] = 1
            durations_output_ohe.append(row)
            
        model_output = [notes_output_ohe, durations_output_ohe]
        
        return (model_input, model_output)