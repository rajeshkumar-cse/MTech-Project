import os
import wfdb
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset
from preprocessing import apply_filter, normalize, augment

class PTBXLDataset(Dataset):
    def __init__(self, data_dir, task='superdiagnostic', split='train', sampling_rate=500, augment_prob=0.0):
        """
        PyTorch Dataset for PTB-XL.
        Args:
            data_dir (str): Path to PTB-XL dataset root.
            task (str): Classification task (default: 'superdiagnostic').
            split (str): 'train', 'val', or 'test'.
            sampling_rate (int): Sampling rate (100 or 500).
            augment_prob (float): Probability of applying augmentation (0-1).
        """
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.augment_prob = augment_prob
        
        # Load database
        self.df = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
        
        # Filter by sampling rate if needed (though we load correct file path later)
        # self.df = self.df[self.df.sampling_rate == sampling_rate]

        # Parse scp_codes
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load SCP statements for label mapping
        agg_df = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
        
        # Select diagnostic class based on 'diagnostic_class' column in scp_statements.csv
        # We focus on the 5 superclasses: NORM, MI, STTC, CD, HYP
        if task == 'superdiagnostic':
            self.class_map = agg_df[agg_df.diagnostic == 1]
        
        # Filter samples that have at least one label in our target classes
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in self.class_map.index:
                    cls = self.class_map.loc[key].diagnostic_class
                    if isinstance(cls, str): # Verify it's not NaN
                         tmp.append(cls)
            return list(set(tmp))

        # Apply label mapping
        self.df['diagnostic_superclass'] = self.df.scp_codes.apply(aggregate_diagnostic)
        
        # Split data based on 'strat_fold' column (provided by dataset authors for standard splits)
        # Train: folds 1-8, Val: fold 9, Test: fold 10
        if split == 'train':
            self.df = self.df[self.df.strat_fold <= 8]
        elif split == 'val':
            self.df = self.df[self.df.strat_fold == 9]
        elif split == 'test':
            self.df = self.df[self.df.strat_fold == 10]
        
        # Binarize labels (Multi-hot encoding)
        self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'AFib']
        
        # Extract AFib from sub-diagnostics (AFIB or AFLT codes)
        self.df['has_afib'] = self.df.scp_codes.apply(
            lambda x: 1 if ('AFIB' in x or 'AFLT' in x) else 0
        )
        
        # Encode labels including AFib
        self.df['y'] = self.df.apply(
            lambda row: self.encode_labels_with_afib(row['diagnostic_superclass'], row['has_afib']),
            axis=1
        )
        
        # Keep only samples with at least one label (some might be empty after filtering)
        # self.df = self.df[self.df.y.apply(lambda x: np.sum(x) > 0)]

        self.records = self.df.index.tolist()
        self.labels = np.array(self.df.y.tolist())

    def encode_labels(self, labels):
        y = np.zeros(len(self.classes), dtype=np.float32)
        for label in labels:
            if label in self.classes:
                idx = self.classes.index(label)
                y[idx] = 1.0
        return y
    
    def encode_labels_with_afib(self, superclasses, has_afib):
        """Encode 5 superclasses + AFib as 6th class"""
        y = np.zeros(len(self.classes), dtype=np.float32)
        # Encode first 5 classes (superclasses)
        for label in superclasses:
            if label in self.classes[:5]:  # Only first 5
                idx = self.classes.index(label)
                y[idx] = 1.0
        # Encode AFib (6th class, index 5)
        y[5] = float(has_afib)
        return y

    def load_raw_data(self, df, idx):
        # Depending on sampling_rate, load from headers
        # Filename example: records500/00000/00001_hr
        # Filename provided in 'filename_hr' for 500Hz, 'filename_lr' for 100Hz
        if self.sampling_rate == 500:
            filename = df.loc[idx].filename_hr
        else:
            filename = df.loc[idx].filename_lr
            
        path = os.path.join(self.data_dir, filename)
        
        # Load signal using wfdb
        try:
            signal, meta = wfdb.rdsamp(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((5000, 12)) # Return dummy/cheated zero signal on failure
            
        return signal

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ecg_id = self.records[idx]
        
        # Load Raw Signal
        signal = self.load_raw_data(self.df, ecg_id)

        # Handle NaNs
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)

        # Transpose to (Channels, Length) -> (12, 5000) for PyTorch Conv1D
        signal = signal.T 

        # Preprocessing
        signal = apply_filter(signal, fs=self.sampling_rate)
        signal = normalize(signal)

        # Augmentation (only for training)
        if self.augment_prob > 0 and np.random.rand() < self.augment_prob:
             signal = augment(signal)

        return signal.astype(np.float32), self.labels[idx].astype(np.float32)
