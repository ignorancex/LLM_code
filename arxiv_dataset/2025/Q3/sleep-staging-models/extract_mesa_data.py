import os
import numpy as np
import pandas as pd
import pyedflib
from scipy import signal, interpolate
from scipy.signal import butter, filtfilt, cheby2
import h5py
from tqdm import tqdm
import warnings
import xml.etree.ElementTree as ET
from collections import Counter

warnings.filterwarnings('ignore')


class MESADataExtractor:
    def __init__(self, mesa_path, output_path, require_ecg=True):
        self.mesa_path = mesa_path
        self.output_path = output_path
        self.require_ecg = require_ecg  
        self.target_fs = 34.133333333  
        self.window_duration = 30  
        self.samples_per_window = 1024  
        self.target_hours = 10  
        self.target_windows = 1200  
        self.target_length = self.target_windows * self.samples_per_window  # 1,228,800 samples

       
        os.makedirs(output_path, exist_ok=True)

    def analyze_signal_labels(self, edf_dir):

        print("Analyzing signal labels in EDF files...")

        edf_files = [f for f in os.listdir(edf_dir) if f.endswith('.edf')]
        all_labels = []
        ecg_labels = []
        ppg_labels = []

        for edf_file in tqdm(edf_files[:100], desc="Analyzing signals"):  
            try:
                f = pyedflib.EdfReader(os.path.join(edf_dir, edf_file))
                labels = f.getSignalLabels()
                all_labels.extend(labels)


                for label in labels:
                    label_lower = label.lower()
                    if 'ecg' in label_lower or 'ekg' in label_lower:
                        ecg_labels.append(label)
                    if 'pleth' in label_lower or 'ppg' in label_lower:
                        ppg_labels.append(label)

                f.close()
            except:
                continue

        print("\nMost common signal labels:")
        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common(20):
            print(f"  {label}: {count}")

        print("\nECG-related labels found:")
        ecg_counts = Counter(ecg_labels)
        for label, count in ecg_counts.most_common():
            print(f"  {label}: {count}")

        print("\nPPG-related labels found:")
        ppg_counts = Counter(ppg_labels)
        for label, count in ppg_counts.most_common():
            print(f"  {label}: {count}")

    def extract_signals_from_edf(self, edf_file):

        try:
            f = pyedflib.EdfReader(edf_file)
            signal_labels = f.getSignalLabels()


            ppg_idx = None
            for idx, label in enumerate(signal_labels):
                label_lower = label.lower()
                if 'pleth' in label_lower or 'ppg' in label_lower:
                    ppg_idx = idx
                    break


            ecg_idx = None
            ecg_priorities = [
                lambda x: 'ecg' in x and ('ii' in x or '2' in x),  # ECG Lead II
                lambda x: 'ekg' in x and ('ii' in x or '2' in x),  # EKG Lead II
                lambda x: 'ecg' in x,  # Any ECG
                lambda x: 'ekg' in x,  # Any EKG
                lambda x: 'ekgr' in x,  # EKGR (found in some files)
            ]

            for priority_func in ecg_priorities:
                for idx, label in enumerate(signal_labels):
                    if priority_func(label.lower()):
                        ecg_idx = idx
                        break
                if ecg_idx is not None:
                    break

            if ppg_idx is None:
                print(f"No PPG signal found in {os.path.basename(edf_file)}")
                f.close()
                return None, None, None, None

            if ecg_idx is None and self.require_ecg:
                print(f"No ECG signal found in {os.path.basename(edf_file)}")
                f.close()
                return None, None, None, None


            ppg_signal = f.readSignal(ppg_idx)
            ppg_fs = f.getSampleFrequency(ppg_idx)

            if ecg_idx is not None:
                ecg_signal = f.readSignal(ecg_idx)
                ecg_fs = f.getSampleFrequency(ecg_idx)
            else:
                ecg_signal = None
                ecg_fs = None

            f.close()

            return ppg_signal, ecg_signal, ppg_fs, ecg_fs

        except Exception as e:
            print(f"Error reading {os.path.basename(edf_file)}: {e}")
            return None, None, None, None

    def preprocess_ppg(self, ppg_signal, original_fs):


        nyq = 0.5 * original_fs
        cutoff = 8 / nyq


        sos = signal.cheby2(N=8, rs=40, Wn=cutoff, btype='lowpass', output='sos')
        filtered_ppg = signal.sosfiltfilt(sos, ppg_signal)


        duration = len(filtered_ppg) / original_fs  
        n_samples = int(duration * self.target_fs)


        old_indices = np.linspace(0, len(filtered_ppg) - 1, len(filtered_ppg))
        new_indices = np.linspace(0, len(filtered_ppg) - 1, n_samples)
        downsampled_ppg = np.interp(new_indices, old_indices, filtered_ppg)


        mean = np.mean(downsampled_ppg)
        std = np.std(downsampled_ppg)


        clipped_ppg = np.clip(downsampled_ppg, mean - 3 * std, mean + 3 * std)


        wavppg = (clipped_ppg - np.mean(clipped_ppg)) / np.std(clipped_ppg)

        return wavppg

    def preprocess_ecg(self, ecg_signal, original_fs):

        nyq = 0.5 * original_fs
        low = 0.5 / nyq
        high = 40.0 / nyq

        if high >= 1:
            high = 0.99


        b, a = butter(4, [low, high], btype='band')
        filtered_ecg = filtfilt(b, a, ecg_signal)


        duration = len(filtered_ecg) / original_fs
        n_samples = int(duration * self.target_fs)

        old_indices = np.linspace(0, len(filtered_ecg) - 1, len(filtered_ecg))
        new_indices = np.linspace(0, len(filtered_ecg) - 1, n_samples)
        downsampled_ecg = np.interp(new_indices, old_indices, filtered_ecg)


        mean = np.mean(downsampled_ecg)
        std = np.std(downsampled_ecg)


        clipped_ecg = np.clip(downsampled_ecg, mean - 3 * std, mean + 3 * std)


        standardized_ecg = (clipped_ecg - np.mean(clipped_ecg)) / np.std(clipped_ecg)

        return standardized_ecg

    def pad_or_truncate_signal(self, signal, target_length):

        current_length = len(signal)

        if current_length >= target_length:

            return signal[:target_length]
        else:

            padding_length = target_length - current_length
            padding = np.zeros(padding_length)
            return np.concatenate([signal, padding])

    def parse_sleep_stages(self, xml_file):

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            scored_events = root.find('.//ScoredEvents')
            if scored_events is None:
                return None

            sleep_stages = []

            for event in scored_events.iter('ScoredEvent'):
                event_type_element = event.find('EventType')
                event_type = event_type_element.text if event_type_element is not None else None

                if event_type == 'Stages|Stages':
                    event_concept = event.find('EventConcept').text
                    start_time = float(event.find('Start').text)
                    duration = float(event.find('Duration').text)


                    if 'Wake' in event_concept:
                        stage = 0
                    elif 'Stage 1 sleep' in event_concept or 'Stage 2 sleep' in event_concept:
                        stage = 1  # Light sleep
                    elif 'Stage 3 sleep' in event_concept or 'Stage 4 sleep' in event_concept:
                        stage = 2  # Deep sleep
                    elif 'REM sleep' in event_concept:
                        stage = 3
                    else:
                        continue

                    sleep_stages.append({
                        'Start': start_time,
                        'Duration': duration,
                        'Stage': stage
                    })

            return pd.DataFrame(sleep_stages)

        except Exception as e:
            print(f"Error parsing {os.path.basename(xml_file)}: {e}")
            return None

    def expand_labels_to_windows(self, df_stages, total_duration):

        num_windows = int(np.ceil(total_duration / self.window_duration))
        labels = np.full(num_windows, -1, dtype=int) 

        for _, row in df_stages.iterrows():
            start_idx = int(row['Start'] // self.window_duration)
            end_idx = int((row['Start'] + row['Duration']) // self.window_duration)

            if end_idx > len(labels):
                end_idx = len(labels)

            labels[start_idx:end_idx] = row['Stage']

        return labels

    def pad_or_truncate_labels(self, labels, target_length):

        current_length = len(labels)

        if current_length >= target_length:

            return labels[:target_length]
        else:

            padding_length = target_length - current_length
            padding = np.full(padding_length, -1, dtype=int)
            return np.concatenate([labels, padding])

    def process_subject(self, edf_file, xml_file):

        ppg, ecg, ppg_fs, ecg_fs = self.extract_signals_from_edf(edf_file)

        if ppg is None:
            return None


        ppg_processed = self.preprocess_ppg(ppg, ppg_fs)


        if ecg is not None:
            ecg_processed = self.preprocess_ecg(ecg, ecg_fs)
        else:

            ecg_processed = np.zeros_like(ppg_processed)


        ppg_final = self.pad_or_truncate_signal(ppg_processed, self.target_length)
        ecg_final = self.pad_or_truncate_signal(ecg_processed, self.target_length)


        ppg_final = ppg_final[:self.target_length]
        ecg_final = ecg_final[:self.target_length]


        if os.path.exists(xml_file):
            df_stages = self.parse_sleep_stages(xml_file)
            if df_stages is not None:

                total_duration = len(ppg_processed) / self.target_fs


                labels = self.expand_labels_to_windows(df_stages, total_duration)


                labels_final = self.pad_or_truncate_labels(labels, self.target_windows)
            else:
                labels_final = np.full(self.target_windows, -1, dtype=int)
        else:
            labels_final = np.full(self.target_windows, -1, dtype=int)


        try:
            ppg_windows = ppg_final.reshape(self.target_windows, self.samples_per_window)
            ecg_windows = ecg_final.reshape(self.target_windows, self.samples_per_window)
        except ValueError as e:
            print(f"Reshape error: {e}")
            print(f"PPG shape: {ppg_final.shape}, expected: ({self.target_windows}, {self.samples_per_window})")
            return None

        return ppg_windows, ecg_windows, labels_final, (ecg is not None)

    def process_all_subjects(self, subject_list=None, analyze_first=False):

        edf_dir = os.path.join(self.mesa_path, 'polysomnography', 'edfs')
        xml_dir = os.path.join(self.mesa_path, 'polysomnography', 'annotations-events-nsrr')


        if analyze_first:
            self.analyze_signal_labels(edf_dir)
            return 0

        edf_files = [f for f in os.listdir(edf_dir) if f.endswith('.edf')]

        if subject_list:
            edf_files = [f for f in edf_files if any(subj in f for subj in subject_list)]

        all_ppg_windows = []
        all_ecg_windows = []
        all_labels = []
        subject_ids = []
        has_real_ecg = [] 

        subjects_with_ecg = 0
        subjects_without_ecg = 0
        failed_subjects = 0

        for edf_filename in tqdm(edf_files, desc="Processing subjects"):

            edf_path = os.path.join(edf_dir, edf_filename)


            subject_id = edf_filename.split('-')[2].split('.')[0]  # mesa-sleep-0001.edf -> 0001


            xml_filename = edf_filename.replace('.edf', '-nsrr.xml')
            xml_path = os.path.join(xml_dir, xml_filename)


            try:
                result = self.process_subject(edf_path, xml_path)

                if result is not None:
                    ppg_windows, ecg_windows, labels, has_ecg = result

                    if has_ecg:
                        subjects_with_ecg += 1
                    else:
                        subjects_without_ecg += 1

                    all_ppg_windows.append(ppg_windows)
                    all_ecg_windows.append(ecg_windows)
                    all_labels.append(labels)
                    subject_ids.extend([subject_id] * len(ppg_windows))
                    has_real_ecg.extend([has_ecg] * len(ppg_windows))
                else:
                    failed_subjects += 1
            except Exception as e:
                print(f"Error processing {edf_filename}: {e}")
                failed_subjects += 1
                continue

        print(f"\nProcessing summary:")
        print(f"  Subjects with ECG: {subjects_with_ecg}")
        print(f"  Subjects without ECG: {subjects_without_ecg}")
        print(f"  Failed subjects: {failed_subjects}")
        print(f"  Total subjects processed: {subjects_with_ecg + subjects_without_ecg}")


        if all_ppg_windows:
            all_ppg_windows = np.vstack(all_ppg_windows)
            all_ecg_windows = np.vstack(all_ecg_windows)
            all_labels = np.concatenate(all_labels)
            subject_ids = np.array(subject_ids)
            has_real_ecg = np.array(has_real_ecg)


            self.save_data_separate(all_ppg_windows, all_ecg_windows, all_labels, subject_ids, has_real_ecg)

            return len(all_ppg_windows)

        return 0

    def save_data_separate(self, ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg):

        ppg_file = os.path.join(self.output_path, 'mesa_ppg_with_labels.h5')
        print(f"\nSaving PPG data to {ppg_file}...")
        with h5py.File(ppg_file, 'w') as f:

            f.create_dataset('ppg', data=ppg_windows, compression='gzip',
                             chunks=(100, self.samples_per_window))
            f.create_dataset('labels', data=labels, compression='gzip')
            f.create_dataset('subject_ids', data=subject_ids.astype('S10'), compression='gzip')


            f.attrs['sampling_rate'] = self.target_fs
            f.attrs['window_duration'] = self.window_duration
            f.attrs['samples_per_window'] = self.samples_per_window
            f.attrs['total_windows'] = len(ppg_windows)
            f.attrs['total_subjects'] = len(np.unique(subject_ids))


        ecg_file = os.path.join(self.output_path, 'mesa_real_ecg.h5')
        print(f"Saving ECG data to {ecg_file}...")
        with h5py.File(ecg_file, 'w') as f:
            f.create_dataset('ecg', data=ecg_windows, compression='gzip',
                             chunks=(100, self.samples_per_window))
            f.create_dataset('has_real_ecg', data=has_real_ecg, compression='gzip')
            f.create_dataset('subject_ids', data=subject_ids.astype('S10'), compression='gzip')
            f.create_dataset('labels', data=labels, compression='gzip')  


            real_ecg_indices = np.where(has_real_ecg)[0]
            f.create_dataset('real_ecg_indices', data=real_ecg_indices, compression='gzip')
            f.attrs['windows_with_real_ecg'] = int(np.sum(has_real_ecg))
            f.attrs['windows_without_real_ecg'] = int(np.sum(~has_real_ecg))


        index_file = os.path.join(self.output_path, 'mesa_subject_index.h5')
        print(f"Creating index file {index_file}...")
        with h5py.File(index_file, 'w') as f:

            unique_subjects = np.unique(subject_ids)
            subject_group = f.create_group('subjects')

            for subj in unique_subjects:
                subj_str = subj.decode() if isinstance(subj, bytes) else str(subj)
                mask = subject_ids == subj
                indices = np.where(mask)[0]


                subj_group = subject_group.create_group(subj_str)
                subj_group.create_dataset('window_indices', data=indices)
                subj_group.attrs['n_windows'] = len(indices)
                subj_group.attrs['has_ecg'] = bool(has_real_ecg[indices[0]])


            f.attrs['total_subjects'] = len(unique_subjects)
            f.attrs['total_windows'] = len(ppg_windows)


        self.save_statistics(ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg)

        print("\nData saved successfully in separate files!")
        print(f"  PPG data: {ppg_file}")
        print(f"  ECG data: {ecg_file}")
        print(f"  Subject index: {index_file}")

    def save_statistics(self, ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg):

        valid_labels = labels[labels != -1]

        stats = {
            'total_windows': len(ppg_windows),
            'total_subjects': len(np.unique(subject_ids)),
            'windows_with_real_ecg': int(np.sum(has_real_ecg)),
            'windows_without_real_ecg': int(np.sum(~has_real_ecg)),
            'ppg_shape': ppg_windows.shape,
            'ecg_shape': ecg_windows.shape,
            'valid_labels': len(valid_labels),
            'label_distribution': dict(zip(*np.unique(valid_labels, return_counts=True))) if len(
                valid_labels) > 0 else {},
            'sampling_rate': self.target_fs,
            'window_duration': self.window_duration,
            'samples_per_window': self.samples_per_window,
            'file_structure': {
                'mesa_ppg_with_labels.h5': ['ppg', 'labels', 'subject_ids'],
                'mesa_real_ecg.h5': ['ecg', 'has_real_ecg', 'subject_ids', 'labels', 'real_ecg_indices'],
                'mesa_subject_index.h5': ['subjects/{subject_id}/window_indices']
            }
        }


        stats_file = os.path.join(self.output_path, 'data_stats.npy')
        np.save(stats_file, stats)


        stats_txt = os.path.join(self.output_path, 'data_stats.txt')
        with open(stats_txt, 'w') as f:
            f.write("MESA Sleep Data Processing Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total windows: {stats['total_windows']}\n")
            f.write(f"Total subjects: {stats['total_subjects']}\n")
            f.write(f"Windows with real ECG: {stats['windows_with_real_ecg']}\n")
            f.write(f"Windows without real ECG: {stats['windows_without_real_ecg']}\n")
            f.write(f"Valid labels: {stats['valid_labels']}\n")
            f.write(f"\nLabel distribution:\n")
            for label, count in stats['label_distribution'].items():
                stage_names = {0: 'Wake', 1: 'Light', 2: 'Deep', 3: 'REM'}
                f.write(f"  {stage_names.get(label, f'Stage{label}')}: {count}\n")
            f.write(f"\nSampling rate: {stats['sampling_rate']} Hz\n")
            f.write(f"Window duration: {stats['window_duration']} seconds\n")
            f.write(f"Samples per window: {stats['samples_per_window']}\n")

        print(f"\nStatistics saved to:")
        print(f"  {stats_file}")
        print(f"  {stats_txt}")


def main():

    MESA_PATH = "G:/mesa"  
    OUTPUT_PATH = "F:/python_project/mesa-x" 


    extractor = MESADataExtractor(MESA_PATH, OUTPUT_PATH, require_ecg=False)


    # extractor.process_all_subjects(analyze_first=True)


    # n_windows = extractor.process_all_subjects(subject_list=['0001', '0002', '0003'])


    n_windows = extractor.process_all_subjects()

    print(f"\nProcessing completed! Total windows: {n_windows}")


if __name__ == "__main__":
    main()
