#!/usr/bin/env python3
"""
Intelligent Frame Extractor
Selects the most diverse/different frames from each video
Maximizes information content for better training
"""
import cv2
import numpy as np
from pathlib import Path
import logging
import time
import json
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class IntelligentFrameExtractor:
    def __init__(self, resolution=768, num_frames=16, color_mode='rgb', sample_rate=2):
        """
        Args:
            resolution: Output frame size
            num_frames: Number of frames to extract
            color_mode: 'rgb' or 'grayscale'
            sample_rate: Sample every N frames for analysis (2 = every other frame)
        """
        self.resolution = resolution
        self.num_frames = num_frames
        self.color_mode = color_mode
        self.sample_rate = sample_rate
        logging.info(f"Intelligent Extractor: {resolution}x{resolution}, {num_frames} diverse frames, {color_mode}")

    def compute_frame_features(self, frame):
        """Extract features from a frame for similarity comparison"""
        # Resize for faster computation
        small = cv2.resize(frame, (64, 64))
        
        # Multiple features for diversity
        features = []
        
        # 1. Color histogram (or intensity for grayscale)
        if len(small.shape) == 3:
            for i in range(3):
                hist = cv2.calcHist([small], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
        else:
            hist = cv2.calcHist([small], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 2. Edge detection (structure)
        edges = cv2.Canny(small, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [2], [0, 256])
        features.extend(edge_hist.flatten() * 10)  # Weight edges more
        
        # 3. Mean and std of regions (spatial information)
        h, w = small.shape[:2]
        regions = [
            small[:h//2, :w//2],     # Top-left
            small[:h//2, w//2:],     # Top-right
            small[h//2:, :w//2],     # Bottom-left
            small[h//2:, w//2:]      # Bottom-right
        ]
        for region in regions:
            features.append(np.mean(region))
            features.append(np.std(region))
        
        return np.array(features)

    def select_diverse_frames(self, frames, indices):
        """Select the most diverse frames using clustering"""
        if len(frames) <= self.num_frames:
            return frames, indices
        
        logging.info(f"  Selecting {self.num_frames} most diverse frames from {len(frames)} candidates")
        
        # Extract features from all frames
        features = []
        for frame in frames:
            feat = self.compute_frame_features(frame)
            features.append(feat)
        features = np.array(features)
        
        # Method 1: K-means clustering
        if len(frames) > self.num_frames * 2:
            # Use clustering for large sets
            kmeans = KMeans(n_clusters=self.num_frames, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            selected_indices = []
            for cluster_id in range(self.num_frames):
                # Get frames in this cluster
                cluster_frames = np.where(labels == cluster_id)[0]
                if len(cluster_frames) > 0:
                    # Select frame closest to cluster center
                    center = kmeans.cluster_centers_[cluster_id]
                    distances = [np.linalg.norm(features[i] - center) for i in cluster_frames]
                    best_idx = cluster_frames[np.argmin(distances)]
                    selected_indices.append(best_idx)
            
            # Fill remaining slots if needed
            while len(selected_indices) < self.num_frames:
                for i in range(len(frames)):
                    if i not in selected_indices:
                        selected_indices.append(i)
                        break
                        
        else:
            # Method 2: Greedy diverse selection for smaller sets
            selected_indices = [0]  # Start with first frame
            
            while len(selected_indices) < self.num_frames:
                max_min_dist = -1
                best_idx = -1
                
                # Find frame with maximum minimum distance to selected frames
                for i in range(len(frames)):
                    if i in selected_indices:
                        continue
                    
                    # Compute minimum distance to already selected frames
                    min_dist = float('inf')
                    for j in selected_indices:
                        dist = np.linalg.norm(features[i] - features[j])
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = i
                
                if best_idx >= 0:
                    selected_indices.append(best_idx)
                else:
                    break
        
        # Sort indices to maintain temporal order
        selected_indices.sort()
        
        selected_frames = [frames[i] for i in selected_indices]
        selected_frame_indices = [indices[i] for i in selected_indices]
        
        return selected_frames, selected_frame_indices

    def extract_video(self, video_path):
        """Extract the most diverse frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            cap.release()
            return None

        # Sample frames for analysis (not all frames to save memory)
        sample_indices = list(range(0, total_frames, self.sample_rate))
        if len(sample_indices) > 200:  # Cap at 200 frames for analysis
            sample_indices = np.linspace(0, total_frames-1, 200, dtype=int).tolist()

        # Read sampled frames
        sampled_frames = []
        frame_indices = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                # Process frame based on color mode
                if self.color_mode == 'rgb':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:  # grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize
                frame = cv2.resize(frame, (self.resolution, self.resolution))
                sampled_frames.append(frame)
                frame_indices.append(idx)

        cap.release()

        if len(sampled_frames) < self.num_frames:
            return None

        # Select most diverse frames
        selected_frames, selected_indices = self.select_diverse_frames(sampled_frames, frame_indices)
        
        if len(selected_frames) == self.num_frames:
            return np.array(selected_frames, dtype=np.uint8)
        return None

    def process_dataset(self, input_dir, output_dir):
        """Process entire dataset with intelligent frame selection"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Collect all videos
        all_videos = []
        for class_dir in input_path.iterdir():
            if class_dir.is_dir():
                videos = list(class_dir.glob('*.mp4'))
                all_videos.extend([(v, class_dir.name) for v in videos])
                logging.info(f"Found {len(videos)} videos in {class_dir.name}")

        logging.info(f"Total videos: {len(all_videos)} - Using INTELLIGENT frame selection")

        processed = 0
        failed = 0
        start_time = time.time()

        for i, (video_path, class_name) in enumerate(all_videos):
            try:
                logging.info(f"Processing {video_path.name} - Finding most diverse frames...")
                frames = self.extract_video(video_path)
                if frames is not None:
                    # Save as compressed numpy
                    output_file = output_path / class_name / f"{video_path.stem}.npz"
                    output_file.parent.mkdir(exist_ok=True)
                    np.savez_compressed(output_file, frames=frames)
                    processed += 1
                else:
                    failed += 1

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (len(all_videos) - i - 1) / rate / 60
                    logging.info(f"Progress: {i+1}/{len(all_videos)} | Success: {processed} | Rate: {rate:.1f} v/s | ETA: {eta:.1f}m")

            except Exception as e:
                logging.error(f"Error processing {video_path}: {e}")
                failed += 1

        # Save metadata
        metadata = {
            'extraction_method': 'intelligent_diverse',
            'resolution': self.resolution,
            'num_frames': self.num_frames,
            'color_mode': self.color_mode,
            'total_videos': len(all_videos),
            'processed': processed,
            'failed': failed,
            'time_seconds': time.time() - start_time
        }

        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Intelligent Extraction Complete! Processed: {processed}, Failed: {failed}")
        logging.info(f"Time: {(time.time() - start_time)/60:.1f} minutes")
        logging.info("Frames selected for maximum diversity - better training data!")

        return processed, failed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Intelligent frame extraction')
    parser.add_argument('input_dir', help='Input directory with videos')
    parser.add_argument('output_dir', help='Output directory for frames')
    parser.add_argument('--resolution', type=int, default=768, help='Output resolution')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to extract')
    parser.add_argument('--color', choices=['rgb', 'grayscale'], default='rgb', help='Color mode')
    parser.add_argument('--sample-rate', type=int, default=2, help='Sample every N frames')
    
    args = parser.parse_args()
    
    extractor = IntelligentFrameExtractor(
        resolution=args.resolution,
        num_frames=args.num_frames,
        color_mode=args.color,
        sample_rate=args.sample_rate
    )
    
    extractor.process_dataset(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()