#!/usr/bin/env python3
"""
Frame Extraction for 768x768 Grayscale
"""
import cv2
import numpy as np
from pathlib import Path
import logging
import time
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class FrameExtractor:
    def __init__(self, resolution=768, num_frames=16):
        self.resolution = resolution
        self.num_frames = num_frames
        logging.info(f"Extractor: {resolution}x{resolution}, {num_frames} frames")
    
    def extract_video(self, video_path):
        """Extract frames from single video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            cap.release()
            return None
        
        indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (self.resolution, self.resolution))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == self.num_frames:
            return np.array(frames, dtype=np.uint8)
        return None
    
    def process_dataset(self, input_dir='sample_dataset', output_dir='frames_768'):
        """Process entire dataset"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_videos = []
        for class_dir in input_path.iterdir():
            if class_dir.is_dir():
                videos = list(class_dir.glob('*.mp4'))
                all_videos.extend([(v, class_dir.name) for v in videos])
                logging.info(f"Found {len(videos)} videos in {class_dir.name}")
        
        logging.info(f"Total videos: {len(all_videos)}")
        
        processed = 0
        failed = 0
        start_time = time.time()
        
        for i, (video_path, class_name) in enumerate(all_videos):
            try:
                frames = self.extract_video(video_path)
                if frames is not None:
                    output_file = output_path / class_name / f"{video_path.stem}.npz"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output_file, frames=frames)
                    processed += 1
                else:
                    failed += 1
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (len(all_videos) - i - 1) / rate / 60
                    logging.info(f"Progress: {i+1}/{len(all_videos)} | Rate: {rate:.1f} v/s | ETA: {eta:.1f}m")
            
            except Exception as e:
                logging.error(f"Error: {e}")
                failed += 1
        
        metadata = {
            'resolution': self.resolution,
            'num_frames': self.num_frames,
            'total_videos': len(all_videos),
            'processed': processed,
            'failed': failed,
            'time_seconds': time.time() - start_time
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Complete! Processed: {processed}, Failed: {failed}")
        return processed, failed

if __name__ == "__main__":
    extractor = FrameExtractor(resolution=768, num_frames=16)
    extractor.process_dataset('sample_dataset', 'frames_768')