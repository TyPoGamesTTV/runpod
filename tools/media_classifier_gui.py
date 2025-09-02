#!/usr/bin/env python3
"""
Media Classifier GUI
Simple interface for classifying and sorting videos into folders
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
from pathlib import Path
import shutil
import time
import sys
import os

# Add tools to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.single_video import VideoClassifier

class MediaClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Media Classifier - 768x768 Model")
        self.root.geometry("900x600")
        
        # Threading
        self.processing = False
        self.stop_requested = False
        self.log_queue = queue.Queue()
        
        # Model and paths
        self.model_path = tk.StringVar()
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.confidence_threshold = tk.DoubleVar(value=50.0)
        
        # Statistics
        self.total_videos = 0
        self.processed_videos = 0
        self.classifications = {'1_Safe': 0, '2_Unsafe': 0, '3_Explicit': 0, 'Uncertain': 0}
        
        self.create_widgets()
        self.update_log()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model selection
        ttk.Label(left_frame, text="Model File (.pth):", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        model_frame = ttk.Frame(left_frame)
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Entry(model_frame, textvariable=self.model_path, width=40).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=1)
        
        # Input folder
        ttk.Label(left_frame, text="Input Folder:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        input_frame = ttk.Frame(left_frame)
        input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Entry(input_frame, textvariable=self.input_folder, width=40).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(input_frame, text="Browse", command=self.browse_input).grid(row=0, column=1)
        
        # Output folder
        ttk.Label(left_frame, text="Output Base Directory:", font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        output_frame = ttk.Frame(left_frame)
        output_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Entry(output_frame, textvariable=self.output_folder, width=40).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=0, column=1)
        
        # Confidence threshold
        ttk.Label(left_frame, text="Confidence Threshold (%):", font=('Arial', 10, 'bold')).grid(row=6, column=0, sticky=tk.W, pady=(0, 5))
        threshold_frame = ttk.Frame(left_frame)
        threshold_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                          variable=self.confidence_threshold, length=300)
        self.threshold_slider.grid(row=0, column=0, padx=(0, 10))
        self.threshold_label = ttk.Label(threshold_frame, text="50%")
        self.threshold_label.grid(row=0, column=1)
        
        self.threshold_slider.configure(command=self.update_threshold_label)
        
        # Options frame
        options_frame = ttk.LabelFrame(left_frame, text="Options", padding="10")
        options_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.copy_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Copy files (don't move)", variable=self.copy_mode).grid(row=0, column=0, sticky=tk.W)
        
        self.create_uncertain = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Create 'Uncertain' folder for low confidence", 
                       variable=self.create_uncertain).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=9, column=0, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Classification", command=self.start_processing, 
                                       style='Accent.TButton')
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_frame, mode='determinate')
        self.progress.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(left_frame, text="Ready", foreground="green")
        self.status_label.grid(row=11, column=0, sticky=tk.W)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(left_frame, text="Statistics", padding="10")
        stats_frame.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.stats_label = ttk.Label(stats_frame, text="No videos processed yet")
        self.stats_label.grid(row=0, column=0)
        
        # Right panel - Log
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(right_frame, text="Processing Log:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(right_frame, width=50, height=35, wrap=tk.WORD)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure colors for log
        self.log_text.tag_config('INFO', foreground='black')
        self.log_text.tag_config('SUCCESS', foreground='green')
        self.log_text.tag_config('WARNING', foreground='orange')
        self.log_text.tag_config('ERROR', foreground='red')
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
    def update_threshold_label(self, value):
        self.threshold_label.config(text=f"{float(value):.0f}%")
    
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
            self.log("INFO", f"Model selected: {Path(filename).name}")
    
    def browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder.set(folder)
            # Count videos
            video_count = self.count_videos(folder)
            self.log("INFO", f"Input folder selected: {video_count} videos found")
    
    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Base Directory")
        if folder:
            self.output_folder.set(folder)
            self.log("INFO", f"Output folder selected: {folder}")
    
    def count_videos(self, folder):
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        count = 0
        for ext in extensions:
            count += len(list(Path(folder).glob(f'**/*{ext}')))
            count += len(list(Path(folder).glob(f'**/*{ext.upper()}')))
        return count
    
    def log(self, level, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put((level, f"[{timestamp}] {message}"))
    
    def update_log(self):
        # Process log queue
        while not self.log_queue.empty():
            level, message = self.log_queue.get()
            self.log_text.insert(tk.END, message + "\n", level)
            self.log_text.see(tk.END)
        
        # Schedule next update
        self.root.after(100, self.update_log)
    
    def update_statistics(self):
        total = sum(self.classifications.values())
        if total > 0:
            stats_text = f"Processed: {total}/{self.total_videos}\n"
            stats_text += f"Safe: {self.classifications['1_Safe']}\n"
            stats_text += f"Unsafe: {self.classifications['2_Unsafe']}\n"
            stats_text += f"Explicit: {self.classifications['3_Explicit']}\n"
            if self.classifications['Uncertain'] > 0:
                stats_text += f"Uncertain: {self.classifications['Uncertain']}"
            self.stats_label.config(text=stats_text)
    
    def validate_inputs(self):
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file")
            return False
        
        if not Path(self.model_path.get()).exists():
            messagebox.showerror("Error", "Model file does not exist")
            return False
        
        if not self.input_folder.get():
            messagebox.showerror("Error", "Please select an input folder")
            return False
        
        if not Path(self.input_folder.get()).exists():
            messagebox.showerror("Error", "Input folder does not exist")
            return False
        
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder")
            return False
        
        return True
    
    def start_processing(self):
        if not self.validate_inputs():
            return
        
        # Reset statistics
        self.classifications = {'1_Safe': 0, '2_Unsafe': 0, '3_Explicit': 0, 'Uncertain': 0}
        self.processed_videos = 0
        
        # Update UI
        self.processing = True
        self.stop_requested = False
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Processing...", foreground="blue")
        
        # Start processing thread
        thread = threading.Thread(target=self.process_videos)
        thread.daemon = True
        thread.start()
    
    def stop_processing(self):
        self.stop_requested = True
        self.log("WARNING", "Stop requested - finishing current video...")
        self.status_label.config(text="Stopping...", foreground="orange")
    
    def process_videos(self):
        try:
            # Load model
            self.log("INFO", "Loading model...")
            classifier = VideoClassifier(self.model_path.get(), device=None)
            self.log("SUCCESS", "Model loaded successfully")
            
            # Get video files
            input_path = Path(self.input_folder.get())
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            video_files = []
            for ext in extensions:
                video_files.extend(input_path.glob(f'**/*{ext}'))
                video_files.extend(input_path.glob(f'**/*{ext.upper()}'))
            
            self.total_videos = len(video_files)
            self.log("INFO", f"Found {self.total_videos} videos to process")
            
            if self.total_videos == 0:
                self.log("WARNING", "No videos found in input folder")
                return
            
            # Create output folders
            output_base = Path(self.output_folder.get())
            output_folders = {
                '1_Safe': output_base / '1_Safe',
                '2_Unsafe': output_base / '2_Unsafe',
                '3_Explicit': output_base / '3_Explicit'
            }
            
            if self.create_uncertain.get():
                output_folders['Uncertain'] = output_base / '0_Uncertain'
            
            for folder in output_folders.values():
                folder.mkdir(parents=True, exist_ok=True)
            
            self.log("INFO", f"Created output folders in {output_base}")
            
            # Process each video
            threshold = self.confidence_threshold.get() / 100.0
            
            for i, video_path in enumerate(video_files):
                if self.stop_requested:
                    self.log("WARNING", "Processing stopped by user")
                    break
                
                # Update progress
                self.progress['value'] = (i / self.total_videos) * 100
                
                try:
                    # Classify video
                    self.log("INFO", f"Processing: {video_path.name}")
                    pred_class, confidence = classifier.classify(video_path)
                    
                    # Determine destination
                    if confidence < threshold and self.create_uncertain.get():
                        dest_folder = output_folders['Uncertain']
                        self.classifications['Uncertain'] += 1
                        self.log("WARNING", f"  → Uncertain ({confidence:.1%} < {threshold:.0%})")
                    else:
                        dest_folder = output_folders[pred_class]
                        self.classifications[pred_class] += 1
                        self.log("SUCCESS", f"  → {pred_class} ({confidence:.1%})")
                    
                    # Move or copy file
                    dest_path = dest_folder / video_path.name
                    if dest_path.exists():
                        # Add timestamp to avoid overwriting
                        timestamp = int(time.time())
                        dest_path = dest_folder / f"{video_path.stem}_{timestamp}{video_path.suffix}"
                    
                    if self.copy_mode.get():
                        shutil.copy2(video_path, dest_path)
                    else:
                        shutil.move(str(video_path), str(dest_path))
                    
                    self.processed_videos += 1
                    self.update_statistics()
                    
                except Exception as e:
                    self.log("ERROR", f"  Error processing {video_path.name}: {str(e)}")
                    continue
            
            # Final statistics
            self.progress['value'] = 100
            self.log("SUCCESS", f"Processing complete! Processed {self.processed_videos}/{self.total_videos} videos")
            self.log("INFO", f"Results saved to: {output_base}")
            
        except Exception as e:
            self.log("ERROR", f"Fatal error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:
            # Reset UI
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Ready", foreground="green")

def main():
    root = tk.Tk()
    app = MediaClassifierGUI(root)
    
    # Set icon if available
    try:
        if sys.platform.startswith('win'):
            root.iconbitmap(default='icon.ico')
    except:
        pass
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()