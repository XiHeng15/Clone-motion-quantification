import os
import csv
import json
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import subprocess
from scipy.signal import find_peaks, savgol_filter, resample
from scipy.ndimage import gaussian_filter1d
from skimage.filters.thresholding import _validate_image_histogram as validate_image_histogram


class RehabDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Rehabilitation Progress Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize data storage
        self.user_data_file = "rehab_data.json"
        self.user_data = self.load_user_data()
        self.current_session_data = None
        self.current_user = None
        
        # Flag to track UI initialization
        self.ui_initialized = False
        
        # Initialize UI component references
        self.user_name_label = None
        self.user_age_label = None
        self.user_condition_label = None
        self.user_start_date_label = None
        self.stats_tree = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.charts_tab = None
        self.summary_tab = None
        self.user_dropdown = None
        self.session_dropdown = None
        self.progress_indicators = {}
        
        # Set up the main UI structure
        self.setup_ui()
    
    def load_user_data(self):
        """Load user data from JSON file or create a new one if it doesn't exist"""
        if os.path.exists(self.user_data_file):
            try:
                with open(self.user_data_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"users": {}}
        else:
            return {"users": {}}
    
    def save_user_data(self):
        """Save user data to JSON file"""
        with open(self.user_data_file, "w") as f:
            json.dump(self.user_data, f, indent=4)
    
    def threshold_otsu(self, image=None, nbins=256, *, hist=None):
        """Return threshold value based on Otsu's method.
        Either image or hist must be provided.
        """
        counts, bin_centers = validate_image_histogram(image, hist, nbins)

        # class probabilities for all possible thresholds
        weight1 = np.cumsum(counts)
        weight2 = np.cumsum(counts[::-1])[::-1]
        # class means for all possible thresholds
        mean1 = np.cumsum(counts * bin_centers) / weight1
        mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # Clip ends to align class 1 and class 2 variables:
        # The last value of ``weight1``/``mean1`` should pair with zero values in
        # ``weight2``/``mean2``, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bin_centers[idx]
        bin_width = bin_centers[1] - bin_centers[0]
        return threshold, bin_width
    
    def calculate_change_rate(self, data, fps=30, gussian_rate=6):
        """Calculate rate of change in the data."""
        data = np.array(data)
        if gussian_rate != 0:
            data = gaussian_filter1d(data, round(gussian_rate*fps/30))
        data = np.diff(data)
        data = data*fps
        return data
    
    def segment_by_peaks_valleys(self, data, fps=30, loops=None, prominences_noise=1):
        """
        Segments the data based on detected peaks and valleys.
        
        Args:
            data: The input kinematic data array
            fps: Frames per second (default: 30)
            loops: Number of loops to detect. If None, uses Otsu thresholding to determine significance
            prominences_noise: Noise level for prominences (default: 1)
            
        Returns:
            A list of [start_index, end_index] for each segment
        """
        if loops is None:
            peaks, properties = find_peaks(np.where(data < 0, 0, data), prominence=0)
            p_prominences = properties['prominences']
            threshold, bin_width = self.threshold_otsu(np.append(p_prominences, prominences_noise))
            p_selection = p_prominences > threshold + 0.5*bin_width
            valleys, properties = find_peaks(-np.where(data > 0, 0, data), prominence=0)
            v_prominences = properties['prominences']
            threshold, bin_width = self.threshold_otsu(np.append(v_prominences, prominences_noise))
            v_selection = v_prominences > threshold + 0.5*bin_width
        else:
            peaks, properties = find_peaks(data, prominence=5)
            p_prominences = properties['prominences']
            p_selection = np.argsort(p_prominences)[-loops:]

            valleys, properties = find_peaks(-data, prominence=5)
            v_prominences = properties['prominences']
            v_selection = np.argsort(v_prominences)[-loops:]

        peaks = peaks[p_selection]
        p_argsort = np.argsort(peaks)
        peaks = peaks[p_argsort]
        p_prominences = p_prominences[p_selection]
        p_prominences = p_prominences[p_argsort]

        valleys = valleys[v_selection]
        v_argsort = np.argsort(valleys)
        valleys = valleys[v_argsort]
        v_prominences = v_prominences[v_selection]
        v_prominences = v_prominences[v_argsort]

        # Check for valid peaks and valleys
        if len(peaks) == 0 or len(valleys) == 0:
            return []
        if len(peaks) < 3 or len(valleys) < 3:
            return []

        # Get top 3 peaks and valleys if we have more
        if len(p_prominences) > 3:
            t_prominences = p_prominences + v_prominences
            t_p_argsort = np.argsort(t_prominences)[-3:]
            peaks = peaks[t_p_argsort]
            valleys = valleys[t_p_argsort]
            peaks = np.sort(peaks)
            valleys = np.sort(valleys)

        result = []
        if peaks[0] < valleys[0]:
            # First peak is before first valley
            flattened_peaks_valleys = np.ravel([peaks, valleys], 'F')
            interval_fm = peaks[1:] - valleys[:-1]
            period_fm = valleys - peaks

            # Calculate padding factors
            pre_fm_x = peaks[0] / period_fm[0]
            post_fm_x = (len(data) - valleys[-1]) / period_fm[-1]
            min_interval_fm_x = np.min([interval_fm[i] / (period_fm[i] + period_fm[i+1]) for i in range(len(interval_fm))])
            min_fm_x = np.min([pre_fm_x, post_fm_x, min_interval_fm_x, 0.75])

            # Create segments
            for peak, valley, p_fm in zip(peaks, valleys, period_fm):
                start = int(peak - min_fm_x * p_fm)
                end = int(valley + min_fm_x * p_fm)
                result.append([start, end])
        else:
            # First peak is after first valley
            flattened_valleys_peaks = np.ravel([valleys, peaks], 'F')
            interval_fm = valleys[1:] - peaks[:-1]
            period_fm = peaks - valleys

            # Calculate padding factors
            pre_fm_x = valleys[0] / period_fm[0]
            post_fm_x = (len(data) - peaks[-1]) / period_fm[-1]
            min_interval_fm_x = np.min([interval_fm[i] / (period_fm[i] + period_fm[i+1]) for i in range(len(interval_fm))])
            min_fm_x = np.min([pre_fm_x, post_fm_x, min_interval_fm_x, 0.75])

            # Create segments
            for peak, valley, p_fm in zip(peaks, valleys, period_fm):
                start = int(valley - min_fm_x * p_fm)
                end = int(peak + min_fm_x * p_fm)
                result.append([start, end])

        return result
    
    def detect_action_phases(self, values, num_phases=3):
        """
        Detects action phases in the motion data using improved peak/valley detection.
        
        Args:
            values: The array of kinematic values (preferably knee angle)
            num_phases: Number of phases to detect (default: 3)
            
        Returns:
            A list of tuples containing (start_index, end_index, phase_name) for each phase
        """
        if not values or len(values) < num_phases*10:  # Need enough data points
            return []
        
        # Calculate rate of change
        change_rate = self.calculate_change_rate(values)
        
        # Segment the data using the improved algorithm
        segments = self.segment_by_peaks_valleys(change_rate, loops=num_phases)
        
        # If segmentation failed or didn't produce enough segments, fall back to simple splitting
        if not segments or len(segments) < num_phases:
            # Simple fallback: divide the data into equal segments
            segment_length = len(values) // num_phases
            segments = []
            for i in range(num_phases):
                start = i * segment_length
                end = (i + 1) * segment_length if i < num_phases - 1 else len(values)
                segments.append([start, end])
        
        # Limit to the requested number of phases
        segments = segments[:num_phases]
        
        # Create phases with default names
        phases = []
        phase_names = ["Preparation", "Action", "Recovery"]
        
        for i, (start, end) in enumerate(segments):
            phase_name = phase_names[i % len(phase_names)]
            phases.append((int(start), int(end), phase_name))
        
        return phases
    
    def setup_ui(self):
        """Set up the main UI components"""
        # Create a frame for the sidebar
        self.sidebar = ttk.Frame(self.root, width=200, padding=10)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create a main content area
        self.main_content = ttk.Frame(self.root, padding=10)
        self.main_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Add sidebar components
        ttk.Label(self.sidebar, text="User Management", font=("Arial", 12, "bold")).pack(pady=10)
        
        # User selection dropdown
        ttk.Label(self.sidebar, text="Select User:").pack(anchor=tk.W, pady=(10, 0))
        self.user_var = tk.StringVar()
        self.user_dropdown = ttk.Combobox(self.sidebar, textvariable=self.user_var, state="readonly")
        self.user_dropdown.pack(fill=tk.X, pady=(5, 10))
        # We'll update the user dropdown at the end after UI is initialized
        self.user_dropdown.bind("<<ComboboxSelected>>", self.on_user_selected)
        
        # Add new user button
        ttk.Button(self.sidebar, text="Add New User", command=self.add_new_user).pack(fill=tk.X, pady=5)
        
        # Data management section
        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(self.sidebar, text="Data Management", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Upload data button
        ttk.Button(self.sidebar, text="Upload Session Data", command=self.upload_data).pack(fill=tk.X, pady=5)
        
        # Generate report button
        ttk.Button(self.sidebar, text="Generate PDF Report", command=self.generate_report).pack(fill=tk.X, pady=5)
        
        # View data options
        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(self.sidebar, text="View Options", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Metric dropdown
        ttk.Label(self.sidebar, text="Select Metric:").pack(anchor=tk.W, pady=(10, 0))
        self.metric_var = tk.StringVar()
        self.metrics = [
            "left_knee_angle", "right_knee_angle", 
            "left_hip_angle", "right_hip_angle",
            "left_shoulder_angle", "right_shoulder_angle",
            "left_elbow_angle", "right_elbow_angle",
            "trunk_flexion", "trunk_lateral_flexion",
            "knee_angle_symmetry", "hip_angle_symmetry", "shoulder_angle_symmetry"
        ]
        self.metric_dropdown = ttk.Combobox(self.sidebar, textvariable=self.metric_var, values=self.metrics, state="readonly")
        self.metric_dropdown.pack(fill=tk.X, pady=(5, 10))
        self.metric_dropdown.bind("<<ComboboxSelected>>", self.update_charts)
        
        # View type (daily max/min/avg, trend over time)
        ttk.Label(self.sidebar, text="View Type:").pack(anchor=tk.W, pady=(10, 0))
        self.view_type_var = tk.StringVar(value="trend")
        ttk.Radiobutton(self.sidebar, text="Trend Over Time", variable=self.view_type_var, value="trend", command=self.update_charts).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(self.sidebar, text="Session Details", variable=self.view_type_var, value="session", command=self.update_charts).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(self.sidebar, text="Action Phases", variable=self.view_type_var, value="action_phases", command=self.update_charts).pack(anchor=tk.W, pady=2)
        
        # Session dropdown (for viewing specific session details)
        ttk.Label(self.sidebar, text="Select Session:").pack(anchor=tk.W, pady=(10, 0))
        self.session_var = tk.StringVar()
        self.session_dropdown = ttk.Combobox(self.sidebar, textvariable=self.session_var, state="readonly")
        self.session_dropdown.pack(fill=tk.X, pady=(5, 10))
        self.session_dropdown.bind("<<ComboboxSelected>>", self.update_charts)
        
        # Number of repetitions input
        ttk.Label(self.sidebar, text="Repetitions in Session:").pack(anchor=tk.W, pady=(10, 0))
        self.reps_var = tk.StringVar(value="3")
        reps_entry = ttk.Entry(self.sidebar, textvariable=self.reps_var, width=5)
        reps_entry.pack(anchor=tk.W, pady=(5, 10))
        reps_entry.bind("<Return>", self.update_charts)
        
        # Number of action phases input
        ttk.Label(self.sidebar, text="Action Phases:").pack(anchor=tk.W, pady=(10, 0))
        self.phases_var = tk.StringVar(value="3")
        phases_entry = ttk.Entry(self.sidebar, textvariable=self.phases_var, width=5)
        phases_entry.pack(anchor=tk.W, pady=(5, 10))
        phases_entry.bind("<Return>", self.update_charts)
        
        # Set up the main content area with tabs
        self.tabs = ttk.Notebook(self.main_content)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        
        # Charts tab
        self.charts_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.charts_tab, text="Charts")
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.summary_tab, text="Summary")
        
        # Setup the charts area
        self.setup_charts_area()
        
        # Setup the summary area
        self.setup_summary_area()
        
        # Mark UI as initialized
        self.ui_initialized = True
        
        # Show welcome message
        self.show_welcome_message()
        
        # Update user dropdown
        self.update_user_dropdown()
    
    def setup_charts_area(self):
        """Set up the charts area with the main chart and progress indicators"""
        # Main chart frame
        self.chart_frame = ttk.Frame(self.charts_tab)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a figure for the main chart
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Progress indicators frame
        self.progress_frame = ttk.Frame(self.charts_tab)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        # Create progress indicator widgets
        self.progress_indicators = {}
        metrics = ["Knee ROM", "Hip ROM", "Shoulder ROM", "Symmetry"]
        for i, metric in enumerate(metrics):
            frame = ttk.LabelFrame(self.progress_frame, text=metric)
            frame.grid(row=0, column=i, padx=10, pady=5, sticky="ew")
            self.progress_frame.columnconfigure(i, weight=1)
            
            # Current value
            value_label = ttk.Label(frame, text="--", font=("Arial", 16))
            value_label.pack(pady=5)
            
            # Change indicator
            change_label = ttk.Label(frame, text="No change", font=("Arial", 10))
            change_label.pack(pady=5)
            
            self.progress_indicators[metric] = {
                "value": value_label,
                "change": change_label
            }
    
    def setup_summary_area(self):
        """Set up the summary area with user info and stats"""
        # User info frame
        self.user_info_frame = ttk.LabelFrame(self.summary_tab, text="User Information")
        self.user_info_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # User info grid
        self.user_name_label = ttk.Label(self.user_info_frame, text="Name: --")
        self.user_name_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        self.user_age_label = ttk.Label(self.user_info_frame, text="Age: --")
        self.user_age_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        self.user_condition_label = ttk.Label(self.user_info_frame, text="Condition: --")
        self.user_condition_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        
        self.user_start_date_label = ttk.Label(self.user_info_frame, text="Started: --")
        self.user_start_date_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Stats frame
        self.stats_frame = ttk.LabelFrame(self.summary_tab, text="Rehabilitation Progress")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Stats tree view
        columns = ('date', 'action', 'left_knee', 'right_knee', 'left_hip', 'right_hip', 'symmetry')
        self.stats_tree = ttk.Treeview(self.stats_frame, columns=columns, show='headings')
        
        # Define headings
        self.stats_tree.heading('date', text='Session Date')
        self.stats_tree.heading('action', text='Action Type')
        self.stats_tree.heading('left_knee', text='Left Knee (°)')
        self.stats_tree.heading('right_knee', text='Right Knee (°)')
        self.stats_tree.heading('left_hip', text='Left Hip (°)')
        self.stats_tree.heading('right_hip', text='Right Hip (°)')
        self.stats_tree.heading('symmetry', text='Symmetry')
        
        # Define columns
        self.stats_tree.column('date', width=100)
        self.stats_tree.column('action', width=100)
        self.stats_tree.column('left_knee', width=80)
        self.stats_tree.column('right_knee', width=80)
        self.stats_tree.column('left_hip', width=80)
        self.stats_tree.column('right_hip', width=80)
        self.stats_tree.column('symmetry', width=80)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.stats_frame, orient=tk.VERTICAL, command=self.stats_tree.yview)
        self.stats_tree.configure(yscroll=scrollbar.set)
        
        # Pack the tree and scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def update_user_dropdown(self):
        """Update the user dropdown with available users"""
        # Check if user_dropdown exists and UI is initialized
        if not hasattr(self, 'user_dropdown') or not self.user_dropdown:
            return
            
        users = list(self.user_data["users"].keys())
        self.user_dropdown["values"] = users
        if users:
            self.user_dropdown.current(0)
            # Only trigger selection if UI is fully initialized
            if hasattr(self, 'ui_initialized') and self.ui_initialized:
                self.on_user_selected(None)
    
    def on_user_selected(self, event):
        """Handle user selection from dropdown"""
        if not hasattr(self, 'user_var'):
            return
            
        self.current_user = self.user_var.get()
        if self.current_user:
            self.update_user_info()
            self.update_session_dropdown()
            self.update_charts()
    
    def update_user_info(self):
        """Update user information display"""
        if self.current_user and self.current_user in self.user_data["users"]:
            user_info = self.user_data["users"][self.current_user]
            
            # Ensure UI components exist before updating them
            if hasattr(self, 'user_name_label') and self.user_name_label:
                self.user_name_label.config(text=f"Name: {self.current_user}")
            
            if hasattr(self, 'user_age_label') and self.user_age_label:
                self.user_age_label.config(text=f"Age: {user_info.get('age', '--')}")
            
            if hasattr(self, 'user_condition_label') and self.user_condition_label:
                self.user_condition_label.config(text=f"Condition: {user_info.get('condition', '--')}")
            
            if hasattr(self, 'user_start_date_label') and self.user_start_date_label:
                self.user_start_date_label.config(text=f"Started: {user_info.get('start_date', '--')}")
            
            # Update stats tree
            self.update_stats_tree()
    
    def update_session_dropdown(self):
        """Update the session dropdown with available sessions for the current user"""
        # Check if session_dropdown exists
        if not hasattr(self, 'session_dropdown') or not self.session_dropdown:
            return
            
        if self.current_user and self.current_user in self.user_data["users"]:
            user_info = self.user_data["users"][self.current_user]
            sessions = list(user_info.get("sessions", {}).keys())
            sessions.sort(reverse=True)  # Most recent first
            self.session_dropdown["values"] = sessions
            if sessions:
                self.session_dropdown.current(0)
    
    def update_stats_tree(self):
        """Update the stats tree with session data"""
        # Check if stats_tree exists
        if not hasattr(self, 'stats_tree') or not self.stats_tree:
            return
            
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        if self.current_user and self.current_user in self.user_data["users"]:
            user_info = self.user_data["users"][self.current_user]
            sessions = user_info.get("sessions", {})
            
            # Sort sessions by date (newest first)
            sorted_sessions = sorted(sessions.items(), key=lambda x: x[0], reverse=True)
            
            for date, session_data in sorted_sessions:
                # Calculate average values for the session
                left_knee = session_data.get("left_knee_angle", {}).get("avg", "--")
                right_knee = session_data.get("right_knee_angle", {}).get("avg", "--")
                left_hip = session_data.get("left_hip_angle", {}).get("avg", "--")
                right_hip = session_data.get("right_hip_angle", {}).get("avg", "--")
                symmetry = session_data.get("knee_angle_symmetry", {}).get("avg", "--")
                
                # Format values to 1 decimal place if they're numbers
                left_knee = f"{left_knee:.1f}" if isinstance(left_knee, (int, float)) else left_knee
                right_knee = f"{right_knee:.1f}" if isinstance(right_knee, (int, float)) else right_knee
                left_hip = f"{left_hip:.1f}" if isinstance(left_hip, (int, float)) else left_hip
                right_hip = f"{right_hip:.1f}" if isinstance(right_hip, (int, float)) else right_hip
                symmetry = f"{symmetry:.1f}" if isinstance(symmetry, (int, float)) else symmetry
                
                # Get action type if available
                action_type = "unknown"
                if "metadata" in session_data and "action_type" in session_data["metadata"]:
                    action_type = session_data["metadata"]["action_type"]
                
                self.stats_tree.insert('', tk.END, values=(date, action_type, left_knee, right_knee, left_hip, right_hip, symmetry))
    
    def update_charts(self, event=None):
        """Update charts based on selected metric and view type"""
        # Check if chart components exist
        if not hasattr(self, 'ax') or not self.ax or not hasattr(self, 'canvas') or not self.canvas:
            return
            
        if not self.current_user or self.current_user not in self.user_data["users"]:
            return
        
        metric = self.metric_var.get()
        view_type = self.view_type_var.get()
        
        if not metric:
            if self.metrics:
                self.metric_var.set(self.metrics[0])
                metric = self.metrics[0]
            else:
                return
                
        user_info = self.user_data["users"][self.current_user]
        sessions = user_info.get("sessions", {})
        
        if not sessions:
            # No data to display
            self.ax.clear()
            self.ax.set_title("No session data available")
            self.canvas.draw()
            return
        
        # Clear the current chart
        self.ax.clear()
        
        if view_type == "trend":
            # Trend over time chart - show progress across sessions
            dates = []
            avg_values = []
            max_values = []
            min_values = []
            
            for date, session_data in sorted(sessions.items()):
                if metric in session_data:
                    dates.append(date)
                    avg_values.append(session_data[metric].get("avg", 0))
                    max_values.append(session_data[metric].get("max", 0))
                    min_values.append(session_data[metric].get("min", 0))
            
            if dates:
                # Convert string dates to datetime for better plotting
                x_dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
                
                # Plot the data
                self.ax.plot(x_dates, avg_values, 'b-', label='Average')
                self.ax.fill_between(x_dates, min_values, max_values, color='b', alpha=0.2, label='Range')
                
                # Format the x-axis to show dates nicely
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//5)))
                self.fig.autofmt_xdate()
                
                # Add labels and legend
                self.ax.set_xlabel('Date')
                self.ax.set_ylabel('Angle (degrees)')
                self.ax.set_title(f'{metric.replace("_", " ").title()} Progression')
                self.ax.legend()
                
                # Update progress indicators
                self.update_progress_indicators(metric, avg_values, dates)
            else:
                self.ax.set_title(f"No data available for {metric}")
                
        elif view_type == "session":
            # Session detail chart - show one session's data as averaged repetitions
            session_date = self.session_var.get()
            
            if session_date and session_date in sessions:
                session_data = sessions[session_date]
                
                if metric in session_data and "frames" in session_data[metric] and "values" in session_data[metric]:
                    frames = session_data[metric]["frames"]
                    values = session_data[metric]["values"]
                    
                    # Get number of repetitions from input
                    try:
                        num_reps = int(self.reps_var.get())
                        if num_reps <= 0:
                            raise ValueError("Number of repetitions must be positive")
                    except ValueError:
                        num_reps = 3  # Default to 3 if invalid input
                        self.reps_var.set("1")
                    
                    # Calculate frames per repetition (assuming equal length repetitions)
                    if len(frames) > 0:
                        frames_per_rep = len(frames) // num_reps
                        
                        if frames_per_rep > 0:
                            # Calculate average values across repetitions
                            avg_rep_values = []
                            std_rep_values = []
                            x_points = list(range(frames_per_rep))
                            
                            for i in range(frames_per_rep):
                                rep_values = []
                                for rep in range(num_reps):
                                    idx = rep * frames_per_rep + i
                                    if idx < len(values):
                                        rep_values.append(values[idx])
                                
                                if rep_values:
                                    avg_rep_values.append(np.mean(rep_values))
                                    std_rep_values.append(np.std(rep_values))
                                else:
                                    avg_rep_values.append(0)
                                    std_rep_values.append(0)
                            
                            # Plot the average movement pattern
                            self.ax.plot(x_points, avg_rep_values, 'b-', label='Avg Repetition')
                            self.ax.fill_between(x_points, 
                                               [avg - std for avg, std in zip(avg_rep_values, std_rep_values)],
                                               [avg + std for avg, std in zip(avg_rep_values, std_rep_values)],
                                               color='b', alpha=0.2, label='Standard Deviation')
                            
                            # Add labels
                            self.ax.set_xlabel('Normalized Frame')
                            self.ax.set_ylabel('Angle (degrees)')
                            self.ax.set_title(f'Average {metric.replace("_", " ").title()} Pattern - {session_date}')
                            
                            # Add statistics to the chart
                            avg_value = session_data[metric].get("avg", 0)
                            max_value = session_data[metric].get("max", 0)
                            min_value = session_data[metric].get("min", 0)
                            
                            self.ax.text(0.02, 0.95, f'Avg: {avg_value:.1f}°', transform=self.ax.transAxes)
                            self.ax.text(0.02, 0.90, f'Max: {max_value:.1f}°', transform=self.ax.transAxes)
                            self.ax.text(0.02, 0.85, f'Min: {min_value:.1f}°', transform=self.ax.transAxes)
                            self.ax.legend()
                        else:
                            self.ax.set_title(f"Not enough frames for {num_reps} repetitions")
                    else:
                        self.ax.set_title(f"No frame data available for {metric}")
                else:
                    self.ax.set_title(f"No {metric} data available for {session_date}")
            else:
                self.ax.set_title("No session selected")
                
        elif view_type == "action_phases":
            # Action phases chart - show average of each action phase
            session_date = self.session_var.get()
            
            if session_date and session_date in sessions:
                session_data = sessions[session_date]
                
                if metric in session_data and "frames" in session_data[metric] and "values" in session_data[metric]:
                    frames = session_data[metric]["frames"]
                    values = session_data[metric]["values"]
                    
                    # Check if phase segmentation is enabled for this session
                    has_phases = self.check_session_has_phases(session_data)
                    
                    if not has_phases:
                        # Show message that segmentation is disabled for this session
                        self.ax.set_title("Action phase segmentation is disabled for this session")
                        self.canvas.draw()
                        return
                    
                    # Get number of action phases from input
                    try:
                        num_phases = int(self.phases_var.get())
                        if num_phases <= 0 or num_phases > 10:
                            raise ValueError("Number of phases must be between 1 and 10")
                    except ValueError:
                        num_phases = 3  # Default to 3 if invalid input
                        self.phases_var.set("3")
                    
                    # Determine reference metric for phase detection (knee angle is preferred)
                    ref_metric = None
                    ref_values = None
                    
                    # Try to find right knee angle data first
                    if "right_knee_angle" in session_data and "values" in session_data["right_knee_angle"]:
                        ref_metric = "right_knee_angle"
                        ref_values = session_data["right_knee_angle"]["values"]
                    # Fall back to left knee angle
                    elif "left_knee_angle" in session_data and "values" in session_data["left_knee_angle"]:
                        ref_metric = "left_knee_angle"
                        ref_values = session_data["left_knee_angle"]["values"]
                    # If no knee angle data, use the current metric
                    else:
                        ref_metric = metric
                        ref_values = values
                    
                    # Detect action phases using the reference metric
                    phases = self.detect_action_phases(ref_values, num_phases)
                    
                    if phases:
                        # Prepare for plotting
                        phase_names = []
                        phase_avg_values = []
                        phase_std_values = []
                        phase_min_values = []
                        phase_max_values = []
                        
                        # Calculate statistics for each phase
                        for i, (start_idx, end_idx, phase_name) in enumerate(phases):
                            # Extract the values for this phase
                            phase_values = values[start_idx:end_idx+1]
                            
                            if phase_values:
                                phase_avg = np.mean(phase_values)
                                phase_std = np.std(phase_values)
                                phase_min = np.min(phase_values)
                                phase_max = np.max(phase_values)
                                
                                phase_names.append(phase_name)
                                phase_avg_values.append(phase_avg)
                                phase_std_values.append(phase_std)
                                phase_min_values.append(phase_min)
                                phase_max_values.append(phase_max)
                        
                        # Plot the phases as a bar chart
                        x_pos = np.arange(len(phase_names))
                        
                        # Plot average values with error bars
                        self.ax.bar(x_pos, phase_avg_values, 
                                  yerr=phase_std_values,
                                  capsize=10, 
                                  color='skyblue',
                                  label='Average Value')
                                  
                        # Plot the range (min to max)
                        for i in range(len(phase_names)):
                            self.ax.plot([i, i], [phase_min_values[i], phase_max_values[i]], 
                                       'r-', linewidth=2, label='Range' if i == 0 else '')
                        
                        # Add labels and decoration
                        self.ax.set_xticks(x_pos)
                        self.ax.set_xticklabels(phase_names)
                        self.ax.set_ylabel('Angle (degrees)')
                        
                        # Get action type if available
                        action_type = "unknown"
                        if "metadata" in session_data and "action_type" in session_data["metadata"]:
                            action_type = session_data["metadata"]["action_type"]
                            
                        self.ax.set_title(f'{metric.replace("_", " ").title()} by Action Phase - {action_type.title()} - {session_date}')
                        
                        # Add a legend and show the grid
                        handles, labels = self.ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        self.ax.legend(by_label.values(), by_label.keys(), loc='best')
                        self.ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        # Add value labels on top of each bar
                        for i, v in enumerate(phase_avg_values):
                            self.ax.text(i, v + phase_std_values[i] + 2, f'{v:.1f}°', 
                                      ha='center', va='bottom', fontweight='bold')
                            
                        # Add ROM (range of motion) values
                        for i in range(len(phase_names)):
                            rom = phase_max_values[i] - phase_min_values[i]
                            self.ax.text(i, phase_min_values[i] - 5, f'ROM: {rom:.1f}°', 
                                       ha='center', va='top', fontsize=9)
                        
                        # Adjust y-axis to make room for labels
                        self.ax.set_ylim(
                            min(phase_min_values) - 20, 
                            max([v + s + 10 for v, s in zip(phase_avg_values, phase_std_values)])
                        )
                    else:
                        self.ax.set_title(f"Unable to detect action phases for {session_date}")
                else:
                    self.ax.set_title(f"No {metric} data available for {session_date}")
            else:
                self.ax.set_title("No session selected")
        
        # Draw the updated chart
        self.canvas.draw()
    
    def check_session_has_phases(self, session_data):
        """Check if a session has phase data and segmentation is enabled"""
        if not session_data or "metadata" not in session_data:
            return True  # Default to true for backward compatibility
            
        # Check if segmentation is explicitly disabled in metadata
        enable_segmentation = session_data["metadata"].get("enable_segmentation", True)
        
        # For sessions created before this feature, check if there are actually phases in the data
        if enable_segmentation:
            # Check any metric for phase data
            for metric in self.metrics:
                if metric in session_data and "phases" in session_data[metric]:
                    return True
                    
            # If we got here and didn't find any phases, check if it's explicitly enabled
            return "enable_segmentation" in session_data["metadata"]
        
        return enable_segmentation
    
    def update_progress_indicators(self, metric, values, dates):
        """Update the progress indicators based on the selected metric"""
        # Check if progress indicators exist
        if not hasattr(self, 'progress_indicators') or not self.progress_indicators:
            return
            
        if not values or len(values) < 2:
            return
            
        # Calculate the change from first to last session
        first_value = values[0]
        last_value = values[-1]
        change = last_value - first_value
        change_percent = (change / first_value * 100) if first_value != 0 else 0
        
        # Determine which indicator to update based on the metric
        indicator_key = None
        if "knee" in metric:
            indicator_key = "Knee ROM"
        elif "hip" in metric:
            indicator_key = "Hip ROM"
        elif "shoulder" in metric or "elbow" in metric:
            indicator_key = "Shoulder ROM"
        elif "symmetry" in metric:
            indicator_key = "Symmetry"
            
        if indicator_key and indicator_key in self.progress_indicators:
            # Update the value display
            if "value" in self.progress_indicators[indicator_key]:
                self.progress_indicators[indicator_key]["value"].config(
                    text=f"{last_value:.1f}°"
                )
            
            # Update the change indicator
            if "change" in self.progress_indicators[indicator_key]:
                change_text = f"{'+' if change > 0 else ''}{change:.1f}° ({'+' if change_percent > 0 else ''}{change_percent:.1f}%)"
                
                if "symmetry" in metric:
                    # For symmetry, lower is better
                    if change < 0:
                        self.progress_indicators[indicator_key]["change"].config(
                            text=f"{change_text} ↑",
                            foreground="green"
                        )
                    elif change > 0:
                        self.progress_indicators[indicator_key]["change"].config(
                            text=f"{change_text} ↓",
                            foreground="red"
                        )
                    else:
                        self.progress_indicators[indicator_key]["change"].config(
                            text="No change",
                            foreground="black"
                        )
                else:
                    # For angles, higher is usually better
                    if change > 0:
                        self.progress_indicators[indicator_key]["change"].config(
                            text=f"{change_text} ↑",
                            foreground="green"
                        )
                    elif change < 0:
                        self.progress_indicators[indicator_key]["change"].config(
                            text=f"{change_text} ↓",
                            foreground="red"
                        )
                    else:
                        self.progress_indicators[indicator_key]["change"].config(
                            text="No change",
                            foreground="black"
                        )
    
    def add_new_user(self):
        """Open a dialog to add a new user"""
        # Create a new top-level window
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New User")
        dialog.geometry("300x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # User info entry fields
        ttk.Label(dialog, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=5)
        name_entry.focus_set()
        
        ttk.Label(dialog, text="Age:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        age_entry = ttk.Entry(dialog, width=30)
        age_entry.grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Condition:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        condition_entry = ttk.Entry(dialog, width=30)
        condition_entry.grid(row=2, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Goal:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        goal_entry = ttk.Entry(dialog, width=30)
        goal_entry.grid(row=3, column=1, padx=10, pady=5)
        
        # Function to handle save button click
        def save_user():
            name = name_entry.get().strip()
            age = age_entry.get().strip()
            condition = condition_entry.get().strip()
            goal = goal_entry.get().strip()
            
            if not name:
                messagebox.showerror("Error", "Name is required", parent=dialog)
                return
                
            if name in self.user_data["users"]:
                messagebox.showerror("Error", f"User '{name}' already exists", parent=dialog)
                return
            
            # Create new user data
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            self.user_data["users"][name] = {
                "age": age,
                "condition": condition,
                "goal": goal,
                "start_date": today,
                "sessions": {}
            }
            
            # Save to file
            self.save_user_data()
            
            # Update dropdown
            self.update_user_dropdown()
            
            # Select the new user
            self.user_var.set(name)
            self.on_user_selected(None)
            
            # Close the dialog
            dialog.destroy()
        
        # Add save button
        ttk.Button(dialog, text="Save", command=save_user).grid(row=4, column=0, columnspan=2, pady=20)
    
    def upload_data(self):
        """Open a file dialog to upload a CSV session data file"""
        if not self.current_user:
            messagebox.showerror("Error", "Please select a user first")
            return
            
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Kinematic Data CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                if "person_id" in df.columns:
                    person_ids = pd.to_numeric(df["person_id"], errors="coerce")
                    df = df[person_ids == 1].copy()
                    if df.empty:
                        messagebox.showerror(
                            "Error",
                            "No rows for person ID 1 were found in this CSV.",
                        )
                        return
                
                # Process the data
                self.process_uploaded_data(df)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def calculate_phase_statistics(self, values, phases, phase_names):
        """
        Calculate statistics for each phase in the given values.
        
        Args:
            values: Array of values for a specific metric
            phases: List of (start_idx, end_idx, phase_name) tuples
            phase_names: List of custom phase names
            
        Returns:
            List of phase information dictionaries
        """
        phase_info = []
        
        for i, (start_idx, end_idx, _) in enumerate(phases):
            # Get the name for this phase
            phase_name = phase_names[i] if i < len(phase_names) else f"Phase {i+1}"
            
            # Extract values for this phase
            phase_values = values[start_idx:end_idx+1]
            
            if phase_values:
                phase_info.append({
                    "name": phase_name,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "avg": float(np.mean(phase_values)),
                    "max": float(np.max(phase_values)),
                    "min": float(np.min(phase_values)),
                    "std": float(np.std(phase_values)),
                    "rom": float(np.max(phase_values) - np.min(phase_values))
                })
        
        return phase_info
    
    def process_uploaded_data(self, df):
        """Process the uploaded CSV data and add it to the user's sessions"""
        # Ask for session date or use today
        dialog = tk.Toplevel(self.root)
        dialog.title("Session Information")
        dialog.geometry("500x450")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Session Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        # Default to today
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        date_entry = ttk.Entry(dialog, width=20)
        date_entry.insert(0, today)
        date_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Action type selection
        ttk.Label(dialog, text="Action Type:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        action_type_var = tk.StringVar(value="squat")
        action_type_combo = ttk.Combobox(dialog, textvariable=action_type_var, width=15, state="readonly")
        action_type_combo["values"] = ["squat", "sit-to-stand"]
        action_type_combo.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(dialog, text="Number of Repetitions:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        reps_entry = ttk.Entry(dialog, width=10)
        reps_entry.insert(0, "3")  # Default to 3 repetitions
        reps_entry.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(dialog, text="Action Phases:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        phases_entry = ttk.Entry(dialog, width=10)
        phases_entry.insert(0, "3")  # Default to 3 phases
        phases_entry.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
        
        # Advanced settings frame with toggle
        adv_frame = ttk.LabelFrame(dialog, text="Advanced Settings")
        adv_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Use advanced segmentation checkbox
        use_adv_segmentation_var = tk.BooleanVar(value=True)
        use_adv_segmentation_cb = ttk.Checkbutton(
            adv_frame, 
            text="Use Advanced Segmentation",
            variable=use_adv_segmentation_var
        )
        use_adv_segmentation_cb.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        
        # Help text for advanced segmentation
        ttk.Label(
            adv_frame, 
            text="Advanced segmentation uses rate-of-change analysis\nto detect action phases more accurately.",
            font=("Arial", 8),
            foreground="gray"
        ).grid(row=1, column=0, padx=10, pady=0, sticky=tk.W)
        
        # Enable action segmentation checkbox
        enable_segmentation_var = tk.BooleanVar(value=True)
        enable_segmentation_cb = ttk.Checkbutton(
            dialog, 
            text="Enable Action Phase Segmentation",
            variable=enable_segmentation_var
        )
        enable_segmentation_cb.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)
        
        # Phase names
        ttk.Label(dialog, text="Phase Names:").grid(row=6, column=0, sticky=tk.NW, padx=10, pady=5)
        
        # Frame for phase name entries
        phase_names_frame = ttk.Frame(dialog)
        phase_names_frame.grid(row=6, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Default phase names
        default_phase_names = ["Preparation", "Action", "Recovery"]
        phase_entries = []
        
        # Add some default phase names
        for i, name in enumerate(default_phase_names):
            phase_entry = ttk.Entry(phase_names_frame, width=15)
            phase_entry.insert(0, name)
            phase_entry.pack(pady=2)
            phase_entries.append(phase_entry)
        
        # Function to handle save button click
        def save_session():
            try:
                # Get session date and number of reps
                session_date = date_entry.get().strip()
                
                try:
                    datetime.datetime.strptime(session_date, "%Y-%m-%d")
                except ValueError:
                    messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD.", parent=dialog)
                    return
                
                # Get action type
                action_type = action_type_var.get()
                
                # Get segmentation settings
                use_advanced_segmentation = use_adv_segmentation_var.get()
                enable_segmentation = enable_segmentation_var.get()
                
                try:
                    num_reps = int(reps_entry.get().strip())
                    if num_reps <= 0:
                        raise ValueError("Number of repetitions must be positive")
                except ValueError:
                    messagebox.showerror("Error", "Invalid number of repetitions.", parent=dialog)
                    return
                
                try:
                    num_phases = int(phases_entry.get().strip())
                    if num_phases <= 0 or num_phases > 10:
                        raise ValueError("Number of phases must be between 1 and 10")
                except ValueError:
                    messagebox.showerror("Error", "Invalid number of phases.", parent=dialog)
                    return
                
                # Get phase names
                phase_names = []
                for entry in phase_entries:
                    name = entry.get().strip()
                    if name:
                        phase_names.append(name)
                    else:
                        phase_names.append("Phase")
                
                # Ensure we have enough phase names
                while len(phase_names) < num_phases:
                    phase_names.append(f"Phase {len(phase_names)+1}")
                
                # Process the data from the CSV
                session_data = {
                    "metadata": {
                        "action_type": action_type,
                        "repetitions": num_reps,
                        "use_advanced_segmentation": use_advanced_segmentation,
                        "enable_segmentation": enable_segmentation
                    }
                }
                
                # Group data by metric
                metrics = {}
                for col in df.columns:
                    if col.lower() in [m.lower() for m in self.metrics]:
                        # Find the exact metric name with matching case
                        metric = next(m for m in self.metrics if m.lower() == col.lower())
                        metrics[metric] = df[col].tolist()
                
                # Create a frames list (assuming 1 frame per row)
                frames = list(range(len(df)))
                
                # Initialize phases to None
                phases = None
                
                # Get segmentation settings from metadata
                enable_segmentation = session_data["metadata"]["enable_segmentation"]
                use_advanced_segmentation = session_data["metadata"]["use_advanced_segmentation"]
                
                # Only perform segmentation if it's enabled
                if enable_segmentation:
                    if use_advanced_segmentation:
                        # Use the improved segmentation approach with knee angles
                        # First, check if we have both knee angles for better detection
                        if "right_knee_angle" in metrics and "left_knee_angle" in metrics:
                            # Use the sum of both knee angles for better detection
                            both_knee_avg = np.array(metrics["left_knee_angle"]) + np.array(metrics["right_knee_angle"])
                            # Calculate rate of change
                            change_rate = self.calculate_change_rate(both_knee_avg)
                            # Detect segments
                            segments = self.segment_by_peaks_valleys(change_rate, loops=num_phases)
                            
                            if segments and len(segments) >= num_phases:
                                # Convert segments to phases format
                                phases = []
                                
                                for i, (start, end) in enumerate(segments[:num_phases]):
                                    phase_name = phase_names[i % len(phase_names)]
                                    phases.append((int(start), int(end), phase_name))
                        
                        # Fall back to single knee angle if advanced segmentation on both knees failed
                        if not phases:
                            ref_metric = None
                            ref_values = None
                            
                            # Try to find right knee angle data first
                            if "right_knee_angle" in metrics:
                                ref_metric = "right_knee_angle"
                                ref_values = metrics[ref_metric]
                            # Fall back to left knee angle
                            elif "left_knee_angle" in metrics:
                                ref_metric = "left_knee_angle"
                                ref_values = metrics[ref_metric]
                            # Fall back to any other metric
                            elif metrics:
                                ref_metric = list(metrics.keys())[0]
                                ref_values = metrics[ref_metric]
                            
                            # Detect phases using the reference metric with advanced method
                            if ref_values:
                                phases = self.detect_action_phases(ref_values, num_phases)
                    
                    # If advanced segmentation is disabled or failed, use simple equal divisions
                    if not phases:
                        # Simple approach: divide data into equal segments
                        segment_length = len(df) // num_phases
                        phases = []
                        
                        for i in range(num_phases):
                            start = i * segment_length
                            end = (i + 1) * segment_length - 1 if i < num_phases - 1 else len(df) - 1
                            phase_name = phase_names[i % len(phase_names)]
                            phases.append((int(start), int(end), phase_name))
                
                # If segmentation is disabled, set phases to None
                if not enable_segmentation:
                    phases = None
                
                # Process each metric
                for metric, values in metrics.items():
                    # Calculate statistics
                    avg = np.mean(values)
                    maximum = np.max(values)
                    minimum = np.min(values)
                    
                    # Store the metric data
                    session_data[metric] = {
                        "frames": frames,
                        "values": values,
                        "avg": float(avg),
                        "max": float(maximum),
                        "min": float(minimum)
                    }
                    
                    # Apply the detected phases to all metrics if enabled
                    if phases and enable_segmentation:
                        phase_info = self.calculate_phase_statistics(values, phases, phase_names)
                        session_data[metric]["phases"] = phase_info
                        
                        # Also store phase information in a more accessible format
                        session_data[metric]["phase_stats"] = {
                            "names": [p["name"] for p in phase_info],
                            "avg": [p["avg"] for p in phase_info],
                            "max": [p["max"] for p in phase_info],
                            "min": [p["min"] for p in phase_info],
                            "rom": [p["rom"] for p in phase_info]
                        }
                
                # Add the session data to the user's data
                if session_date in self.user_data["users"][self.current_user]["sessions"]:
                    # Ask for confirmation to overwrite
                    if not messagebox.askyesno("Confirm", 
                                             f"Session data for {session_date} already exists. Overwrite?",
                                             parent=dialog):
                        return
                
                self.user_data["users"][self.current_user]["sessions"][session_date] = session_data
                
                # Save to file
                self.save_user_data()
                
                # Update the session dropdown
                self.update_session_dropdown()
                
                # Select the new session
                self.session_var.set(session_date)
                
                # Update the charts
                self.update_charts()
                
                # Close the dialog
                dialog.destroy()
                
                # Show success message
                messagebox.showinfo("Success", f"Session data for {session_date} saved successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save session data: {str(e)}", parent=dialog)
        
        # Add save button
        ttk.Button(dialog, text="Save Session", command=save_session).grid(row=4, column=0, columnspan=2, pady=20)
        
    def generate_report(self):
        """Generate a PDF report of the user's progress"""
        if not self.current_user:
            messagebox.showerror("Error", "Please select a user first")
            return
            
        try:
            # Call the generate_report.py script
            subprocess.run(["python", "generate_report.py", 
                          "--user", self.current_user,
                          "--data", self.user_data_file])
            
            # Show success message
            messagebox.showinfo("Success", f"Report generated successfully for {self.current_user}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")

    def show_welcome_message(self):
        """Show a welcome message when the application starts"""
        if not hasattr(self, 'ax') or not self.ax:
            return
            
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Welcome to the Rehabilitation Progress Dashboard\n\nPlease select a user or add a new one to get started.",
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12, transform=self.ax.transAxes)
                   
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.draw()


# Main function to run the application
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rehabilitation Progress Dashboard")
    args = parser.parse_args()
    
    # Create the root window
    root = tk.Tk()
    
    # Create the dashboard application
    app = RehabDashboard(root)
    
    # Run the application
    root.mainloop()
