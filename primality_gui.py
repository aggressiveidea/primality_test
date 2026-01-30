import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import tkinter.font as tkfont
from PIL import Image, ImageTk
import json
import time
import math
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Import the existing primality testing system
try:
    from main import (
        TestType, Result, TestResult, NumberAnalysis,
        PrimalityTestManager, analyze_number_api, test_single_api
    )
except ImportError:
    # Fallback if running standalone
    import sys
    sys.path.append('.')
    from main import (
        TestType, Result, TestResult, NumberAnalysis,
        PrimalityTestManager, analyze_number_api, test_single_api
    )


class ModernTheme:

    
    PRIMARY = "#0F1419"        # Almost black background - very dark
    SECONDARY = "#1A1F28"      # Slightly lighter secondary
    TERTIARY = "#252D3A"       # Medium dark tertiary
    
    # Accent colors - Vibrant and distinctive
    ACCENT_PRIMARY = "#00A8E8"   # Bright cyan - primary action
    ACCENT_SECONDARY = "#7C3AED" # Vibrant purple
    
    # Status colors - Clear and readable
    SUCCESS = "#26C485"        # Fresh green
    WARNING = "#FFB84D"        # Bright amber
    ERROR = "#FF5757"          # Bright red
    INFO = "#3B82F6"           # Blue
    
    # Text colors - MAXIMUM CONTRAST
    TEXT = "#FFFFFF"           # Pure white - main text
    TEXT_SECONDARY = "#E8ECFF" # Near-white with slight blue tint
    TEXT_TERTIARY = "#A0A9BE"  # Light gray for secondary labels
    
    # Values and results - VERY HIGH CONTRAST
    VALUE_TEXT = "#FFFFFF"     # Pure white for values
    LABEL_TEXT = "#8B92A8"     # Muted gray-blue for labels
    RESULT_TEXT = "#FFFFFF"    # Pure white for results
    
    BACKGROUND = "#0F1419"     # Very dark background
    CARD_BG = "#1A1F28"        # Card background - slightly lighter
    SECTION_BG = "#252D3A"     # Section divider background
    BORDER = "#3A4452"         # Border color
    
    # Fonts - Improved readability
    TITLE_FONT = ("Segoe UI", 26, "bold")
    HEADING_FONT = ("Segoe UI", 14, "bold")
    BODY_FONT = ("Segoe UI", 11)
    LABEL_FONT = ("Segoe UI", 10)
    VALUE_FONT = ("Segoe UI", 11, "bold")
    MONOSPACE_FONT = ("Consolas", 10, "bold")
    
    # Padding - Better spacing
    PAD_SMALL = 6
    PAD_MEDIUM = 12
    PAD_LARGE = 20


def calculate_complexity_metrics(test_type_value: str, number: int, iterations: int = 10) -> dict:
    """Calculate algorithmic complexity metrics for different primality tests"""
    bit_length = number.bit_length()
    
    complexity_info = {
        "Miller-Rabin": {
            "time_complexity": f"O(k·log³n)",
            "space_complexity": "O(log n)",
            "actual_iterations": iterations,
            "estimated_ops": f"{iterations * (bit_length ** 3):,}",
            "error_probability": f"4^(-k) = 4^(-{iterations})"
        },
        "Fermat Test": {
            "time_complexity": "O(log³n)",
            "space_complexity": "O(log n)",
            "actual_iterations": iterations,
            "estimated_ops": f"{bit_length ** 3:,}",
            "error_probability": "Variable"
        },
        "Fermat": {
            "time_complexity": "O(log³n)",
            "space_complexity": "O(log n)",
            "actual_iterations": iterations,
            "estimated_ops": f"{bit_length ** 3:,}",
            "error_probability": "Variable - Unreliable"
        },
        "Trial Division": {
            "time_complexity": "O(√n)",
            "space_complexity": "O(1)",
            "actual_iterations": 1,
            "estimated_ops": f"{int(math.sqrt(number)):,}",
            "error_probability": "None"
        },
        "Lucas-Lehmer": {
            "time_complexity": "O(n·log²n)",
            "space_complexity": "O(log n)",
            "actual_iterations": 1,
            "estimated_ops": f"{number * (bit_length ** 2):,}",
            "error_probability": "None"
        },
        "Format": {
            "time_complexity": "O(log²n)",
            "space_complexity": "O(log n)",
            "actual_iterations": 1,
            "estimated_ops": f"{bit_length ** 2:,}",
            "error_probability": "Deterministic"
        },
        "Baillie-PSW": {
            "time_complexity": "O(log³n)",
            "space_complexity": "O(log n)",
            "actual_iterations": 1,
            "estimated_ops": f"{bit_length ** 3:,}",
            "error_probability": "Negligible (~0%)"
        },
        "Solovay-Strassen": {
            "time_complexity": "O(k·log³n)",
            "space_complexity": "O(log n)",
            "actual_iterations": iterations,
            "estimated_ops": f"{iterations * (bit_length ** 3):,}",
            "error_probability": f"2^(-k) = 2^(-{iterations})"
        },
        "Trial Division": {
            "time_complexity": "O(√n)",
            "space_complexity": "O(1)",
            "actual_iterations": 1,
            "estimated_ops": f"{int(math.sqrt(number)):,}",
            "error_probability": "None"
        },
        "AKS (Deterministic)": {
            "time_complexity": "O(log⁶n)",
            "space_complexity": "O(log n)",
            "actual_iterations": 1,
            "estimated_ops": f"{bit_length ** 6:,}",
            "error_probability": "Deterministic"
        }
    }
    
    return complexity_info.get(test_type_value, {
        "time_complexity": "O(unknown)",
        "space_complexity": "O(unknown)",
        "actual_iterations": iterations,
        "estimated_ops": "Unknown",
        "error_probability": "Unknown"
    })


class PerformanceHistory:
    """Tracks performance metrics across multiple test runs for visualization"""
    
    def __init__(self):
        self.history: Dict[str, List[dict]] = {}
        self.chart_figure: Optional[Figure] = None
        self.chart_canvas: Optional[FigureCanvasTkAgg] = None
        self.chart_ax: Optional[plt.Axes] = None
    
    def add_result(self, test_type: str, number: int, execution_time: float, certainty: float):
        """Add a test result to history"""
        if test_type not in self.history:
            self.history[test_type] = []
        
        self.history[test_type].append({
            'number': number,
            'bit_length': number.bit_length(),
            'time': execution_time,
            'certainty': certainty
        })
    
    def get_plot_data(self, test_type: str) -> tuple:
        """Get sorted data for plotting"""
        if test_type not in self.history or not self.history[test_type]:
            return [], []
        
        data = sorted(self.history[test_type], key=lambda x: x['bit_length'])
        bit_lengths = [d['bit_length'] for d in data]
        times = [d['time'] for d in data]
        return bit_lengths, times
    
    def clear(self):
        """Clear all history"""
        self.history.clear()
        self.clear_charts()

    def setup_charts(self, parent_widget):
        """Set up the Matplotlib chart area within the Tkinter GUI"""
        self.chart_figure = Figure(figsize=(6, 4), dpi=100)
        self.chart_ax = self.chart_figure.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.chart_figure, master=parent_widget)
        self.chart_canvas_widget = self.chart_canvas.get_tk_widget()
        self.chart_canvas_widget.pack(fill="both", expand=True)

    def clear_charts(self):
        """Clear the chart display"""
        if self.chart_ax:
            self.chart_ax.clear()
            self.chart_ax.text(0.5, 0.5, "No data to display", horizontalalignment='center', verticalalignment='center', transform=self.chart_ax.transAxes)
            self.chart_ax.set_title("Performance Over Bit Length")
            self.chart_ax.set_xlabel("Number Bit Length")
            self.chart_ax.set_ylabel("Execution Time (s)")
            if self.chart_canvas:
                self.chart_canvas.draw()
    

class RoundedButton(tk.Button):
    """Custom button with hover effects"""
    
    def __init__(self, master=None, **kwargs):
        self.hover_bg = kwargs.pop("hover_bg", ModernTheme.ACCENT_PRIMARY)
        super().__init__(master, **kwargs)
        self.default_bg = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    
    def on_enter(self, e):
        self["background"] = self.hover_bg
        self["relief"] = "raised"
    
    def on_leave(self, e):
        self["background"] = self.default_bg
        self["relief"] = "flat"


class TestCard(ttk.Frame):
    """Card widget for displaying test results with MAXIMUM visibility"""
    
    def __init__(self, parent, test_result: TestResult, number: int = None, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.test_result = test_result
        self.number = number
        self.setup_ui()
    
    def setup_ui(self):
        """Setup card UI with clear visual hierarchy"""
        self.columnconfigure(0, weight=1)
        
        # Main result container with padding
        result_frame = tk.Frame(self, bg=ModernTheme.CARD_BG, highlightthickness=0)
        result_frame.pack(fill="both", expand=True, padx=0, pady=0)
        result_frame.columnconfigure(0, weight=1)
        
        # Test name - Label text
        name_label = tk.Label(
            result_frame,
            text=f"✓ {self.test_result.test_type.value}",
            font=("Segoe UI", 11, "bold"),
            bg=ModernTheme.CARD_BG,
            fg=ModernTheme.TEXT_SECONDARY
        )
        name_label.grid(row=0, column=0, sticky="w", padx=ModernTheme.PAD_MEDIUM, pady=(ModernTheme.PAD_MEDIUM, 8))
        
        result_color = {
            Result.PRIME: ModernTheme.SUCCESS,
            Result.COMPOSITE: ModernTheme.ERROR,
            Result.PROBABLY_PRIME: ModernTheme.WARNING,
            Result.ERROR: ModernTheme.ERROR
        }.get(self.test_result.result, ModernTheme.TEXT)
        
        result_label = tk.Label(
            result_frame,
            text=self.test_result.result.value.upper(),
            font=("Segoe UI", 18, "bold"),
            bg=ModernTheme.CARD_BG,
            fg=result_color
        )
        result_label.grid(row=1, column=0, sticky="w", padx=ModernTheme.PAD_MEDIUM, pady=(0, 14))
        
        metrics_frame = tk.Frame(result_frame, bg=ModernTheme.CARD_BG)
        metrics_frame.grid(row=2, column=0, sticky="ew", padx=ModernTheme.PAD_MEDIUM, pady=(0, 12))
        metrics_frame.columnconfigure(1, weight=1)
        
        # Left column metrics
        left_col = tk.Frame(metrics_frame, bg=ModernTheme.CARD_BG)
        left_col.pack(side="left", anchor="nw")
        
        metric_pairs = [
            ("Certainty:", f"{self.test_result.certainty:.6f}"),
            ("Time:", f"{self.test_result.execution_time:.4f}s"),
        ]
        
        for i, (label, value) in enumerate(metric_pairs):
            # Label - muted gray
            label_w = tk.Label(
                left_col,
                text=label,
                font=("Segoe UI", 9),
                bg=ModernTheme.CARD_BG,
                fg=ModernTheme.LABEL_TEXT
            )
            label_w.grid(row=i, column=0, sticky="w", pady=3)
            
            # Value - pure white, bold
            value_w = tk.Label(
                left_col,
                text=value,
                font=("Consolas", 11, "bold"),
                bg=ModernTheme.CARD_BG,
                fg=ModernTheme.VALUE_TEXT
            )
            value_w.grid(row=i, column=1, sticky="w", padx=(15, 0), pady=3)
        
        # Right column metrics
        right_col = tk.Frame(metrics_frame, bg=ModernTheme.CARD_BG)
        right_col.pack(side="left", anchor="nw", padx=(40, 0))
        
        right_pairs = []
        if self.test_result.iterations > 1:
            right_pairs.append(("Iterations:", str(self.test_result.iterations)))
        
        for i, (label, value) in enumerate(right_pairs):
            label_w = tk.Label(
                right_col,
                text=label,
                font=("Segoe UI", 9),
                bg=ModernTheme.CARD_BG,
                fg=ModernTheme.LABEL_TEXT
            )
            label_w.grid(row=i, column=0, sticky="w", pady=3)
            
            value_w = tk.Label(
                right_col,
                text=value,
                font=("Consolas", 11, "bold"),
                bg=ModernTheme.CARD_BG,
                fg=ModernTheme.VALUE_TEXT
            )
            value_w.grid(row=i, column=1, sticky="w", padx=(15, 0), pady=3)
        
        if self.number:
            complexity = calculate_complexity_metrics(
                self.test_result.test_type.value,
                self.number,
                self.test_result.iterations
            )
            
            # Separator
            sep = tk.Frame(result_frame, bg=ModernTheme.BORDER, height=1)
            sep.grid(row=3, column=0, sticky="ew", pady=(8, 0), padx=ModernTheme.PAD_MEDIUM)
            
            # Complexity section header with underline effect
            complexity_frame = tk.Frame(result_frame, bg=ModernTheme.CARD_BG)
            complexity_frame.grid(row=4, column=0, sticky="w", padx=ModernTheme.PAD_MEDIUM, pady=(8, 8))
            
            complexity_label = tk.Label(
                complexity_frame,
                text="⚡ Complexity Analysis",
                font=("Segoe UI", 10, "bold"),
                bg=ModernTheme.CARD_BG,
                fg=ModernTheme.ACCENT_PRIMARY
            )
            complexity_label.pack(anchor="w")
            
            # Complexity details grid with WHITE text
            details_frame = tk.Frame(result_frame, bg=ModernTheme.CARD_BG)
            details_frame.grid(row=5, column=0, sticky="ew", padx=ModernTheme.PAD_MEDIUM, pady=(0, 8))
            
            comp_pairs = [
                ("Time Complexity:", complexity.get('time_complexity', 'N/A')),
                ("Space Complexity:", complexity.get('space_complexity', 'N/A')),
                ("Est. Operations:", complexity.get('estimated_ops', 'N/A')),
            ]
            
            for i, (label, value) in enumerate(comp_pairs):
                label_w = tk.Label(
                    details_frame,
                    text=label,
                    font=("Segoe UI", 9),
                    bg=ModernTheme.CARD_BG,
                    fg=ModernTheme.LABEL_TEXT
                )
                label_w.grid(row=i, column=0, sticky="w", pady=2)
                
                value_w = tk.Label(
                    details_frame,
                    text=str(value),
                    font=("Consolas", 10, "bold"),
                    bg=ModernTheme.CARD_BG,
                    fg=ModernTheme.VALUE_TEXT
                )
                value_w.grid(row=i, column=1, sticky="w", padx=(20, 0), pady=2)
        
        if self.test_result.message:
            msg_frame = tk.Frame(result_frame, bg=ModernTheme.SECTION_BG, highlightthickness=0)
            msg_frame.grid(row=6, column=0, sticky="ew", padx=-0, pady=(8, 0))
            
            message_text = tk.Label(
                msg_frame,
                text=self.test_result.message,
                font=("Segoe UI", 10),
                bg=ModernTheme.SECTION_BG,
                fg=ModernTheme.TEXT,
                wraplength=400,
                justify="left"
            )
            message_text.pack(anchor="w", padx=ModernTheme.PAD_MEDIUM, pady=ModernTheme.PAD_MEDIUM)


class TabButton(tk.Button):
    """Custom tab button with dark theme styling"""
    
    def __init__(self, parent, text, callback, is_active=False, **kwargs):
        super().__init__(
            parent,
            text=text,
            font=("Segoe UI", 10, "bold"),
            bg=ModernTheme.ACCENT_SECONDARY if is_active else ModernTheme.SECONDARY,
            fg=ModernTheme.TEXT,
            activebackground=ModernTheme.ACCENT_SECONDARY,
            activeforeground=ModernTheme.TEXT,
            relief="flat",
            border=0,
            padx=20,
            pady=10,
            command=callback,
            **kwargs
        )
        self.is_active = is_active
        self.inactive_bg = ModernTheme.SECONDARY
        self.active_bg = ModernTheme.ACCENT_SECONDARY
        self.bind("<Enter>", self.on_hover)
        self.bind("<Leave>", self.on_leave)
    
    def on_hover(self, e):
        if not self.is_active:
            self.config(bg=ModernTheme.TERTIARY)
    
    def on_leave(self, e):
        if not self.is_active:
            self.config(bg=self.inactive_bg)
    
    def set_active(self):
        self.is_active = True
        self.config(bg=self.active_bg)
    
    def set_inactive(self):
        self.is_active = False
        self.config(bg=self.inactive_bg)


class PrimalityGUI:
    """Main Tkinter GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PrimeCheck- Primality Testing System")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 750)
        
        # Set icon (if available)
        try:
            self.root.iconbitmap("prime_icon.ico")
        except:
            pass
        
        # Initialize test manager
        self.test_manager = PrimalityTestManager()
        self.current_analysis = None
        
        self.performance_history = PerformanceHistory()
        
        # Setup UI
        self.setup_styles()
        self.setup_ui()
        
        # Center window
        self.center_window()
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        
        # Modern dark theme
        self.root.configure(bg=ModernTheme.BACKGROUND)
        
        style.configure("Main.TFrame", background=ModernTheme.BACKGROUND)
        style.configure("Sidebar.TFrame", background=ModernTheme.PRIMARY)
        style.configure("Card.TFrame", background=ModernTheme.CARD_BG, relief="flat", borderwidth=1)
        
        # Label styles
        style.configure("Title.TLabel",
                       font=ModernTheme.TITLE_FONT,
                       background=ModernTheme.BACKGROUND,
                       foreground=ModernTheme.TEXT)
        style.configure("Heading.TLabel",
                       font=ModernTheme.HEADING_FONT,
                       background=ModernTheme.BACKGROUND,
                       foreground=ModernTheme.TEXT)
        
        # Button styles
        style.configure("Primary.TButton",
                       font=("Segoe UI", 11, "bold"),
                       padding=(20, 10),
                       background=ModernTheme.ACCENT_PRIMARY,
                       foreground=ModernTheme.PRIMARY)
        style.configure("Secondary.TButton",
                       font=("Segoe UI", 9),
                       padding=(12, 6),
                       background=ModernTheme.TERTIARY,
                       foreground=ModernTheme.TEXT)
        
        style.configure("Modern.TEntry",
                       fieldbackground=ModernTheme.PRIMARY,  # Changed to darker PRIMARY background for better visibility
                       foreground=ModernTheme.VALUE_TEXT,   # Changed to VALUE_TEXT (pure white) for maximum contrast
                       insertcolor=ModernTheme.ACCENT_PRIMARY,
                       borderwidth=1,
                       relief="solid",
                       padding=10)
        
        style.map("Modern.TEntry",
                 fieldbackground=[("focus", ModernTheme.SECONDARY)],
                 foreground=[("focus", ModernTheme.VALUE_TEXT)])  # Ensure white text on focus too
        
        # Notebook styles - Fixed text visibility on tabs
        style.configure("Modern.TNotebook",
                       background=ModernTheme.BACKGROUND,
                       borderwidth=0)
        style.configure("Modern.TNotebook.Tab",
                       font=("Segoe UI", 10, "bold"),
                       padding=(20, 10),
                       background=ModernTheme.SECONDARY,
                       foreground=ModernTheme.TEXT)
        style.map("Modern.TNotebook.Tab",
                 background=[("selected", ModernTheme.ACCENT_SECONDARY)],
                 foreground=[("selected", ModernTheme.TEXT)])
        
        # Scrollbar styles
        style.configure("Modern.Vertical.TScrollbar",
                       background=ModernTheme.SECONDARY,
                       troughcolor=ModernTheme.PRIMARY,
                       bordercolor=ModernTheme.PRIMARY,
                       arrowcolor=ModernTheme.ACCENT_PRIMARY,
                       relief="flat")
        
    def setup_ui(self):
        """Setup the main UI layout"""
        
        main_container = ttk.Frame(self.root, style="Main.TFrame")
        main_container.pack(fill="both", expand=True)
        
        # Left sidebar
        sidebar = ttk.Frame(main_container, style="Sidebar.TFrame", width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        # Right content area
        content = ttk.Frame(main_container, style="Main.TFrame")
        content.pack(side="right", fill="both", expand=True)
        
        # Setup sidebar widgets
        self.setup_sidebar(sidebar)
        
        # Setup content area
        self.setup_content(content)
        
    def setup_sidebar(self, parent):
        """Setup the sidebar with controls"""
        
        header_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        header_frame.pack(fill="x", pady=ModernTheme.PAD_LARGE)
        
        # Logo/Title with better styling
        logo_label = ttk.Label(header_frame,
                              text="∑",
                              font=("Segoe UI", 40, "bold"),
                              background=ModernTheme.PRIMARY,
                              foreground=ModernTheme.ACCENT_PRIMARY)
        logo_label.pack(pady=(0, 5))
        
        ttk.Label(header_frame,
                 text="PrimeCheck",
                 font=("Segoe UI", 18, "bold"),
                 background=ModernTheme.PRIMARY,
                 foreground=ModernTheme.TEXT,
                 justify="center").pack()
        
        ttk.Label(header_frame,
                 text="Pro",
                 font=("Segoe UI", 11, "bold"),
                 background=ModernTheme.PRIMARY,
                 foreground=ModernTheme.ACCENT_SECONDARY,
                 justify="center").pack(pady=(2, 0))
        
        ttk.Label(header_frame,
                 text="Primality Testing System",
                 font=("Segoe UI", 8),
                 background=ModernTheme.PRIMARY,
                 foreground=ModernTheme.TEXT_TERTIARY,
                 justify="center").pack(pady=(5, 0))
        
        # Separator
        separator = tk.Frame(parent, bg=ModernTheme.BORDER, height=1)
        separator.pack(fill="x", padx=ModernTheme.PAD_MEDIUM, pady=ModernTheme.PAD_LARGE)
        
        input_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        input_frame.pack(fill="x", padx=ModernTheme.PAD_MEDIUM, pady=(0, ModernTheme.PAD_LARGE))
        
        ttk.Label(input_frame,
                 text="Number to Test:",
                 font=("Segoe UI", 10, "bold"),
                 background=ModernTheme.PRIMARY,
                 foreground=ModernTheme.LABEL_TEXT).pack(anchor="w", pady=(0, 5))
        
        self.number_var = tk.StringVar(value="1000003")
        self.number_entry = ttk.Entry(input_frame,
                                     textvariable=self.number_var,
                                     font=("Segoe UI", 11),
                                     style="Modern.TEntry")
        self.number_entry.pack(fill="x", pady=(0, 0))
        self.number_entry.bind("<Return>", lambda e: self.run_analysis())
        
        # Test selection with better styling
        ttk.Label(parent,
                 text="Select Tests:",
                 font=("Segoe UI", 10, "bold"),
                 background=ModernTheme.PRIMARY,
                 foreground=ModernTheme.LABEL_TEXT).pack(anchor="w", padx=ModernTheme.PAD_MEDIUM, pady=(ModernTheme.PAD_LARGE, ModernTheme.PAD_SMALL))
        
        self.test_vars = {}
        test_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        test_frame.pack(fill="x", padx=ModernTheme.PAD_MEDIUM)
        
        for i, test_type in enumerate(TestType):
            var = tk.BooleanVar(value=True)
            self.test_vars[test_type] = var
            cb = ttk.Checkbutton(test_frame,
                               text=test_type.value,
                               variable=var,
                               onvalue=True,
                               offvalue=False)
            cb.pack(anchor="w", pady=3)
        
        # Iterations control
        iter_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        iter_frame.pack(fill="x", padx=ModernTheme.PAD_MEDIUM, pady=(ModernTheme.PAD_LARGE, 0))
        
        ttk.Label(iter_frame,
                 text="Iterations (probabilistic tests):",
                 font=("Segoe UI", 9, "bold"),
                 background=ModernTheme.PRIMARY,
                 foreground=ModernTheme.LABEL_TEXT).pack(anchor="w", pady=(0, 5))
        
        self.iter_var = tk.IntVar(value=10)
        iter_spin = ttk.Spinbox(iter_frame,
                               from_=1,
                               to=100,
                               textvariable=self.iter_var,
                               font=ModernTheme.BODY_FONT,
                               width=10)
        iter_spin.pack(anchor="w")
        
        button_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        button_frame.pack(fill="x", padx=ModernTheme.PAD_MEDIUM, pady=ModernTheme.PAD_LARGE)
        
        self.run_btn = RoundedButton(button_frame,
                                    text="▶ RUN ANALYSIS",
                                    command=self.run_analysis,
                                    bg=ModernTheme.ACCENT_PRIMARY,
                                    fg=ModernTheme.PRIMARY,
                                    font=("Segoe UI", 11, "bold"),
                                    relief="flat",
                                    cursor="hand2",
                                    padx=30,
                                    pady=12,
                                    activebackground=ModernTheme.ACCENT_SECONDARY,
                                    activeforeground=ModernTheme.TEXT,
                                    hover_bg=ModernTheme.ACCENT_SECONDARY)
        self.run_btn.pack(fill="x", pady=(0, 10))
        
        RoundedButton(button_frame,
                     text="📋 Clear Results",
                     command=self.clear_results,
                     bg=ModernTheme.TERTIARY,
                     fg=ModernTheme.TEXT,
                     font=("Segoe UI", 9),
                     relief="flat",
                     cursor="hand2",
                     padx=20,
                     pady=8,
                     activebackground=ModernTheme.SECONDARY,
                     activeforeground=ModernTheme.ACCENT_PRIMARY,
                     hover_bg=ModernTheme.SECONDARY).pack(fill="x", pady=5)
        
        RoundedButton(button_frame,
                     text="💾 Export JSON",
                     command=self.export_results,
                     bg=ModernTheme.TERTIARY,
                     fg=ModernTheme.TEXT,
                     font=("Segoe UI", 9),
                     relief="flat",
                     cursor="hand2",
                     padx=20,
                     pady=8,
                     activebackground=ModernTheme.SECONDARY,
                     activeforeground=ModernTheme.ACCENT_PRIMARY,
                     hover_bg=ModernTheme.SECONDARY).pack(fill="x", pady=5)
        
        RoundedButton(button_frame,
                     text="📊 Batch Test",
                     command=self.batch_test,
                     bg=ModernTheme.TERTIARY,
                     fg=ModernTheme.TEXT,
                     font=("Segoe UI", 9),
                     relief="flat",
                     cursor="hand2",
                     padx=20,
                     pady=8,
                     activebackground=ModernTheme.SECONDARY,
                     activeforeground=ModernTheme.ACCENT_PRIMARY,
                     hover_bg=ModernTheme.SECONDARY).pack(fill="x", pady=5)
        
        # Status bar at bottom
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent,
                 textvariable=self.status_var,
                 font=("Segoe UI", 8),
                 background=ModernTheme.PRIMARY,
                 foreground=ModernTheme.TEXT_TERTIARY).pack(side="bottom", fill="x", padx=ModernTheme.PAD_MEDIUM, pady=(0, 10))
        
    def setup_content(self, parent):
        """Setup the main content area"""
        
        # Create tab button container
        tab_container = tk.Frame(parent, bg=ModernTheme.BACKGROUND)
        tab_container.pack(fill="x", padx=ModernTheme.PAD_MEDIUM, pady=(ModernTheme.PAD_MEDIUM, 0))
        
        # Store tab references
        self.tab_buttons = {}
        self.tab_frames = {}
        self.current_tab = "results"
        
        def switch_tab(tab_name):
            # Hide all tabs
            for tab_frame in self.tab_frames.values():
                tab_frame.pack_forget()
            # Deactivate all buttons
            for btn in self.tab_buttons.values():
                btn.set_inactive()
            # Show selected tab
            self.tab_frames[tab_name].pack(fill="both", expand=True, padx=ModernTheme.PAD_MEDIUM, pady=ModernTheme.PAD_MEDIUM)
            self.tab_buttons[tab_name].set_active()
            self.current_tab = tab_name
            if tab_name == "stats":
                self.update_statistics_charts()
        
        # Create tab buttons
        self.tab_buttons["results"] = TabButton(
            tab_container, "📊 Results",
            lambda: switch_tab("results"),
            is_active=True
        )
        self.tab_buttons["results"].pack(side="left", padx=(0, 5))
        
        self.tab_buttons["details"] = TabButton(
            tab_container, "📝 Details",
            lambda: switch_tab("details")
        )
        self.tab_buttons["details"].pack(side="left", padx=5)
        
        self.tab_buttons["stats"] = TabButton(
            tab_container, "📈 Statistics",
            lambda: switch_tab("stats")
        )
        self.tab_buttons["stats"].pack(side="left", padx=5)
        
        # Create content frames
        content_container = tk.Frame(parent, bg=ModernTheme.BACKGROUND)
        content_container.pack(fill="both", expand=True)
        
        # Results tab frame
        self.results_frame = tk.Frame(content_container, bg=ModernTheme.BACKGROUND)
        self.tab_frames["results"] = self.results_frame
        self.results_frame.pack(fill="both", expand=True)
        
        summary_container = tk.Frame(self.results_frame, bg=ModernTheme.CARD_BG)
        summary_container.pack(fill="x", padx=0, pady=(0, ModernTheme.PAD_MEDIUM))
        
        self.summary_label = tk.Label(summary_container,
                                      text="No analysis performed yet",
                                      font=ModernTheme.TITLE_FONT,
                                      bg=ModernTheme.CARD_BG,
                                      fg=ModernTheme.TEXT)
        self.summary_label.pack(pady=(ModernTheme.PAD_LARGE, ModernTheme.PAD_MEDIUM))
        
        # Details grid
        details_frame = tk.Frame(summary_container, bg=ModernTheme.CARD_BG)
        details_frame.pack(fill="x", padx=ModernTheme.PAD_LARGE, pady=(0, ModernTheme.PAD_LARGE))
        
        # Time and certainty labels
        self.time_label = tk.Label(details_frame,
                                   text="⏱ Time: --",
                                   font=("Segoe UI", 10),
                                   bg=ModernTheme.CARD_BG,
                                   fg=ModernTheme.LABEL_TEXT)
        self.time_label.grid(row=0, column=0, padx=(0, 30), sticky="w")
        
        self.certainty_label = tk.Label(details_frame,
                                        text="🎯 Certainty: --",
                                        font=("Segoe UI", 10),
                                        bg=ModernTheme.CARD_BG,
                                        fg=ModernTheme.LABEL_TEXT)
        self.certainty_label.grid(row=0, column=1, padx=30, sticky="w")
        
        self.verdict_label = tk.Label(details_frame,
                                      text="✓ Verdict: --",
                                      font=("Segoe UI", 10, "bold"),
                                      bg=ModernTheme.CARD_BG,
                                      fg=ModernTheme.SUCCESS)
        self.verdict_label.grid(row=0, column=2, padx=30, sticky="w")
        
        # Test results container with scrollbar
        results_container = tk.Frame(self.results_frame, bg=ModernTheme.BACKGROUND)
        results_container.pack(fill="both", expand=True)
        
        # Create canvas and scrollbar with proper bind sequence
        canvas = tk.Canvas(results_container, bg=ModernTheme.BACKGROUND, highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview, style="Modern.Vertical.TScrollbar")
        self.scrollable_frame = tk.Frame(canvas, bg=ModernTheme.BACKGROUND)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Update canvas width on resize
        def configure_canvas(event):
            try:
                canvas.itemconfig(canvas_window, width=event.width)
            except:
                pass  # Handle edge case during window initialization
        
        canvas.bind("<Configure>", configure_canvas)
        
        # Details tab frame
        details_tab = tk.Frame(content_container, bg=ModernTheme.BACKGROUND)
        self.tab_frames["details"] = details_tab
        
        # JSON view
        json_frame = tk.Frame(details_tab, bg=ModernTheme.BACKGROUND)
        json_frame.pack(fill="both", expand=True, padx=ModernTheme.PAD_LARGE, pady=ModernTheme.PAD_LARGE)
        
        json_label = tk.Label(json_frame,
                             text="JSON Output:",
                             font=ModernTheme.HEADING_FONT,
                             bg=ModernTheme.BACKGROUND,
                             fg=ModernTheme.LABEL_TEXT)
        json_label.pack(anchor="w", pady=(0, ModernTheme.PAD_MEDIUM))
        
        self.json_text = scrolledtext.ScrolledText(json_frame,
                                                  font=ModernTheme.MONOSPACE_FONT,
                                                  bg=ModernTheme.CARD_BG,
                                                  fg=ModernTheme.TEXT,
                                                  insertbackground=ModernTheme.ACCENT_PRIMARY,
                                                  wrap="word",
                                                  height=20)
        self.json_text.pack(fill="both", expand=True)
        
        # Statistics tab frame
        stats_tab = tk.Frame(content_container, bg=ModernTheme.BACKGROUND)
        self.tab_frames["stats"] = stats_tab
        
        stats_label = tk.Label(stats_tab,
                              text="Performance Statistics:",
                              font=ModernTheme.HEADING_FONT,
                              bg=ModernTheme.BACKGROUND,
                              fg=ModernTheme.LABEL_TEXT)
        stats_label.pack(anchor="w", padx=ModernTheme.PAD_LARGE, pady=(ModernTheme.PAD_LARGE, ModernTheme.PAD_MEDIUM))
        
        self.stats_canvas_frame = tk.Frame(stats_tab, bg=ModernTheme.CARD_BG)
        self.stats_canvas_frame.pack(fill="both", expand=True, padx=ModernTheme.PAD_LARGE, pady=(0, ModernTheme.PAD_LARGE))
        
        # Placeholder text when no data
        self.stats_placeholder = tk.Label(
            self.stats_canvas_frame,
            text="Run tests to see performance statistics and complexity curves",
            font=("Segoe UI", 12),
            bg=ModernTheme.CARD_BG,
            fg=ModernTheme.LABEL_TEXT
        )
        self.stats_placeholder.pack(expand=True)
        
        # Initialize chart setup within the canvas frame.
        # This will be called by setup_charts, which in turn is called by update_statistics_charts.
        # We need to make sure the frame is packed and available.
        # The actual chart embedding logic is moved to update_statistics_charts to be called when tab is selected.
        
        # Switch to results tab initially
        switch_tab("results")
    
    def run_analysis(self):
        """Run the selected tests on the input number"""
        try:
            # Removed comma stripping and added explicit error handling for integer conversion
            n_str = self.number_entry.get()
            n = int(n_str.replace(",", "")) # Remove commas for parsing
            
            # Get selected tests
            selected_tests = [test for test, var in self.test_vars.items() if var.get()]
            
            if not selected_tests:
                messagebox.showwarning("No Tests Selected", "Please select at least one test to run.")
                return
            
            # Update UI
            self.run_btn.config(state="disabled", text="⏳ Testing...")
            self.status_var.set(f"Testing {n:,}...")
            self.root.update()
            
            # Run analysis
            start_time = time.time()
            analysis = self.test_manager.analyze_number(n, selected_tests)
            self.current_analysis = analysis
            elapsed = time.time() - start_time
            
            # Display results
            self.display_results(analysis, n) # Pass n to display_results
            
            # Update status
            self.status_var.set(f"Analysis complete in {elapsed:.2f}s")
            
        except ValueError:
            messagebox.showerror("Invalid Input", f"'{self.number_entry.get()}' is not a valid integer.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.run_btn.config(state="normal", text="▶ RUN ANALYSIS")
    
    def display_results(self, analysis: NumberAnalysis, number: int = None):
        """Display the analysis results in the UI"""
        
        verdict_color = {
            "Definitely Prime": ModernTheme.SUCCESS,
            "Definitely Composite": ModernTheme.ERROR,
            "Almost Certainly Prime": ModernTheme.WARNING,
            "Probably Prime": ModernTheme.WARNING,
            "Inconclusive": ModernTheme.TEXT_TERTIARY
        }.get(analysis.overall_verdict, ModernTheme.TEXT)
        
        self.summary_label.config(
            text=f"n = {analysis.number:,}",
            foreground=verdict_color
        )
        
        self.time_label.config(
            text=f"⏱ Total Time: {analysis.total_time:.4f}s",
            foreground=ModernTheme.LABEL_TEXT
        )
        
        # Calculate average certainty
        if analysis.test_results:
            # Added check for None certainty before summing
            valid_certainties = [r.certainty for r in analysis.test_results if r.certainty is not None]
            if valid_certainties:
                avg_certainty = sum(valid_certainties) / len(valid_certainties)
                self.certainty_label.config(
                    text=f"🎯 Avg Certainty: {avg_certainty:.6f}",
                    foreground=ModernTheme.LABEL_TEXT
                )
            else:
                self.certainty_label.config(text="🎯 Avg Certainty: N/A", foreground=ModernTheme.LABEL_TEXT)
        else:
            self.certainty_label.config(text="🎯 Avg Certainty: N/A", foreground=ModernTheme.LABEL_TEXT)
        
        self.verdict_label.config(
            text=f"✓ Verdict: {analysis.overall_verdict}",
            foreground=verdict_color
        )
        
        for result in analysis.test_results:
            self.performance_history.add_result(
                result.test_type.value,
                analysis.number,
                result.execution_time,
                result.certainty if result.certainty is not None else 1.0
            )
        
        # Clear previous test cards
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        for i, test_result in enumerate(analysis.test_results):
            card = TestCard(self.scrollable_frame, test_result, number=number or analysis.number)
            card.pack(fill="x", padx=ModernTheme.PAD_SMALL, pady=(0, ModernTheme.PAD_MEDIUM))
        
        # Update JSON view
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(analysis.to_dict(), indent=2))
        
        # Update statistics view (basic summary) - called when results are displayed
        self.update_stats_display()

        # Switch to results tab
        self.tab_buttons["results"].set_active()
        self.tab_frames["results"].pack(fill="both", expand=True)
    
    def update_statistics_charts(self):
        """Generate and display performance charts"""
        if not self.performance_history.history:
            # Placeholder should be visible if there's no data
            if not self.stats_placeholder.winfo_ismapped():
                self.stats_placeholder.pack(expand=True)
            return
        
        # Hide placeholder if it's visible
        if self.stats_placeholder.winfo_ismapped():
            self.stats_placeholder.pack_forget()
        
        # Clear previous canvas widgets
        for widget in self.stats_canvas_frame.winfo_children():
            widget.destroy()
        
        # Create figure with subplots
        fig = Figure(figsize=(6.5, 5.5), dpi=100, facecolor=ModernTheme.CARD_BG) # Adjusted size for better fit
        fig.patch.set_facecolor(ModernTheme.CARD_BG)
        
        # Color palette
        colors = ['#00A8E8', '#7C3AED', '#26C485', '#FFB84D', '#FF5757', '#3B82F6', '#EC4899']
        
        # Plot 1: Execution Time vs Bit Length for each algorithm
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor(ModernTheme.TERTIARY)
        ax1.set_title('Time vs Bit Length', color=ModernTheme.TEXT, fontsize=10, fontweight='bold', pad=8)
        ax1.set_xlabel('Bit Length', color=ModernTheme.LABEL_TEXT, fontsize=8)
        ax1.set_ylabel('Time (s)', color=ModernTheme.LABEL_TEXT, fontsize=8)
        ax1.tick_params(colors=ModernTheme.LABEL_TEXT, labelsize=7)
        ax1.grid(True, alpha=0.2, color=ModernTheme.BORDER)
        ax1.spines['bottom'].set_color(ModernTheme.BORDER)
        ax1.spines['left'].set_color(ModernTheme.BORDER)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Ensure consistent ordering of test types for consistent colors
        sorted_test_types = sorted(self.performance_history.history.keys())
        
        for idx, test_type in enumerate(sorted_test_types):
            color = colors[idx % len(colors)]
            bit_lengths, times = self.performance_history.get_plot_data(test_type)
            if bit_lengths and times:
                ax1.scatter(bit_lengths, times, label=test_type, color=color, s=30, alpha=0.7, edgecolors='white', linewidth=0.3)
                # Add trend line if we have enough points
                if len(bit_lengths) > 1:
                    try:
                        z = np.polyfit(bit_lengths, times, 1)
                        p = np.poly1d(z)
                        bit_range = np.linspace(min(bit_lengths), max(bit_lengths), 100)
                        ax1.plot(bit_range, p(bit_range), color=color, alpha=0.3, linewidth=1)
                    except np.linalg.LinAlgError:
                        # Handle cases where polyfit might fail (e.g., all points are the same)
                        pass
        
        ax1.legend(fontsize=7, loc='best', facecolor=ModernTheme.SECONDARY, edgecolor=ModernTheme.BORDER, labelcolor=ModernTheme.TEXT)
        
        # Plot 2: Algorithm Comparison (Average Time)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor(ModernTheme.TERTIARY)
        ax2.set_title('Average Performance', color=ModernTheme.TEXT, fontsize=10, fontweight='bold', pad=8)
        ax2.set_ylabel('Avg Time (s)', color=ModernTheme.LABEL_TEXT, fontsize=8)
        ax2.tick_params(colors=ModernTheme.LABEL_TEXT, labelsize=7)
        ax2.spines['bottom'].set_color(ModernTheme.BORDER)
        ax2.spines['left'].set_color(ModernTheme.BORDER)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        test_names = []
        avg_times = []
        for test_type in sorted_test_types:
            data = self.performance_history.history.get(test_type, [])
            if data:
                times = [d['time'] for d in data]
                test_names.append(test_type)
                avg_times.append(np.mean(times))
        
        bars = ax2.bar(range(len(test_names)), avg_times, color=[colors[i % len(colors)] for i in range(len(test_names))], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.set_xticks(range(len(test_names)))
        ax2.set_xticklabels(test_names, rotation=45, ha='right', fontsize=7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s',
                    ha='center', va='bottom', fontsize=6, color=ModernTheme.TEXT)
        
        # Plot 3: Complexity Curves (Theoretical)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor(ModernTheme.TERTIARY)
        ax3.set_title('Theoretical Complexity', color=ModernTheme.TEXT, fontsize=10, fontweight='bold', pad=8)
        ax3.set_xlabel('Bit Length', color=ModernTheme.LABEL_TEXT, fontsize=8)
        ax3.set_ylabel('Operations (log scale)', color=ModernTheme.LABEL_TEXT, fontsize=8)
        ax3.tick_params(colors=ModernTheme.LABEL_TEXT, labelsize=7)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.2, color=ModernTheme.BORDER, which='both')
        ax3.spines['bottom'].set_color(ModernTheme.BORDER)
        ax3.spines['left'].set_color(ModernTheme.BORDER)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Get bit length range from actual data
        all_bit_lengths = []
        for data_list in self.performance_history.history.values():
            all_bit_lengths.extend([d['bit_length'] for d in data_list])
        
        if all_bit_lengths:
            min_bits = max(8, min(all_bit_lengths)) # Ensure a reasonable minimum bit length
            max_bits = max(all_bit_lengths)
            bit_range = np.linspace(min_bits, max_bits + 5, 100) # Extend range slightly
            
            # Plot theoretical complexities
            # Adjusted complexity functions to be more representative
            complexity_funcs = {
                'O(log²n)': lambda b: b**2.5, # A bit more than log^2
                'O(√n)': lambda b: (2**b)**0.5, # sqrt(n) where n = 2^b
                'O(log³n)': lambda b: b**3,
                'O(k·log³n)': lambda b: 10 * b**3, # k=10 for illustration
            }
            
            # Use colors that stand out
            colors_theory = ['#FFB84D', '#FF5757', '#00A8E8', '#7C3AED'] 
            
            # Ensure we don't try to plot if bit_range is empty or invalid
            if len(bit_range) > 1:
                for (name, func), color in zip(complexity_funcs.items(), colors_theory):
                    try:
                        # Calculate operations for the theoretical complexity functions
                        # Note: The direct `b` here represents bit length, not the number `n`.
                        # For complexity like O(sqrt(n)), n = 2^b, so sqrt(n) = sqrt(2^b) = 2^(b/2).
                        # For polynomial in log(n), log(n) is proportional to b.
                        
                        if name == 'O(√n)':
                            ops = [2**(b/2) for b in bit_range]
                        elif name == 'O(log²n)':
                             ops = [b**2.5 for b in bit_range] # Adjust exponent slightly for visibility
                        elif name == 'O(log³n)':
                            ops = [b**3 for b in bit_range]
                        elif name == 'O(k·log³n)':
                            ops = [10 * b**3 for b in bit_range]
                        else: # Default for unknown or other forms
                            ops = [b for b in bit_range] # Placeholder

                        # Check for extremely large values that might cause plotting issues
                        ops = np.clip(ops, 1e-10, 1e10) # Clip values to avoid infinite scales
                        
                        ax3.plot(bit_range, ops, label=name, color=color, linewidth=1.5, alpha=0.8)
                    except Exception as e:
                        print(f"Error plotting complexity {name}: {e}") # Debugging
                        pass
            
            ax3.legend(fontsize=7, loc='best', facecolor=ModernTheme.SECONDARY, edgecolor=ModernTheme.BORDER, labelcolor=ModernTheme.TEXT)
        
        # Plot 4: Test Summary Statistics
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor(ModernTheme.TERTIARY)
        ax4.axis('off')
        
        # Create summary text
        summary_text = "Summary\n" + "="*25 + "\n"
        total_tests = sum(len(data) for data in self.performance_history.history.values())
        summary_text += f"Total Tests: {total_tests}\n"
        summary_text += f"Algos Tested: {len(self.performance_history.history)}\n\n"
        
        # Add per-algorithm stats
        for test_type in sorted_test_types:
            data = self.performance_history.history.get(test_type, [])
            if data:
                times = [d['time'] for d in data]
                summary_text += f"{test_type}:\n"
                summary_text += f"  Tests: {len(data)}\n"
                if times:
                    summary_text += f"  Avg Time: {np.mean(times):.4f}s\n"
                    summary_text += f"  Min Time: {min(times):.4f}s\n"
                    summary_text += f"  Max Time: {max(times):.4f}s\n\n"
                else:
                    summary_text += f"  No timing data.\n\n"
        
        # Use text widget within the subplot for better text wrapping and control
        # The bbox will make it look like it's on a card.
        props = dict(boxstyle='round', facecolor=ModernTheme.SECONDARY, 
                     alpha=0.8, edgecolor=ModernTheme.BORDER)
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=7, verticalalignment='top', fontfamily='monospace',
                color=ModernTheme.TEXT, bbox=props)
        
        fig.tight_layout(pad=1.5) # Adjust padding between subplots
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.stats_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.performance_history.chart_canvas = canvas # Store canvas reference for potential clearing later

    def update_stats_display(self):
        """Update the statistics tab with performance data (textual summary)"""
        # This function is now primarily to trigger the chart update and ensure
        # the placeholder is hidden if there is data. The actual chart generation
        # happens in update_statistics_charts.
        
        if self.performance_history.history:
            if self.stats_placeholder.winfo_ismapped():
                self.stats_placeholder.pack_forget()
        else:
            if not self.stats_placeholder.winfo_ismapped():
                self.stats_placeholder.pack(expand=True) # Show placeholder if no data
        
        self.update_statistics_charts() # Ensure charts are updated
            
    def clear_results(self):
        """Clear all results and history"""
        self.summary_label.config(
            text="No analysis performed yet",
            foreground=ModernTheme.TEXT_TERTIARY
        )
        self.time_label.config(text="⏱ Time: --", foreground=ModernTheme.LABEL_TEXT)
        self.certainty_label.config(text="🎯 Certainty: --", foreground=ModernTheme.LABEL_TEXT)
        self.verdict_label.config(text="✓ Verdict: --", foreground=ModernTheme.SUCCESS)
        
        # Clear test cards
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Clear JSON
        self.json_text.delete("1.0", tk.END)
        
        # Clear statistics
        self.performance_history.clear() # Clear performance history
        
        # Update the chart display to reflect empty history
        # self.update_statistics_charts() # This will now correctly show placeholder
        if not self.performance_history.history:
            self.stats_placeholder.pack(expand=True) # Show placeholder again if history is cleared
            # Also need to clear any existing canvas from the frame if it was previously shown
            for widget in self.stats_canvas_frame.winfo_children():
                widget.destroy()
        else:
            self.stats_placeholder.pack_forget()

        self.current_analysis = None
        self.status_var.set("Ready")
    
    def export_results(self):
        """Export results to JSON file"""
        if not self.current_analysis:
            messagebox.showinfo("No Results", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"primality_{self.current_analysis.number}.json"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_analysis.to_dict(), f, indent=2)
                messagebox.showinfo("Success", f"Results exported to {filename}")
                self.status_var.set(f"Exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")
    
    def batch_test(self):
        """Run tests on multiple numbers from a file"""
        filename = filedialog.askopenfilename(
            title="Select file with numbers",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            numbers_to_process = []
            with open(filename, 'r') as f:
                for line in f:
                    cleaned_line = line.strip().replace(",", "")
                    if cleaned_line:
                        try:
                            numbers_to_process.append(int(cleaned_line))
                        except ValueError:
                            messagebox.showwarning("Invalid Line", f"Skipping invalid number in file: '{line.strip()}'")
            
            if not numbers_to_process:
                messagebox.showwarning("Empty File", "The selected file contains no valid numbers.")
                return
            
            if len(numbers_to_process) > 100:
                if not messagebox.askyesno("Large Batch", 
                                          f"This will test {len(numbers_to_process)} numbers. This may take a while. Continue?"):
                    return
            
            # Create progress window
            progress = tk.Toplevel(self.root)
            progress.title("Batch Testing")
            progress.geometry("400x300")
            progress.configure(bg=ModernTheme.BACKGROUND)
            
            # Center progress window
            progress.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - progress.winfo_width()) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - progress.winfo_height()) // 2
            progress.geometry(f"+{x}+{y}")
            
            # Progress UI
            ttk.Label(progress,
                     text=f"Testing {len(numbers_to_process)} numbers...",
                     font=ModernTheme.HEADING_FONT,
                     background=ModernTheme.BACKGROUND,
                     foreground=ModernTheme.TEXT).pack(pady=20)
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress,
                                          variable=progress_var,
                                          maximum=len(numbers_to_process))
            progress_bar.pack(fill="x", padx=20, pady=10)
            
            status_label = ttk.Label(progress,
                                    text="Starting...",
                                    background=ModernTheme.BACKGROUND,
                                    foreground=ModernTheme.TEXT_TERTIARY)
            status_label.pack(pady=10)
            
            results_text = scrolledtext.ScrolledText(progress,
                                                    height=8,
                                                    font=ModernTheme.MONOSPACE_FONT,
                                                    bg=ModernTheme.CARD_BG,
                                                    fg=ModernTheme.TEXT)
            results_text.pack(fill="both", expand=True, padx=20, pady=10)
            
            analysis_results_for_export = [] # Store full analysis objects for JSON export
            
            # Determine which tests to run for batch processing (use currently selected ones)
            selected_tests = [test for test, var in self.test_vars.items() if var.get()]
            if not selected_tests:
                messagebox.showwarning("No Tests Selected", "Please select at least one test for batch processing.")
                progress.destroy()
                return

            for i, n in enumerate(numbers_to_process, 1):
                progress_var.set(i)
                status_label.config(text=f"Testing {n:,} ({i}/{len(numbers_to_process)})")
                progress.update()
                
                try:
                    # Use analyze_number_api for batch processing to get all test results
                    analysis = self.test_manager.analyze_number(n, selected_tests)
                    analysis_results_for_export.append(analysis.to_dict()) # Store as dict for JSON
                    
                    verdict_char = "?"
                    if analysis.overall_verdict == "Definitely Prime":
                        verdict_char = "P"
                    elif analysis.overall_verdict == "Definitely Composite":
                        verdict_char = "C"
                    elif "Probably" in analysis.overall_verdict or "Almost Certainly" in analysis.overall_verdict:
                        verdict_char = "p"
                    
                    results_text.insert(tk.END, f"{n:12,} : {verdict_char} {analysis.overall_verdict}\n")
                    results_text.see(tk.END)
                except Exception as e:
                    results_text.insert(tk.END, f"{n:12,} : ERROR ({str(e)[:50]})\n")
                    analysis_results_for_export.append({"number": n, "error": str(e)}) # Store error info
            
            # Save results
            timestamp = int(time.time())
            output_file_base = f"batch_results_{timestamp}"
            output_file_json = f"{output_file_base}.json"
            output_file_txt = f"{output_file_base}_summary.txt"

            # Save JSON
            with open(output_file_json, 'w') as f:
                json.dump(analysis_results_for_export, f, indent=2)
            
            # Save summary text (optional, but good for quick review)
            summary_content = results_text.get("1.0", tk.END)
            with open(output_file_txt, 'w') as f:
                f.write(f"Batch Test Summary ({len(numbers_to_process)} numbers)\n")
                f.write(f"Timestamp: {time.ctime(timestamp)}\n")
                f.write("----------------------------------------\n\n")
                f.write(summary_content)

            status_label.config(text=f"Complete! Results saved to {output_file_json} and {output_file_txt}")
            
            # Add close button
            close_btn = RoundedButton(progress,
                                     text="Close",
                                     command=progress.destroy,
                                     bg=ModernTheme.ACCENT_PRIMARY,
                                     fg=ModernTheme.PRIMARY,
                                     font=("Segoe UI", 10, "bold"),
                                     padx=20, pady=8,
                                     hover_bg=ModernTheme.ACCENT_SECONDARY)
            close_btn.pack(pady=10)
            
        except FileNotFoundError:
            messagebox.showerror("File Error", f"Could not find the specified file: {filename}")
        except Exception as e:
            messagebox.showerror("Batch Test Error", f"An unexpected error occurred during batch testing:\n{str(e)}")
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PrimalityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
