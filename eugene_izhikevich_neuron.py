import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

class IzhikevichGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Izhikevich Neuron Model")
        
        self.initial_values = {
            'I': 10,  # Typical value for stimulation current
            'v': -65,
            'a': 0.02,
            'b': 0.2,
            'c': -65,
            'd': 8
        }
        
        self.preset_values = {
            'RS': {'I': 10, 'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
            'IB': {'I': 10, 'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
            'CH': {'I': 10, 'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
            'FS': {'I': 10, 'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
            'LTS': {'I': 10, 'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
            'TC': {'I': 10, 'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
            'RZ': {'I': 10, 'a': 0.1, 'b': 0.26, 'c': -65, 'd': 2}
        }
        
        self.value_labels = {}  # Dictionary to hold value labels
        self.current_neuron_type = 'RS'
        self.create_widgets()
        
    def create_widgets(self):
        self.sliders_frame = ttk.Frame(self.root)
        self.sliders_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.I_slider = self.create_slider("I (mA)", -20, 20, 1, self.initial_values['I'])
        self.v_slider = self.create_slider("v (mV)", -200, 200, 1, self.initial_values['v'])
        self.a_slider = self.create_slider("a", 0, 0.1, 0.01, self.initial_values['a'])
        self.b_slider = self.create_slider("b", 0, 0.3, 0.01, self.initial_values['b'])
        self.c_slider = self.create_slider("c", -80, -40, 1, self.initial_values['c'])
        self.d_slider = self.create_slider("d", 0, 30, 1, self.initial_values['d'])
        
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, pady=10)
        
        self.reset_button = ttk.Button(self.button_frame, text="Reset", command=self.reset_sliders)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.create_preset_buttons()

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Initial plot
        self.plot_graph()
        
    def create_slider(self, label_text, min_val, max_val, step, initial_value):
        frame = ttk.Frame(self.sliders_frame)
        frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        label = ttk.Label(frame, text=label_text)
        label.pack(side=tk.LEFT, padx=5)
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL)
        slider.set(initial_value)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        value_label = ttk.Label(frame, text=f"{initial_value:.2f}")
        value_label.pack(side=tk.LEFT, padx=5)
        
        slider.bind("<Motion>", lambda event: self.update_value_label_and_plot(value_label, slider))
        
        self.value_labels[slider] = value_label  # Store the value label in the dictionary
        
        return slider
    
    def create_preset_buttons(self):
        preset_frame = ttk.Frame(self.root)
        preset_frame.pack(side=tk.TOP, pady=5)

        for neuron_type in self.preset_values.keys():
            button = ttk.Button(preset_frame, text=neuron_type, command=lambda n=neuron_type: self.set_preset_values(n))
            button.pack(side=tk.LEFT, padx=5)

    def set_preset_values(self, neuron_type):
        self.current_neuron_type = neuron_type
        preset = self.preset_values[neuron_type]
        self.I_slider.set(preset['I'])
        self.a_slider.set(preset['a'])
        self.b_slider.set(preset['b'])
        self.c_slider.set(preset['c'])
        self.d_slider.set(preset['d'])

        # Update value labels
        for slider, label in self.value_labels.items():
            value = slider.get()
            label.config(text=f"{value:.2f}")

        self.plot_graph()
    
    def update_value_label_and_plot(self, label, slider):
        value = slider.get()
        label.config(text=f"{value:.2f}")
        self.plot_graph()
    
    def reset_sliders(self):
        preset = self.preset_values[self.current_neuron_type]
        self.I_slider.set(preset['I'])
        self.a_slider.set(preset['a'])
        self.b_slider.set(preset['b'])
        self.c_slider.set(preset['c'])
        self.d_slider.set(preset['d'])
        
        # Update value labels
        for slider, label in self.value_labels.items():
            value = slider.get()
            label.config(text=f"{value:.2f}")
        
        self.plot_graph()
        
    def plot_graph(self):
        I = self.I_slider.get()
        v = self.v_slider.get()
        a = self.a_slider.get()
        b = self.b_slider.get()
        c = self.c_slider.get()
        d = self.d_slider.get()
        
        if self.current_neuron_type == 'TC':
            t = np.linspace(0, 200, 2000)  # Double the time steps for TC neurons
            dt = t[1] - t[0]  # Time step
            half_step = len(t) // 2
            I_values = np.ones(len(t)) * I
            v_values = [v]
            u = b * v  # Initial value of u
            
            for i in range(1, half_step):
                v, u = self.calculate_v_and_u(v, u, I_values[i], a, b, c, d, dt)
                v_values.append(v)
            
            v = -95  # Reset voltage for the second half
            u = b * v  # Reset recovery variable u
            for i in range(half_step, len(t)):
                v, u = self.calculate_v_and_u(v, u, I_values[i], a, b, c, d, dt)
                v_values.append(v)
        else:
            t = np.linspace(0, 100, 1000)
            dt = t[1] - t[0]  # Time step
            I_values = np.ones(len(t)) * I
            v_values = [v]
            u = b * v  # Initial value of u
            
            for i in range(1, len(t)):
                v, u = self.calculate_v_and_u(v, u, I_values[i], a, b, c, d, dt)
                v_values.append(v)
        
        self.ax.clear()
        self.ax.plot(t, v_values, label='v (mV)')
        
        if self.current_neuron_type == 'TC':
            self.ax.axvline(t[half_step], color='r', linestyle='--', label='Voltage step change')
        
        self.ax.set_title('Neuron Voltage over Time')
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Voltage (mV)')
        self.ax.legend()
        self.canvas.draw()
        mplcursors.cursor(self.ax)  # Add interactive cursor
        
    def on_scroll(self, event):
        if event.button == 'up':
            self.ax.set_xlim(self.ax.get_xlim()[0] * 1.1, self.ax.get_xlim()[1] * 1.1)
            self.ax.set_ylim(self.ax.get_ylim()[0] * 1.1, self.ax.get_ylim()[1] * 1.1)
        elif event.button == 'down':
            self.ax.set_xlim(self.ax.get_xlim()[0] / 1.1, self.ax.get_xlim()[1] / 1.1)
            self.ax.set_ylim(self.ax.get_ylim()[0] / 1.1, self.ax.get_ylim()[1] / 1.1)
        self.canvas.draw_idle()
        
    def dv_dt(self, v, u, I): # t in ms
        return 0.04 * v * v + 5 * v + 140 - u + I
    
    def du_dt(self, v, u, a, b): # t in ms
        return a * (b * v - u)
    
    def calculate_v_and_u(self, v, u, I, a, b, c, d, dt):
        v += dt * self.dv_dt(v, u, I)
        u += dt * self.du_dt(v, u, a, b)
        
        if v >= 30:  # Threshold condition
            v = c
            u += d
        
        return v, u

if __name__ == "__main__":
    root = tk.Tk()
    app = IzhikevichGUI(root)
    root.mainloop()
