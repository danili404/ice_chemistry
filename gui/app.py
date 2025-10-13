import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import importlib.util
import pandas as pd
from core.reactions import rcn, make_species_list, make_species_dictionary
from core.data_loader import loaddata
from core.kinetics import kin_solve, frob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import re

class ReactionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reaction Fit GUI")

        self.p27_file = None
        self.p27_err_file = None
        self.rs_file = None
        self.rs = []
        self.reaction_sources = []
        self.selected_reactions = []

        self.setup_ui()

    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        display_frame = tk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Button(control_frame, text="Загрузка эксп. данных", command=self.load_p27).pack(fill=tk.X)
        tk.Button(control_frame, text="Загрузка ошибок", command=self.load_p27_error).pack(fill=tk.X)
        tk.Button(control_frame, text="Загрузка файла реакций", command=self.load_rs).pack(fill=tk.X)
        tk.Button(control_frame, text="Добавить реакцию", command=self.add_reaction).pack(fill=tk.X)
        tk.Button(control_frame, text="Запуск", command=self.run_fit).pack(fill=tk.X, pady=10)

        tk.Button(control_frame, text="Симуляция без оптимизации", command=self.print_frob1_output).pack(fill=tk.X)
        tk.Button(control_frame, text="График эксп. данных", command=self.show_p27_plot).pack(fill=tk.X)


        self.reaction_listbox = tk.Listbox(display_frame, selectmode=tk.MULTIPLE, width=100)
        self.reaction_listbox.pack(fill=tk.BOTH, expand=True)

        # Новые кнопки управления реакциями
        tk.Button(control_frame, text="Сменить Fit", command=self.toggle_fit_flag).pack(fill=tk.X)
        tk.Button(control_frame, text="Удалить выбранное", command=self.delete_selected).pack(fill=tk.X)

    def load_p27(self):
        self.p27_file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if self.p27_file:
            messagebox.showinfo("Файл загружен", f"Loaded: {os.path.basename(self.p27_file)}")

    def load_p27_error(self):
        self.p27_err_file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if self.p27_err_file:
            messagebox.showinfo("Файл загружен", f"Loaded: {os.path.basename(self.p27_err_file)}")

    def load_rs(self):
        self.rs_file = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not self.rs_file:
            return
        with open(self.rs_file, 'r') as f:
            lines = f.readlines()
        self.rs.clear()
        self.reaction_sources.clear()
        self.reaction_listbox.delete(0, tk.END)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                result = eval(line)
                reaction = result[0] if isinstance(result, tuple) else result
                if isinstance(reaction, rcn):
                    self.rs.append(reaction)
                    self.reaction_sources.append(line)
                    self.reaction_listbox.insert(tk.END, line)
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}")

    def add_reaction(self):
        formula = simpledialog.askstring("Ввод", "Введите формулу реакции:")
        if not formula:
            return
        try:
            rcf = float(simpledialog.askstring("Ввод", "Enter rcf (rate coefficient):"))
        except:
            messagebox.showerror("Error", "Invalid rcf value")
            return
        name = simpledialog.askstring("Ввод", "Введите название реакции:")
        par_index = simpledialog.askstring("Ввод", "Enter par index (e.g. 0):")
        if not par_index or not par_index.isdigit():
            messagebox.showerror("Error", "par index must be an integer")
            return

        time_expr = simpledialog.askstring("Ввод", "Enter time dependence (e.g. t<60), or leave blank:")
        if time_expr:
            fun_expr = f"par[{par_index}]*({time_expr})"
        else:
            fun_expr = f"par[{par_index}]"

        line = f"rcn('{formula}',rcf={rcf},name='{name}',fit=True,fun=lambda t,par: {fun_expr})"

        try:
            new_r = eval(line)
            with open(self.rs_file, 'a') as f:
                f.write(line + "")
            self.rs.append(new_r)
            self.reaction_sources.append(line)
            self.reaction_listbox.insert(tk.END, f"{new_r.name}: {new_r.formula}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add reaction: {e}")

    def toggle_fit_flag(self):
        selected = list(self.reaction_listbox.curselection())
        if not selected or not self.rs_file:
            return
        with open(self.rs_file, 'r') as f:
            lines = f.readlines()
        for i in selected:
            original = lines[i].strip()
            if 'fit=True' in original:
                lines[i] = original.replace('fit=True', 'fit=False') + ''
            elif 'fit=False' in original:
                lines[i] = original.replace('fit=False', 'fit=True') + ''
        with open(self.rs_file, 'w') as f:
            f.writelines(lines)
        self.load_rs()

    def delete_selected(self):
        selected = list(self.reaction_listbox.curselection())
        if not selected or not self.rs_file:
            return
        with open(self.rs_file, 'r') as f:
            lines = f.readlines()
        lines = [line for i, line in enumerate(lines) if i not in selected]
        with open(self.rs_file, 'w') as f:
            f.writelines(lines)
        self.load_rs()

    def run_fit(self):
        selected_indices = self.reaction_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "No reactions selected")
            return
        selected_rs = [self.rs[i] for i in selected_indices]
        selected_lines = [self.reaction_sources[i] for i in selected_indices]
        self._execute_fit(selected_rs, selected_lines)

    def _execute_fit(self, selected_rs, selected_lines):
        try:
            tlist, names, pops, errs = loaddata(self.p27_file, NoPlot=True, errname=self.p27_err_file)
            specs = make_species_list(selected_rs)
            dct = make_species_dictionary(specs)
            y0 = np.zeros(len(specs))
            for i in range(len(names)):
                if names[i] in dct:
                    val = pops[i][0] if isinstance(pops[i], (np.ndarray, list, pd.Series)) else pops[i]
                    y0[dct[names[i]]] = float(val)

            max_index = 0
            for line in selected_lines:
                matches = re.findall(r"par\[(\d+)\]", line)
                indices = [int(i) for i in matches]
                if indices:
                    max_index = max(max_index, max(indices))

            par_size = max(len(selected_rs), max_index + 1) + 2
            par = np.ones(par_size)
            b1 = np.hstack((1e-7*np.ones(par_size - 2), 0.01*np.ones(2)))
            b2 = np.hstack((1e7*np.ones(par_size - 2), 1000*np.ones(2)))
            bounds = (b1, b2)
            frob1 = lambda par, tlist, NoPlot: frob(par, list(range(len(selected_rs))), names, pops, 
                                                  np.ones(len(selected_rs)), tlist, y0, selected_rs, specs, dct, errs, NoPlot, use_ch4_correction=True)
            fitres = least_squares(frob1, par, bounds=bounds, args=(tlist, True), verbose=2)
            frob1(fitres.x, tlist, False)
            print("Optimized parameters:", fitres.x)
            plt.legend()
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            formula_str = "fit"
            filename = f"fit_{formula_str}_{timestamp}.png"
            plt.savefig(filename)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def print_frob1_output(self):
        selected_indices = self.reaction_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "No reactions selected")
            return
        selected_rs = [self.rs[i] for i in selected_indices]
        selected_lines = [self.reaction_sources[i] for i in selected_indices]
        try:
            tlist, names, pops, errs = loaddata(self.p27_file, NoPlot=True, errname=self.p27_err_file)
            specs = make_species_list(selected_rs)
            dct = make_species_dictionary(specs)
            y0 = np.zeros(len(specs))
            for i in range(len(names)):
                if names[i] in dct:
                    val = pops[i][0] if isinstance(pops[i], (np.ndarray, list, pd.Series)) else pops[i]
                    y0[dct[names[i]]] = float(val)

            max_index = 0
            for line in selected_lines:
                matches = re.findall(r"par\[(\d+)\]", line)
                indices = [int(i) for i in matches]
                if indices:
                    max_index = max(max_index, max(indices))

            par_size = max(len(selected_rs), max_index + 1) + 2
            par = np.ones(par_size)
            frob1 = lambda par, tlist, NoPlot: frob(
                par, list(range(len(selected_rs))), names, pops,
                np.ones(len(selected_rs)), tlist, y0, selected_rs, specs, dct, errs, NoPlot,
                use_ch4_correction=True
            )
            out = frob1(par, tlist, True)
            print("frob1 raw output:", out)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_p27_plot(self):
        try:
            tlist, names, pops, errs = loaddata(self.p27_file, NoPlot=False, errname=self.p27_err_file)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ReactionApp(root)
    root.mainloop()
