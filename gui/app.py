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
from simple_ga import SimpleGA

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
        tk.Button(control_frame, text="Запуск (least squares)", command=self.run_fit).pack(fill=tk.X, pady=10)

        tk.Button(control_frame, text="Симуляция без оптимизации", command=self.print_frob1_output).pack(fill=tk.X)
        tk.Button(control_frame, text="График эксп. данных", command=self.show_p27_plot).pack(fill=tk.X)

        self.reaction_listbox = tk.Listbox(display_frame, selectmode=tk.MULTIPLE, width=100)
        self.reaction_listbox.pack(fill=tk.BOTH, expand=True)

        tk.Button(control_frame, text="Сменить Fit", command=self.toggle_fit_flag).pack(fill=tk.X)
        tk.Button(control_frame, text="Удалить выбранное", command=self.delete_selected).pack(fill=tk.X)
        tk.Button(control_frame, text="Генетический алгоритм", command=self.run_genetic_algorithm, bg='lightgreen').pack(fill=tk.X, pady=10)
        tk.Button(control_frame, text="ГА калибровка", command=self.run_simple_ga, bg='orange').pack(fill=tk.X, pady=5)

    def check_prerequisites(self):
        """Проверка необходимых условий перед запуском GA"""
        if not self.p27_file:
            messagebox.showwarning("Предупреждение", "Сначала загрузите экспериментальные данные")
            return False
        if not self.rs_file:
            messagebox.showwarning("Предупреждение", "Сначала загрузите файл реакций")
            return False
        selected_indices = self.reaction_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Предупреждение", "Выберите реакции для калибровки")
            return False
        return True

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
        """Запуск оптимизации методом наименьших квадратов"""
        selected_indices = self.reaction_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "No reactions selected")
            return
        selected_rs = [self.rs[i] for i in selected_indices]
        selected_lines = [self.reaction_sources[i] for i in selected_indices]
        self._execute_fit(selected_rs, selected_lines)

    def _execute_fit(self, selected_rs, selected_lines):
        """Выполнение оптимизации методом наименьших квадратов"""
        try:
            print("\n" + "="*60)
            print("🚀 ЗАПУСК МЕТОДА НАИМЕНЬШИХ КВАДРАТОВ")
            print("="*60)
            
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
            
            # Определяем функцию для оптимизации
            frob1 = lambda par, tlist, NoPlot: frob(par, list(range(len(selected_rs))), names, pops, 
                                                  np.ones(len(selected_rs)), tlist, y0, selected_rs, specs, dct, errs, NoPlot, use_ch4_correction=True)
            
            print(f"\nНачальные параметры: {par}")
            print(f"Границы: нижние={b1}, верхние={b2}")
            
            # Запуск оптимизации
            fitres = least_squares(frob1, par, bounds=bounds, args=(tlist, True), verbose=2)
            
            print(f"\n" + "="*60)
            print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
            print("="*60)
            print(f"Количество вызовов функции: {fitres.nfev}")
            print(f"Количество итераций: {fitres.njev}")
            print(f"Статус: {fitres.status}")
            print(f"Сообщение: {fitres.message}")
            print(f"\nОптимизированные параметры:")
            for i, param in enumerate(fitres.x):
                print(f"  par[{i}]: {param:.6e}")
            print(f"Сумма квадратов остатков: {np.sum(fitres.fun**2):.6e}")
            
            # Построение графика
            frob1(fitres.x, tlist, False)
            plt.title("Результат оптимизации методом наименьших квадратов")
            plt.legend()
            
            # Сохранение графика
            import datetime
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            filename = f"fit_least_squares_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nГрафик сохранен как: {filename}")
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()

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

    def run_genetic_algorithm(self):
        """Запуск генетического алгоритма с настраиваемыми параметрами"""
        if not self.check_prerequisites():
            return
    
        # Создаем окно с параметрами GA
        ga_window = tk.Toplevel(self.root)
        ga_window.title("Генетический алгоритм - Настройки")
        ga_window.geometry("400x300")
    
        # Параметры GA
        tk.Label(ga_window, text="Размер популяции:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        pop_size_var = tk.StringVar(value="30")
        tk.Entry(ga_window, textvariable=pop_size_var, width=15).grid(row=0, column=1, padx=10, pady=10)
    
        tk.Label(ga_window, text="Количество поколений:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        generations_var = tk.StringVar(value="50")
        tk.Entry(ga_window, textvariable=generations_var, width=15).grid(row=1, column=1, padx=10, pady=10)
    
        tk.Label(ga_window, text="Вероятность мутации:", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
        mutation_rate_var = tk.StringVar(value="0.1")
        tk.Entry(ga_window, textvariable=mutation_rate_var, width=15).grid(row=2, column=1, padx=10, pady=10)
    
        tk.Label(ga_window, text="Элитных особей:", font=("Arial", 10)).grid(row=3, column=0, sticky=tk.W, padx=10, pady=10)
        elite_count_var = tk.StringVar(value="2")
        tk.Entry(ga_window, textvariable=elite_count_var, width=15).grid(row=3, column=1, padx=10, pady=10)
    
        # Переменная для выбора уточнения после GA
        refine_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ga_window, text="Уточнить методом наименьших квадратов", 
                  variable=refine_var, font=("Arial", 9)).grid(row=4, column=0, columnspan=2, pady=10)
    
        def start_ga():
            """Функция запуска GA с выбранными параметрами"""
            try:
                # Получаем параметры
                pop_size = int(pop_size_var.get())
                generations = int(generations_var.get())
                mutation_rate = float(mutation_rate_var.get())
                elite_count = int(elite_count_var.get())
            
                # Закрываем окно настроек
                ga_window.destroy()
            
                # Запускаем GA с выбранными параметрами
                self._run_ga_with_params(pop_size, generations, mutation_rate, elite_count, refine_var.get())
            
            except ValueError as e:
                messagebox.showerror("Ошибка", f"Некорректные значения параметров: {e}")
    
        # Кнопки
        button_frame = tk.Frame(ga_window)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
    
        tk.Button(button_frame, text="Запуск", command=start_ga, bg='lightgreen', width=15, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Отмена", command=ga_window.destroy, bg='lightcoral', width=15).pack(side=tk.LEFT, padx=10)

    def run_simple_ga(self):
        """Простой запуск генетического алгоритма"""
        if not self.check_prerequisites():
            return
        
        selected_indices = self.reaction_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "No reactions selected")
            return
        
        selected_rs = [self.rs[i] for i in selected_indices]
        selected_lines = [self.reaction_sources[i] for i in selected_indices]
        
        try:
            print("\n" + "="*60)
            print("🧬 ЗАПУСК ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
            print("="*60)
            
            # Загружаем экспериментальные данные
            tlist, names, pops, errs = loaddata(self.p27_file, NoPlot=True, errname=self.p27_err_file)
            
            # Подготавливаем данные для моделирования
            specs = make_species_list(selected_rs)
            dct = make_species_dictionary(specs)
            y0 = np.zeros(len(specs))
            
            for i in range(len(names)):
                if names[i] in dct:
                    val = pops[i][0] if isinstance(pops[i], (np.ndarray, list, pd.Series)) else pops[i]
                    y0[dct[names[i]]] = float(val)
            
            # Находим максимальный индекс параметров
            max_index = 0
            for line in selected_lines:
                matches = re.findall(r"par\[(\d+)\]", line)
                indices = [int(i) for i in matches]
                if indices:
                    max_index = max(max_index, max(indices))
            
            # Размер полного массива параметров
            par_size = max(len(selected_rs), max_index + 1) + 2
            
            # Создаем фитнес-функцию для GA
            def fitness_func(params):
                """Фитнес-функция для GA: отрицательная сумма квадратов остатков"""
                try:
                    # params - массив констант скоростей (для GA)
                    # Создаем полный массив параметров для frob
                    full_params = np.ones(par_size)
                    
                    # Заполняем первые len(selected_rs) параметров
                    for i in range(min(len(params), len(selected_rs))):
                        full_params[i] = params[i]
                    
                    # Вычисляем невязки
                    residuals = frob(
                        full_params,
                        list(range(len(selected_rs))),
                        names, pops,
                        np.ones(len(selected_rs)),
                        tlist, y0, selected_rs, specs, dct, errs,
                        NoPlot=True,
                        use_ch4_correction=True
                    )
                    
                    # Сумма квадратов остатков (чем меньше, тем лучше)
                    ssr = np.sum(residuals**2)
                    
                    # GA максимизирует, поэтому возвращаем отрицательное значение
                    return -ssr
                    
                except Exception as e:
                    print(f"Ошибка в фитнес-функции: {e}")
                    return -1e10  # Очень плохой фитнес при ошибке
            
            # Запускаем GA
            n_params = len(selected_rs)  # Количество оптимизируемых параметров
            bounds = [(1e-20, 1e10)] * n_params  # Границы для констант скоростей
            
            print(f"\nПараметры GA:")
            print(f"  Количество параметров: {n_params}")
            print(f"  Размер популяции: 30")
            print(f"  Количество поколений: 50")
            print(f"  Границы параметров: от {bounds[0][0]} до {bounds[0][1]}")
            
            ga = SimpleGA(n_params, bounds, pop_size=30, generations=50, verbose=2)
            best_params, best_fitness = ga.optimize(fitness_func)
            
            print(f"\n" + "="*60)
            print("✅ ОПТИМИЗАЦИЯ ГА ЗАВЕРШЕНА")
            print("="*60)
            print(f"Лучший фитнес (отрицательная сумма квадратов): {best_fitness:.6f}")
            print(f"Сумма квадратов остатков: {-best_fitness:.6e}")
            print(f"\nОптимизированные параметры:")
            for i, param in enumerate(best_params):
                print(f"  k{i}: {param:.6e}")
            
            # Построение графика сходмости GA
            ga.plot_convergence()
            
            # Создаем полный массив параметров для построения графика
            full_params = np.ones(par_size)
            for i in range(len(best_params)):
                full_params[i] = best_params[i]
            
            # Построение графика результатов
            frob1 = lambda par, tlist, NoPlot: frob(
                par, list(range(len(selected_rs))), names, pops,
                np.ones(len(selected_rs)), tlist, y0, selected_rs, specs, dct, errs,
                NoPlot, use_ch4_correction=True
            )
            
            print("\nПостроение графика с оптимизированными параметрами...")
            frob1(full_params, tlist, False)
            plt.title("Результат оптимизации генетическим алгоритмом")
            plt.legend()
            
            # Сохранение графика
            import datetime
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            filename = f"fit_genetic_algorithm_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nГрафик сохранен как: {filename}")
            plt.show()
            
            # Показываем результаты в messagebox
            result_text = f"Генетический алгоритм завершен!\n\n"
            result_text += f"Лучший фитнес: {best_fitness:.6f}\n"
            result_text += f"Сумма квадратов остатков: {-best_fitness:.6e}\n\n"
            result_text += "Оптимизированные параметры:\n"
            for i, param in enumerate(best_params):
                result_text += f"  k{i}: {param:.6e}\n"
            
            messagebox.showinfo("Результаты GA", result_text)
            
            # Предлагаем уточнить методом наименьших квадратов
            if messagebox.askyesno("Уточнение", "Использовать найденные параметры для уточнения методом наименьших квадратов?"):
                print("\n" + "="*60)
                print("🔄 УТОЧНЕНИЕ МЕТОДОМ НАИМЕНЬШИХ КВАДРАТОВ")
                print("="*60)
                
                # Создаем начальное приближение для least_squares
                initial_par = np.ones(par_size)
                for i in range(len(best_params)):
                    initial_par[i] = best_params[i]
                
                # Границы для least_squares
                b1 = np.hstack((1e-7*np.ones(par_size - 2), 0.01*np.ones(2)))
                b2 = np.hstack((1e7*np.ones(par_size - 2), 1000*np.ones(2)))
                bounds_ls = (b1, b2)
                
                print(f"\nНачальные параметры для least_squares (из GA):")
                for i, param in enumerate(initial_par[:len(best_params)]):
                    print(f"  par[{i}]: {param:.6e}")
                
                # Запуск least_squares
                fitres = least_squares(frob1, initial_par, bounds=bounds_ls, args=(tlist, True), verbose=2)
                
                print(f"\n" + "="*60)
                print("✅ УТОЧНЕНИЕ ЗАВЕРШЕНО")
                print("="*60)
                print(f"Количество вызовов функции: {fitres.nfev}")
                print(f"Количество итераций: {fitres.njev}")
                print(f"Статус: {fitres.status}")
                print(f"Сообщение: {fitres.message}")
                print(f"\nУточненные параметры:")
                for i, param in enumerate(fitres.x):
                    print(f"  par[{i}]: {param:.6e}")
                print(f"Сумма квадратов остатков: {np.sum(fitres.fun**2):.6e}")
                
                # Построение графика уточненных результатов
                frob1(fitres.x, tlist, False)
                plt.title("Результат уточнения методом наименьших квадратов (после GA)")
                plt.legend()
                
                timestamp = datetime.datetime.now().strftime("%H-%M-%S")
                filename = f"fit_GA_refined_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"\nГрафик сохранен как: {filename}")
                plt.show()
                
        except Exception as e:
            messagebox.showerror("Ошибка GA", str(e))
            import traceback
            traceback.print_exc()

    def _run_ga_with_params(self, pop_size, generations, mutation_rate, elite_count, refine_after=True):
        """Запуск GA с заданными параметрами"""
        try:
            print("\n" + "="*70)
            print("🧬 ЗАПУСК ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
            print("="*70)
        
            selected_indices = self.reaction_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Предупреждение", "Не выбраны реакции")
                return
        
            selected_rs = [self.rs[i] for i in selected_indices]
            selected_lines = [self.reaction_sources[i] for i in selected_indices]
        
            # Загружаем экспериментальные данные
            tlist, names, pops, errs = loaddata(self.p27_file, NoPlot=True, errname=self.p27_err_file)
        
            # Подготавливаем данные для моделирования
            specs = make_species_list(selected_rs)
            dct = make_species_dictionary(specs)
            y0 = np.zeros(len(specs))
        
            for i in range(len(names)):
                if names[i] in dct:
                    val = pops[i][0] if isinstance(pops[i], (np.ndarray, list, pd.Series)) else pops[i]
                    y0[dct[names[i]]] = float(val)
        
            # Находим максимальный индекс параметров
            max_index = 0
            for line in selected_lines:
                matches = re.findall(r"par\[(\d+)\]", line)
                indices = [int(i) for i in matches]
                if indices:
                    max_index = max(max_index, max(indices))
        
            # Размер полного массива параметров
            par_size = max(len(selected_rs), max_index + 1) + 2
        
            # Создаем фитнес-функцию для GA
            def fitness_func(params):
                """Фитнес-функция для GA: отрицательная сумма квадратов остатков"""
                try:
                    # params - массив констант скоростей (для GA)
                    # Создаем полный массив параметров для frob
                    full_params = np.ones(par_size)
                
                    # Заполняем первые len(selected_rs) параметров
                    for i in range(min(len(params), len(selected_rs))):
                        full_params[i] = params[i]
                
                    # Вычисляем невязки
                    residuals = frob(
                        full_params,
                        list(range(len(selected_rs))),
                        names, pops,
                        np.ones(len(selected_rs)),
                        tlist, y0, selected_rs, specs, dct, errs,
                        NoPlot=True,
                        use_ch4_correction=True
                    )
                
                    # Сумма квадратов остатков (чем меньше, тем лучше)
                    ssr = np.sum(residuals**2)
                
                    # GA максимизирует, поэтому возвращаем отрицательное значение
                    return -ssr
                
                except Exception as e:
                    print(f"Ошибка в фитнес-функции: {e}")
                    return -1e10  # Очень плохой фитнес при ошибке
        
            # Запускаем GA
            n_params = len(selected_rs)  # Количество оптимизируемых параметров
            bounds = [(1e-20, 1e10)] * n_params  # Границы для констант скоростей
        
            print(f"\n📊 ПАРАМЕТРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА:")
            print(f"   Количество реакций: {n_params}")
            print(f"   Размер популяции: {pop_size}")
            print(f"   Поколений: {generations}")
            print(f"   Вероятность мутации: {mutation_rate}")
            print(f"   Границы параметров: от 1e-20 до 1e10")
        
            ga = SimpleGA(
                n_params=n_params,
                bounds=bounds,
                pop_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate,
                elite_count=elite_count,
                verbose=2
            )
        
            best_params, best_fitness = ga.optimize(fitness_func)
        
            print(f"\n" + "="*70)
            print("✅ ОПТИМИЗАЦИЯ ГА ЗАВЕРШЕНА")
            print("="*70)
            print(f"Лучший фитнес (отрицательная сумма квадратов): {best_fitness:.6f}")
            print(f"Сумма квадратов остатков: {-best_fitness:.6e}")
            print(f"\n📈 ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ:")
            for i, param in enumerate(best_params):
                print(f"   k{i}: {param:.6e}")
        
            # Построение графика сходимости GA
            print("\n📊 Построение графика сходимости GA...")
            # Вместо:
            # ga.plot_convergence()

            # Используйте:
            if hasattr(ga, 'plot_convergence'):
                print("\n📊 Построение графика сходимости GA...")
                ga.plot_convergence()
            else:
                # Создаем свой график сходимости
                print("\n📊 Построение графика сходимости GA...")
                if hasattr(ga, 'best_fitness_history') and ga.best_fitness_history:
                    import matplotlib.pyplot as plt
                    import datetime
        
                    plt.figure(figsize=(10, 6))
                    generations = range(1, len(ga.best_fitness_history) + 1)
        
                    plt.plot(generations, ga.best_fitness_history, 'b-', linewidth=2, label='Лучший фитнес')
        
                    if hasattr(ga, 'avg_fitness_history') and ga.avg_fitness_history:
                        plt.plot(generations, ga.avg_fitness_history, 'r--', linewidth=1.5, alpha=0.7, label='Средний фитнес')
        
                    plt.xlabel('Поколение', fontsize=12)
                    plt.ylabel('Фитнес (чем выше, тем лучше)', fontsize=12)
                    plt.title('Сходимость генетического алгоритма', fontsize=14, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.legend(fontsize=11)
                    plt.tight_layout()
        
                    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
                    filename = f"ga_convergence_{timestamp}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"💾 График сходимости сохранен как: {filename}")
                    plt.show()
                else:
                    print("⚠️  История фитнеса недоступна для построения графика")
        
            # Создаем полный массив параметров для построения графика
            full_params = np.ones(par_size)
            for i in range(len(best_params)):
                full_params[i] = best_params[i]
        
            # Построение графика результатов
            frob1 = lambda par, tlist, NoPlot: frob(
                par, list(range(len(selected_rs))), names, pops,
                np.ones(len(selected_rs)), tlist, y0, selected_rs, specs, dct, errs,
                NoPlot, use_ch4_correction=True
            )
        
            print("\n📈 Построение графика с оптимизированными параметрами...")
            frob1(full_params, tlist, False)
            plt.title(f"Результат оптимизации генетическим алгоритмом\n"
                     f"Популяция: {pop_size}, Поколений: {generations}, Мутация: {mutation_rate}")
            plt.legend()
        
            # Сохранение графика
            import datetime
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            filename = f"fit_genetic_algorithm_{timestamp}.png"
#            plt.savefig(filename, dpi=300, bbox_inches='tight')
#            print(f"\n💾 График сохранен как: {filename}")
            plt.show()
        
            # Уточнение методом наименьших квадратов, если выбрано
            if refine_after:
                if messagebox.askyesno("Уточнение", 
                                  "Использовать найденные параметры для уточнения методом наименьших квадратов?"):
                
                    print("\n" + "="*70)
                    print("🔄 УТОЧНЕНИЕ МЕТОДОМ НАИМЕНЬШИХ КВАДРАТОВ")
                    print("="*70)
                
                    # Создаем начальное приближение для least_squares
                    initial_par = np.ones(par_size)
                    for i in range(len(best_params)):
                        initial_par[i] = best_params[i]
                
                    # Границы для least_squares
                    b1 = np.hstack((1e-7*np.ones(par_size - 2), 0.01*np.ones(2)))
                    b2 = np.hstack((1e7*np.ones(par_size - 2), 1000*np.ones(2)))
                    bounds_ls = (b1, b2)
                
                    print(f"\nНачальные параметры для least_squares (из GA):")
                    for i, param in enumerate(initial_par[:len(best_params)]):
                        print(f"   par[{i}]: {param:.6e}")
                
                    # Запуск least_squares
                    fitres = least_squares(frob1, initial_par, bounds=bounds_ls, args=(tlist, True), verbose=2)
                
                    print(f"\n" + "="*70)
                    print("✅ УТОЧНЕНИЕ ЗАВЕРШЕНО")
                    print("="*70)
                    print(f"Количество вызовов функции: {fitres.nfev}")
                    print(f"Количество итераций: {fitres.njev}")
                    print(f"Статус: {fitres.status}")
                    print(f"Сообщение: {fitres.message}")
                    print(f"\n📈 УТОЧНЕННЫЕ ПАРАМЕТРЫ:")
                    for i, param in enumerate(fitres.x[:len(best_params)]):
                        print(f"   par[{i}]: {param:.6e}")
                    print(f"Сумма квадратов остатков: {np.sum(fitres.fun**2):.6e}")
                
                    # Построение графика уточненных результатов
                    frob1(fitres.x, tlist, False)
                    plt.title("Результат уточнения методом наименьших квадратов (после GA)")
                    plt.legend()
                
                    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
                    filename = f"fit_GA_refined_{timestamp}.png"
#                    plt.savefig(filename, dpi=300, bbox_inches='tight')
#                    print(f"\n💾 График сохранен как: {filename}")
                    plt.show()
        
        except Exception as e:
            messagebox.showerror("Ошибка GA", str(e))
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = ReactionApp(root)
    root.mainloop()