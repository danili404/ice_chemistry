import numpy as np
import random
import time
from scipy.optimize import least_squares


'''
class SimpleGA:
    """Простой генетический алгоритм для вашего проекта"""
    
    def __init__(self, n_params, bounds, pop_size=50, generations=100, 
                 mutation_rate=0.1, elite_count=1, verbose=1):
        """
        Инициализация генетического алгоритма
        
        Параметры:
        ----------
        n_params : int
            Количество оптимизируемых параметров
        bounds : list of tuples
            Границы для каждого параметра [(min, max), ...]
        pop_size : int
            Размер популяции
        generations : int
            Количество поколений
        mutation_rate : float
            Вероятность мутации
        elite_count : int
            Количество лучших особей, переходящих в следующее поколение
        verbose : int
            Уровень детализации вывода (0 - нет вывода, 1 - минимальный, 2 - подробный)
        """
        self.n_params = n_params
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.verbose = verbose
        
        # История для отслеживания сходимости
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        
        # Лучшее решение
        self.best_individual = None
        self.best_fitness = -np.inf
        
        # Счетчики
        self.function_evaluations = 0
        
    def optimize(self, fitness_func):
        """
        Основная функция оптимизации
        
        Параметры:
        ----------
        fitness_func : callable
            Функция оценки фитнеса (чем выше, тем лучше)
            
        Возвращает:
        -----------
        best_individual : np.ndarray
            Лучшая найденная особь
        best_fitness : float
            Фитнес лучшей особи
        """
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print("🧬 НАЧАЛО РАБОТЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
            print(f"{'='*60}")
            print(f"Количество параметров: {self.n_params}")
            print(f"Размер популяции: {self.pop_size}")
            print(f"Количество поколений: {self.generations}")
            print(f"Вероятность мутации: {self.mutation_rate}")
            print(f"Количество элитных особей: {self.elite_count}")
            print(f"{'='*60}")
        
        # Сброс счетчиков
        self.function_evaluations = 0
        start_time = time.time()
        
        # 1. Инициализация популяции
        population = self._initialize_population()
        
        for gen in range(self.generations):
            gen_start_time = time.time()
            
            # 2. Оценка фитнеса
            fitness_scores = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                fitness_scores[i] = fitness_func(population[i])
                self.function_evaluations += 1
            
            # 3. Статистика
            current_best_fitness = np.max(fitness_scores)
            current_avg_fitness = np.mean(fitness_scores)
            current_worst_fitness = np.min(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            
            # Сохраняем историю
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(current_avg_fitness)
            self.worst_fitness_history.append(current_worst_fitness)
            
            # 4. Обновление лучшего решения
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = population[best_idx].copy()
            
            # 5. Вывод информации о поколении
            gen_time = time.time() - gen_start_time
            
            if self.verbose >= 2:
                print(f"Поколение {gen+1:3d}/{self.generations}: "
                      f"Лучший = {current_best_fitness:10.4f}, "
                      f"Средний = {current_avg_fitness:10.4f}, "
                      f"Худший = {current_worst_fitness:10.4f}, "
                      f"Время = {gen_time:.2f}с")
            elif self.verbose >= 1 and gen % 10 == 0:
                print(f"Поколение {gen+1:3d}/{self.generations}: "
                      f"Лучший = {current_best_fitness:10.4f}, "
                      f"Средний = {current_avg_fitness:10.4f}")
            
            # 6. Создание новой популяции
            new_population = []
            
            # 6.1. Элитизм
            elite_indices = np.argsort(fitness_scores)[-self.elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 6.2. Создаем остальных особей
            while len(new_population) < self.pop_size:
                # Селекция (турнирная)
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Скрещивание
                child = self._crossover(parent1, parent2)
                
                # Мутация
                child = self._mutate(child)
                
                new_population.append(child)
            
            population = np.array(new_population)
        
        # Завершение оптимизации
        total_time = time.time() - start_time
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print("✅ ОПТИМИЗАЦИЯ ГЕНЕТИЧЕСКИМ АЛГОРИТМОМ ЗАВЕРШЕНА")
            print(f"{'='*60}")
            print(f"Общее время выполнения: {total_time:.2f} секунд")
            print(f"Всего оценок функции: {self.function_evaluations}")
            print(f"Лучший фитнес: {self.best_fitness:.6f}")
            print(f"Сумма квадратов остатков: {-self.best_fitness:.6e}")
            print(f"\n📊 ЛУЧШИЕ ПАРАМЕТРЫ:")
            for i, param in enumerate(self.best_individual):
                print(f"  Параметр {i}: {param:.6e}")
            print(f"{'='*60}")
        
        return self.best_individual, self.best_fitness
    
    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = []
            for low, high in self.bounds:
                log_val = random.gauss(mu=0.0, sigma=1.0)
                param = 10**log_val
                param = max(min(param, high), low)
                individual.append(param)
            population.append(np.array(individual))
    
        return np.array(population)

    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Турнирная селекция"""
        participants = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = participants[np.argmax(fitness_scores[participants])]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Арифметическое скрещивание"""
        alpha = random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    
    def _mutate(self, individual):
        """Гауссовская мутация в логарифмической шкале"""
        mutated = individual.copy()
        for i in range(self.n_params):
            if random.random() < self.mutation_rate:
                low, high = self.bounds[i]
                
                # Мутация в логарифмической шкале
                log_val = np.log10(max(mutated[i], 1e-10)) #earlier was 1-20 (str 198 also 202, 203)
                log_val += random.gauss(0, 1)  # Случайное смещение
                
                # Применяем границы
                log_low = np.log10(max(low, 1e-10))
                log_high = np.log10(max(high, 1e-10))
                log_val = np.clip(log_val, log_low, log_high)
                
                mutated[i] = 10**log_val
        
        return mutated
'''


class HybridGA:
    """Гибридный генетический алгоритм с локальной оптимизацией"""
    
    def __init__(self, n_params, bounds, pop_size=50, generations=100, 
                 mutation_rate=0.1, elite_count=1, verbose=1,
                 local_steps=10, local_fraction=0.3):
        """
        Инициализация гибридного генетического алгоритма
        
        Параметры:
        ----------
        n_params : int
            Количество оптимизируемых параметров
        bounds : list of tuples
            Границы для каждого параметра [(min, max), ...]
        pop_size : int
            Размер популяции
        generations : int
            Количество поколений
        mutation_rate : float
            Вероятность мутации
        elite_count : int
            Количество лучших особей, переходящих в следующее поколение
        verbose : int
            Уровень детализации вывода
        local_steps : int
            Количество шагов метода наименьших квадратов для локальной оптимизации
        local_fraction : float
            Доля особей, подвергающихся локальной оптимизации
        """
        self.n_params = n_params
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.verbose = verbose
        self.local_steps = local_steps
        self.local_fraction = local_fraction
        
        # История для отслеживания сходимости
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        self.local_optimizations_count = 0
        
        # Лучшее решение
        self.best_individual = None
        self.best_fitness = -np.inf
        
        # Счетчики
        self.function_evaluations = 0
        self.local_optimizations_performed = 0
        
        # Внешняя функция для локальной оптимизации (будет установлена позже)
        self.local_optimization_func = None
        
    def set_local_optimization_func(self, func):
        """Установка функции для локальной оптимизации"""
        self.local_optimization_func = func
    
    def optimize(self, fitness_func):
        """
        Основная функция оптимизации
        
        Параметры:
        ----------
        fitness_func : callable
            Функция оценки фитнеса (чем выше, тем лучше)
            
        Возвращает:
        -----------
        best_individual : np.ndarray
            Лучшая найденная особь
        best_fitness : float
            Фитнес лучшей особи
        """
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print("🧬🚀 ГИБРИДНЫЙ ГЕНЕТИЧЕСКИЙ АЛГОРИТМ С ЛОКАЛЬНОЙ ОПТИМИЗАЦИЕЙ")
            print(f"{'='*70}")
            print(f"Количество параметров: {self.n_params}")
            print(f"Размер популяции: {self.pop_size}")
            print(f"Количество поколений: {self.generations}")
            print(f"Вероятность мутации: {self.mutation_rate}")
            print(f"Количество элитных особей: {self.elite_count}")
            print(f"Локальные шаги (LS): {self.local_steps}")
            print(f"Доля локальной оптимизации: {self.local_fraction*100}%")
            print(f"{'='*70}")
        
        # Сброс счетчиков
        self.function_evaluations = 0
        self.local_optimizations_performed = 0
        start_time = time.time()
        
        # 1. Инициализация популяции
        population = self._initialize_population()
        
        for gen in range(self.generations):
            gen_start_time = time.time()
            
            # 2. Оценка фитнеса
            fitness_scores = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                fitness_scores[i] = fitness_func(population[i])
                self.function_evaluations += 1
            
            # 3. Локальная оптимизация лучших особей
            if self.local_optimization_func and gen % 3 == 0:  # Каждые 3 поколения
                self._apply_local_optimization(population, fitness_scores, fitness_func)
            
            # 4. Статистика
            current_best_fitness = np.max(fitness_scores)
            current_avg_fitness = np.mean(fitness_scores)
            current_worst_fitness = np.min(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            
            # Сохраняем историю
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(current_avg_fitness)
            self.worst_fitness_history.append(current_worst_fitness)
            
            # 5. Обновление лучшего решения
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = population[best_idx].copy()
                
                # Локальная оптимизация лучшей особи
                if self.local_optimization_func:
                    improved = self._optimize_individual(self.best_individual, fitness_func)
                    if improved is not None:
                        self.best_individual = improved
                        self.best_fitness = fitness_func(improved)
                        self.function_evaluations += 1
            
            # 6. Вывод информации о поколении
            gen_time = time.time() - gen_start_time
            
            if self.verbose >= 2:
                print(f"Поколение {gen+1:3d}/{self.generations}: "
                      f"Лучший = {current_best_fitness:10.4f}, "
                      f"Средний = {current_avg_fitness:10.4f}, "
                      f"Лок.опт. = {self.local_optimizations_performed:3d}, "
                      f"Время = {gen_time:.2f}с")
            elif self.verbose >= 1 and gen % 10 == 0:
                print(f"Поколение {gen+1:3d}/{self.generations}: "
                      f"Лучший = {current_best_fitness:10.4f}, "
                      f"Средний = {current_avg_fitness:10.4f}")
            
            # 7. Создание новой популяции с гибридным подходом
            new_population = self._create_next_generation(population, fitness_scores, fitness_func)
            population = np.array(new_population)
        
        # Финальная локальная оптимизация лучшей особи
        if self.local_optimization_func and self.best_individual is not None:
            print("\n" + "="*70)
            print("🔄 ФИНАЛЬНАЯ ЛОКАЛЬНАЯ ОПТИМИЗАЦИЯ ЛУЧШЕЙ ОСОБИ")
            print("="*70)
            improved = self._optimize_individual(self.best_individual, fitness_func, max_iter=50)
            if improved is not None:
                self.best_individual = improved
                self.best_fitness = fitness_func(improved)
                self.function_evaluations += 1
        
        # Завершение оптимизации
        total_time = time.time() - start_time
        
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print("✅ ГИБРИДНАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
            print(f"{'='*70}")
            print(f"Общее время выполнения: {total_time:.2f} секунд")
            print(f"Всего оценок функции: {self.function_evaluations}")
            print(f"Локальных оптимизаций: {self.local_optimizations_performed}")
            print(f"Лучший фитнес: {self.best_fitness:.6f}")
            print(f"Сумма квадратов остатков: {-self.best_fitness:.6e}")
            print(f"\n📊 ЛУЧШИЕ ПАРАМЕТРЫ:")
            for i, param in enumerate(self.best_individual):
                print(f"  Параметр {i}: {param:.6e}")
            print(f"{'='*70}")
        
        return self.best_individual, self.best_fitness
    
    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = []
            for low, high in self.bounds:
                log_val = random.gauss(mu=0.0, sigma=1.0)
                param = 10**log_val
                param = max(min(param, high), low)
                individual.append(param)
            population.append(np.array(individual))
    
        return np.array(population)

    def _create_next_generation(self, population, fitness_scores, fitness_func):
        """Создание нового поколения с турнирами и локальной оптимизацией"""
        new_population = []
        
        # 1. Элитизм
        elite_indices = np.argsort(fitness_scores)[-self.elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # 2. Создаем остальных особей с турнирами и локальной оптимизацией
        while len(new_population) < self.pop_size:
            # Турнирная селекция
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Скрещивание
            child = self._crossover(parent1, parent2)
            
            # Мутация
            child = self._mutate(child)
            
            # С вероятностью local_fraction применяем локальную оптимизацию
            if (self.local_optimization_func and 
                random.random() < self.local_fraction and
                len(new_population) < self.pop_size * 0.8):  # Не применяем к последним 20%
                
                optimized_child = self._optimize_individual(child, fitness_func)
                if optimized_child is not None:
                    child = optimized_child
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Турнирная селекция"""
        participants = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = participants[np.argmax(fitness_scores[participants])]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Арифметическое скрещивание"""
        alpha = random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    
    def _mutate(self, individual):
        """Гауссовская мутация в логарифмической шкале"""
        mutated = individual.copy()
        for i in range(self.n_params):
            if random.random() < self.mutation_rate:
                low, high = self.bounds[i]
                
                # Мутация в логарифмической шкале
                log_val = np.log10(max(mutated[i], 1e-10))
                log_val += random.gauss(0, 1)  # Случайное смещение
                
                # Применяем границы
                log_low = np.log10(max(low, 1e-10))
                log_high = np.log10(max(high, 1e-10))
                log_val = np.clip(log_val, log_low, log_high)
                
                mutated[i] = 10**log_val
        
        return mutated
    
    def _apply_local_optimization(self, population, fitness_scores, fitness_func):
        """Применение локальной оптимизации к лучшим особям"""
        if not self.local_optimization_func:
            return
        
        # Выбираем топ-N особей для локальной оптимизации
        num_to_optimize = max(1, int(self.pop_size * 0.1))  # 10% лучших
        best_indices = np.argsort(fitness_scores)[-num_to_optimize:]
        
        for idx in best_indices:
            individual = population[idx]
            optimized = self._optimize_individual(individual, fitness_func)
            if optimized is not None:
                population[idx] = optimized
                # Обновляем фитнес
                fitness_scores[idx] = fitness_func(optimized)
                self.function_evaluations += 1
    
    def _optimize_individual(self, individual, fitness_func, max_iter=50):
        """
        Локальная оптимизация одной особи методом наименьших квадратов
        
        Параметры:
        ----------
        individual : np.ndarray
            Особь для оптимизации
        fitness_func : callable
            Функция фитнеса
        max_iter : int or None
            Максимальное количество итераций LS
            
        Возвращает:
        -----------
        optimized_individual : np.ndarray or None
            Оптимизированная особь или None при ошибке
        """
        if not self.local_optimization_func:
            return None
        
        try:
            # Преобразуем фитнес-функцию в функцию невязок для least_squares
            def residuals(params):
                fitness = fitness_func(params)
                # Преобразуем фитнес в невязки
                # Для GA: fitness = -sum(residuals^2), поэтому residuals = sqrt(-fitness)
                if fitness >= 0:
                    # Если фитнес положительный, это ошибка
                    return np.ones(self.n_params) * 1e10
                return np.sqrt(-fitness) * np.ones(self.n_params) / self.n_params
            
            # Границы для LS
            bounds_ls = ([b[0] for b in self.bounds], [b[1] for b in self.bounds])
            
            # Настройки LS
            max_nfev = max_iter if max_iter else self.local_steps * 10
            ftol = 1e-6
            xtol = 1e-6
            gtol = 1e-6
            
            # Запускаем LS
            result = least_squares(
                residuals,
                individual,
                bounds=bounds_ls,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=0
            )
            
            if result.success:
                self.local_optimizations_performed += 1
                return result.x
            else:
                # Если LS не сошелся, возвращаем исходную особь
                return individual
                
        except Exception as e:
            if self.verbose >= 2:
                print(f"Ошибка локальной оптимизации: {e}")
            return individual
    
    def plot_convergence(self, show=True, save_path=None):
        """График сходимости гибридного GA"""
        import matplotlib.pyplot as plt
        
        if not self.best_fitness_history:
            print("Нет данных для построения графика")
            return
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        generations = range(1, len(self.best_fitness_history) + 1)
        
        plt.plot(generations, self.best_fitness_history, 'b-', linewidth=2, label='Лучший фитнес')
        plt.plot(generations, self.avg_fitness_history, 'r--', linewidth=1.5, alpha=0.7, label='Средний фитнес')
        plt.plot(generations, self.worst_fitness_history, 'g:', linewidth=1.5, alpha=0.7, label='Худший фитнес')
        
        # Отметки локальных оптимизаций
        if self.local_optimizations_performed > 0:
            plt.axhline(y=np.max(self.best_fitness_history), color='orange', 
                       linestyle='--', alpha=0.5, label='С локальной оптимизацией')
        
        plt.xlabel('Поколение', fontsize=12)
        plt.ylabel('Фитнес (чем выше, тем лучше)', fontsize=12)
        plt.title('Сходимость гибридного генетического алгоритма\n' + 
                 f'Локальных оптимизаций: {self.local_optimizations_performed}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        plt.subplot(1, 2, 2)
        # График улучшения фитнеса
        if len(self.best_fitness_history) > 1:
            improvement = np.diff(self.best_fitness_history)
            plt.bar(range(2, len(self.best_fitness_history) + 1), improvement, 
                   alpha=0.6, color='green', label='Улучшение фитнеса')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Поколение', fontsize=12)
            plt.ylabel('Улучшение фитнеса', fontsize=12)
            plt.title('Улучшение лучшего фитнеса по поколениям', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен как: {save_path}")
        if show:
            plt.show()


import numpy as np
import random
import time
from scipy.optimize import least_squares

class SimpleGA:
    """Генетический алгоритм с интегрированным методом наименьших квадратов"""
    
    def __init__(self, n_params, bounds, pop_size=50, generations=100, 
                 mutation_rate=0.1, elite_count=1, verbose=1,
                 use_local_opt=True, local_opt_fraction=0.3, local_opt_steps=10):
        """
        Инициализация генетического алгоритма с интегрированным LS
        
        Параметры:
        ----------
        n_params : int
            Количество оптимизируемых параметров
        bounds : list of tuples
            Границы для каждого параметра [(min, max), ...]
        pop_size : int
            Размер популяции
        generations : int
            Количество поколений
        mutation_rate : float
            Вероятность мутации
        elite_count : int
            Количество лучших особей, переходящих в следующее поколение
        verbose : int
            Уровень детализации вывода
        use_local_opt : bool
            Использовать ли локальную оптимизацию методом наименьших квадратов
        local_opt_fraction : float
            Доля особей, подвергающихся локальной оптимизации после турнира
        local_opt_steps : int
            Максимальное количество шагов для локальной оптимизации
        """
        self.n_params = n_params
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.verbose = verbose
        self.use_local_opt = use_local_opt
        self.local_opt_fraction = local_opt_fraction
        self.local_opt_steps = local_opt_steps
        
        # История для отслеживания сходимости
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        self.local_opt_counter = 0  # Счетчик локальных оптимизаций
        
        # Лучшее решение
        self.best_individual = None
        self.best_fitness = -np.inf
        
        # Счетчики
        self.function_evaluations = 0
        
        # Внешняя функция для локальной оптимизации (будет установлена позже)
        self.local_opt_func = None
        
    def set_local_opt_func(self, func):
        """Установка функции для локальной оптимизации"""
        self.local_opt_func = func
    
    def optimize(self, fitness_func):
        """
        Основная функция оптимизации
        
        Параметры:
        ----------
        fitness_func : callable
            Функция оценки фитнеса (чем выше, тем лучше)
            
        Возвращает:
        -----------
        best_individual : np.ndarray
            Лучшая найденная особь
        best_fitness : float
            Фитнес лучшей особи
        """
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print("🧬 ГЕНЕТИЧЕСКИЙ АЛГОРИТМ С ИНТЕГРИРОВАННЫМ LS")
            print(f"{'='*60}")
            print(f"Количество параметров: {self.n_params}")
            print(f"Размер популяции: {self.pop_size}")
            print(f"Количество поколений: {self.generations}")
            print(f"Вероятность мутации: {self.mutation_rate}")
            print(f"Количество элитных особей: {self.elite_count}")
            if self.use_local_opt:
                print(f"Локальная оптимизация: ДА ({self.local_opt_fraction*100}% особей, {self.local_opt_steps} шагов)")
            else:
                print(f"Локальная оптимизация: НЕТ")
            print(f"{'='*60}")
        
        # Сброс счетчиков
        self.function_evaluations = 0
        self.local_opt_counter = 0
        start_time = time.time()
        
        # 1. Инициализация популяции
        population = self._initialize_population()
        
        for gen in range(self.generations):
            gen_start_time = time.time()
            
            # 2. Оценка фитнеса
            fitness_scores = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                fitness_scores[i] = fitness_func(population[i])
                self.function_evaluations += 1
            
            # 3. Статистика
            current_best_fitness = np.max(fitness_scores)
            current_avg_fitness = np.mean(fitness_scores)
            current_worst_fitness = np.min(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            
            # Сохраняем историю
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(current_avg_fitness)
            self.worst_fitness_history.append(current_worst_fitness)
            
            # 4. Обновление лучшего решения
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = population[best_idx].copy()
            
            # 5. Вывод информации о поколении
            gen_time = time.time() - gen_start_time
            
            if self.verbose >= 2:
                print(f"Поколение {gen+1:3d}/{self.generations}: "
                      f"Лучший = {current_best_fitness:10.4f}, "
                      f"Средний = {current_avg_fitness:10.4f}, "
                      f"Худший = {current_worst_fitness:10.4f}, "
                      f"Лок.опт. = {self.local_opt_counter:3d}, "
                      f"Время = {gen_time:.2f}с")
            elif self.verbose >= 1 and gen % 10 == 0:
                print(f"Поколение {gen+1:3d}/{self.generations}: "
                      f"Лучший = {current_best_fitness:10.4f}, "
                      f"Средний = {current_avg_fitness:10.4f}")
            
            # 6. Создание новой популяции с интегрированным LS
            new_population = self._create_next_generation_hybrid(
                population, fitness_scores, fitness_func, gen
            )
            population = np.array(new_population)
        
        # Финальная локальная оптимизация лучшей особи
        if self.use_local_opt and self.local_opt_func and self.best_individual is not None:
            if self.verbose >= 1:
                print(f"\n🔄 ФИНАЛЬНАЯ ЛОКАЛЬНАЯ ОПТИМИЗАЦИЯ ЛУЧШЕЙ ОСОБИ")
            
            improved = self._apply_local_optimization(
                self.best_individual, 
                fitness_func,
                max_iter=self.local_opt_steps * 5  # Больше шагов для финальной оптимизации
            )
            
            if improved is not None:
                self.best_individual = improved
                self.best_fitness = fitness_func(improved)
                self.function_evaluations += 1
        
        # Завершение оптимизации
        total_time = time.time() - start_time
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print("✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
            print(f"{'='*60}")
            print(f"Общее время выполнения: {total_time:.2f} секунд")
            print(f"Всего оценок функции: {self.function_evaluations}")
            if self.use_local_opt:
                print(f"Локальных оптимизаций LS: {self.local_opt_counter}")
            print(f"Лучший фитнес: {self.best_fitness:.6f}")
            print(f"Сумма квадратов остатков: {-self.best_fitness:.6e}")
            print(f"\n📊 ЛУЧШИЕ ПАРАМЕТРЫ:")
            for i, param in enumerate(self.best_individual):
                print(f"  Параметр {i}: {param:.6e}")
            print(f"{'='*60}")
        
        return self.best_individual, self.best_fitness
    
    def _initialize_population(self):
        """Инициализация популяции в логарифмической шкале"""
        population = []
        for _ in range(self.pop_size):
            individual = []
            for low, high in self.bounds:
                log_val = random.gauss(mu=0.0, sigma=1.0)
                param = 10**log_val
                param = max(min(param, high), low)
                individual.append(param)
            population.append(np.array(individual))
    
        return np.array(population)
    
    
    def _create_next_generation_hybrid(self, population, fitness_scores, fitness_func, generation):
        """
        Создание нового поколения с интегрированным LS
        
        Ключевое изменение: после каждого турнира с вероятностью local_opt_fraction
        применяем метод наименьших квадратов к полученной особи
        """
        new_population = []
        
        # 1. Элитизм
        elite_indices = np.argsort(fitness_scores)[-self.elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # 2. Создаем остальных особей
        while len(new_population) < self.pop_size:
            # Турнирная селекция
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Скрещивание
            child = self._crossover(parent1, parent2)
            
            # Мутация
            child = self._mutate(child)
            
            # ИНТЕГРИРОВАННЫЙ LS: После турнира применяем локальную оптимизацию
            if (self.use_local_opt and 
                self.local_opt_func and 
                random.random() < self.local_opt_fraction and
                generation > 0):  # Начинаем со второго поколения
                
                # Адаптивное количество шагов LS
                ls_steps = self._get_adaptive_ls_steps(generation)
                
                # Применяем локальную оптимизацию
                improved_child = self._apply_local_optimization(child, fitness_func, max_iter=ls_steps)
                
                if improved_child is not None:
                    child = improved_child
            
            new_population.append(child)
        
        return new_population
    

    '''''''''
    def _create_next_generation_hybrid(self, population, fitness_scores, fitness_func, generation):

        new_population = []
        
        # 1. Элитизм
        elite_indices = np.argsort(fitness_scores)[-self.elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # ПОДГОТОВКА РУЛЕТКИ
        # Преобразуем фитнес в вероятности (как в вашем примере)
        min_fitness = np.min(fitness_scores)
        if min_fitness < 0:
            fitness_positive = fitness_scores - min_fitness + 1e-10
        else:
            fitness_positive = fitness_scores + 1e-10
        
        # Строим колесо рулетки (накопленные вероятности)
        total_fitness = np.sum(fitness_positive)
        cumulative = np.cumsum(fitness_positive / total_fitness)
        
        # 2. Создаем остальных особей
        while len(new_population) < self.pop_size:
            # РУЛЕТОЧНАЯ СЕЛЕКЦИЯ (вместо турнирной)
            # Выбор первого родителя
            r1 = random.random()
            idx1 = np.searchsorted(cumulative, r1)
            parent1 = population[idx1].copy()
            
            # Выбор второго родителя
            r2 = random.random()
            idx2 = np.searchsorted(cumulative, r2)
            parent2 = population[idx2].copy()
            
            # Скрещивание
            child = self._crossover(parent1, parent2)
            
            # Мутация
            child = self._mutate(child)
            
            # ИНТЕГРИРОВАННЫЙ LS
            if (self.use_local_opt and 
                self.local_opt_func and 
                random.random() < self.local_opt_fraction and
                generation > 0):
                
                ls_steps = self._get_adaptive_ls_steps(generation)
                improved_child = self._apply_local_optimization(child, fitness_func, max_iter=ls_steps)
                if improved_child is not None:
                    child = improved_child
            
            new_population.append(child)
        
        return new_population
    '''''''''
        
    def _get_adaptive_ls_steps(self, generation):
        """Адаптивное количество шагов LS в зависимости от поколения"""
        # В начале эволюции меньше шагов LS, в конце - больше
        if generation < self.generations * 0.3:  # Первые 30%
            return max(5, self.local_opt_steps // 2)
        elif generation < self.generations * 0.7:  # Следующие 40%
            return self.local_opt_steps
        else:  # Последние 30%
            return min(20, self.local_opt_steps * 2)
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Турнирная селекция"""
        participants = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = participants[np.argmax(fitness_scores[participants])]
        return population[winner_idx].copy()
    
    """""""""
    def _roulette_selection(self, fitness_scores, num_parents):
        min_fitness = np.min(fitness_scores)
        if min_fitness < 0:
            fitness_positive = fitness_scores - min_fitness + 1e-10
        else:
            fitness_positive = fitness_scores + 1e-10
        
        total = np.sum(fitness_positive)
        cumulative = []  # накопленные вероятности
        cumsum = 0
        for f in fitness_positive:
            cumsum += f / total
            cumulative.append(cumsum)
        
        selected_indices = []
        for _ in range(num_parents):
            r = random.random()
            idx = np.searchsorted(cumulative, r)
            selected_indices.append(idx)
        
        return selected_indices
    """""""""
    '''''''''        
    def _crossover(self, parent1, parent2):
        """Арифметическое скрещивание"""
        alpha = random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    '''''''''    
    def _crossover(self, parent1, parent2):
        k = random.randint(1, self.n_params-1)

        child = np.zeros_like(parent1)

        child[:k] = parent1[:k]
        child[k:] = parent2[k:]

        return child

    def _mutate(self, individual):
        """Гауссовская мутация в логарифмической шкале"""
        mutated = individual.copy()
        for i in range(self.n_params):
            if random.random() < self.mutation_rate:
                low, high = self.bounds[i]
                
                # Мутация в логарифмической шкале
                log_val = np.log10(max(mutated[i], 1e-20))
                log_val += random.gauss(0, 1)  # Случайное смещение
                
                # Применяем границы
                log_low = np.log10(max(low, 1e-20))
                log_high = np.log10(max(high, 1e-20))
                log_val = np.clip(log_val, log_low, log_high)
                
                mutated[i] = 10**log_val
        
        return mutated
    
    def _apply_local_optimization(self, individual, fitness_func, max_iter=50):
        """
        Применение метода наименьших квадратов для локальной оптимизации особи
        
        Параметры:
        ----------
        individual : np.ndarray
            Особь для оптимизации
        fitness_func : callable
            Функция фитнеса (для вычисления невязок)
        max_iter : int or None
            Максимальное количество итераций LS
            
        Возвращает:
        -----------
        optimized_individual : np.ndarray or None
            Оптимизированная особь или None при ошибке
        """
        if not self.local_opt_func:
            return None
        
        try:
            # Используем функцию невязок для LS
            def residuals_for_ls(params):
                # Вызываем внешнюю функцию невязок
                return self.local_opt_func(params)
            
            # Границы для LS
            bounds_ls = ([b[0] for b in self.bounds], [b[1] for b in self.bounds])
            
            # Настройки LS
            max_nfev = max_iter if max_iter else self.local_opt_steps * 10
            
            # Запускаем LS с ограниченным числом итераций
            result = least_squares(
                residuals_for_ls,
                individual,
                bounds=bounds_ls,
                max_nfev=max_nfev,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
                verbose=0,  # Без вывода
                loss='linear'
            )
            print(result.cost)
            self.local_opt_counter += 1
            
            if result.success or result.nfev >= max_nfev:
                # Возвращаем улучшенную особь даже если не сошлось до конца
                return result.x
            else:
                # Если LS полностью провалился, возвращаем исходную особь
                return individual
                
        except Exception as e:
            if self.verbose >= 3:
                print(f"Ошибка локальной оптимизации: {e}")
            return individual
    
    def plot_convergence(self, show=True, save_path=None):
        """График сходимости GA с интегрированным LS"""
        import matplotlib.pyplot as plt
        
        if not self.best_fitness_history:
            print("Нет данных для построения графика")
            return
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        generations = range(1, len(self.best_fitness_history) + 1)
        
        plt.plot(generations, self.best_fitness_history, 'b-', linewidth=2, label='Лучший фитнес')
        plt.plot(generations, self.avg_fitness_history, 'r--', linewidth=1.5, alpha=0.7, label='Средний фитнес')
        
        if self.use_local_opt:
            # Отметки где применялся LS
            for gen in range(1, len(self.best_fitness_history)):
                if gen % 5 == 0:  # Примерно каждые 5 поколений
                    plt.scatter(gen, self.best_fitness_history[gen-1], 
                               color='green', s=20, alpha=0.5, zorder=5)
            
            plt.scatter([], [], color='green', s=20, alpha=0.5, 
                       label=f'LS применен ({self.local_opt_counter} раз)')
        
        plt.xlabel('Поколение', fontsize=12)
        plt.ylabel('Фитнес (чем выше, тем лучше)', fontsize=12)
        plt.title('Сходимость генетического алгоритма', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        plt.subplot(1, 2, 2)
        # График скорости сходимости
        if len(self.best_fitness_history) > 1:
            improvements = []
            for i in range(1, len(self.best_fitness_history)):
                if self.best_fitness_history[i] > self.best_fitness_history[i-1]:
                    improvements.append(self.best_fitness_history[i] - self.best_fitness_history[i-1])
                else:
                    improvements.append(0)
            
            plt.bar(range(1, len(self.best_fitness_history)), improvements, 
                   alpha=0.6, color='blue', label='Улучшение')
            
            if self.use_local_opt and self.local_opt_counter > 0:
                # Среднее улучшение после LS
                avg_improvement = np.mean([imp for imp in improvements if imp > 0])
                plt.axhline(y=avg_improvement, color='red', linestyle='--', 
                           label=f'Ср. улучшение: {avg_improvement:.4f}')
            
            plt.xlabel('Поколение', fontsize=12)
            plt.ylabel('Улучшение фитнеса', fontsize=12)
            plt.title('Динамика улучшений', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
        
        plt.suptitle(f"Генетический алгоритм {'с' if self.use_local_opt else 'без'} интегрированным LS\n"
                    f"Локальных оптимизаций: {self.local_opt_counter}", 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose >= 1:
                print(f"График сохранен как: {save_path}")
        if show:
            plt.show()