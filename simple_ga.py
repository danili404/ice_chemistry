import numpy as np
import random
import time

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
        """Создание начальной популяции"""
        population = []
        for _ in range(self.pop_size):
            individual = []
            for low, high in self.bounds:
                # Генерация в логарифмической шкале для кинетических констант
                log_low = np.log10(max(low, 1e-20))
                log_high = np.log10(max(high, 1e-20))
                log_val = np.random.uniform(log_low, log_high)
                individual.append(10**log_val)
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
                log_val = np.log10(max(mutated[i], 1e-20))
                log_val += random.gauss(0, 0.5)  # Случайное смещение
                
                # Применяем границы
                log_low = np.log10(max(low, 1e-20))
                log_high = np.log10(max(high, 1e-20))
                log_val = np.clip(log_val, log_low, log_high)
                
                mutated[i] = 10**log_val
        
        return mutated