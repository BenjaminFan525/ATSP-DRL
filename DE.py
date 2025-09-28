import numpy as np
import random
import copy


class DifferentialEvolution():
    def __init__(self, env, population_size=50, generations=20, F=0.5, Cr=0.8):
        # 初始化环境
        self.env = env
        self.planes_num = len(env.planes)
        self.sites_num = len(env.sites)

        # 参数
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.Cr = Cr

    def initialize_population(self):
        """初始化满足资源约束的种群"""
        population = []
        for _ in range(self.population_size):
            chromosome = []
            available_sites = list(range(21))  # 可用的站位ID（0~20）
            used_sites = set()  # 记录已使用的站位ID
            for plane_id in range(self.planes_num):
                avail_actions = self.env.get_avail_agent_actions(plane_id)
                if avail_actions[-2] == 1:  # 动作为22（正忙）
                    chromosome.append(self.sites_num+1)
                elif avail_actions[-1] == 1:  # 动作为23（已完成）
                    chromosome.append(self.sites_num+2)
                else:
                    # 优先选择可用站位
                    avail_actions = [a for a in range(self.sites_num) if avail_actions[a] == 1 and a in available_sites and a not in used_sites]
                    if avail_actions:
                        action = random.choice(avail_actions)
                        used_sites.add(action)  # 移除已选站位
                    else:
                        action = self.sites_num  # 无可用站位，选择等待
                    chromosome.append(action)
            population.append(chromosome)
        return population

    def fitness(self, chromosome):
        """计算适应度"""
        env_copy = copy.deepcopy(self.env)  # 复制环境以模拟
        for i, action in enumerate(chromosome):
                if action < len(self.env.sites):
                        env_copy.has_chosen_action(action, i)
        reward, done, info = env_copy.step(chromosome)
        return reward

    def mutation(self, population):
        mutated_pop = []
        for i in range(self.population_size):
            # 随机选择三个不同个体
            candidates = [x for x in range(self.population_size) if x != i]
            a, b, c = random.sample(candidates, 3)
            
            # 生成变异向量（离散空间处理）
            mutant = []
            for j in range(self.planes_num):
                # 离散差分操作：随机交换或基于概率选择
                if random.random() < self.F:
                    mutant.append(population[c][j])
                else:
                    new_val = population[a][j] + (population[b][j] - population[c][j])
                    new_val = max(0, min(self.sites_num+2, new_val))  # 约束到动作范围
                    mutant.append(int(new_val))
            mutated_pop.append(mutant)
        return mutated_pop

    def crossover(self, target, mutant):
        trial = []
        for j in range(self.planes_num):
            if target[j] in [self.sites_num, self.sites_num+1, self.sites_num+2]:
                trial.append(target[j])  # 保持特殊动作不变
            else:
                # 二项式交叉
                if random.random() < self.Cr or j == random.randint(0, self.planes_num-1):
                    trial.append(mutant[j])
                else:
                    trial.append(target[j])
        return self.repair_chromosome(trial)  # 必须修复染色体

    def repair_chromosome(self, chromosome):
        """修复染色体，确保每个动作都是可用的且站位不重复。"""
        used_sites = set()  # 记录已使用的站位
        for i, action in enumerate(chromosome):
            plane_id = i  # 假设染色体索引对应飞机ID
            # 获取该飞机的可用动作
            avail_actions = self.env.get_avail_agent_actions(plane_id)
            candidate_actions = [a for a in range(self.sites_num) if avail_actions[a] == 1]
            if action in candidate_actions:
                # 如果动作可用，检查站位是否重复
                if action < len(self.env.sites):  # 普通站位动作
                    if action not in used_sites:
                        used_sites.add(action)  # 站位未使用，记录
                    else:
                        # 站位重复，选择一个未使用的可用站位
                        available_sites = [a for a in candidate_actions if a not in used_sites]
                        if available_sites:
                            chromosome[i] = random.choice(available_sites)
                            used_sites.add(chromosome[i])
                        else:
                            chromosome[i] = len(self.env.sites)  # 无可用站位，选择等待
                # 特殊动作（如等待、正忙、完成）无需检查重复
            else:
                # 动作不可用，选择一个可用的动作
                if self.sites_num+1 in candidate_actions:  # 飞机正忙
                    chromosome[i] = self.sites_num+1
                elif self.sites_num+2 in candidate_actions:  # 飞机已完成
                    chromosome[i] = self.sites_num+2
                else:
                    # 从可用站位中选择未使用的站位
                    available_sites = [a for a in candidate_actions if a not in used_sites]
                    if available_sites:
                        chromosome[i] = random.choice(available_sites)
                        used_sites.add(chromosome[i])
                    else:
                        chromosome[i] = len(self.env.sites)  # 无可用站位，选择等待
        return chromosome
    
    
    def run(self):
        population = self.initialize_population()
        best_solution = None
        best_fitness = -np.inf
        # 初始化全局最优
        for chrom in population:
            current_fitness = self.fitness(chrom)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = copy.deepcopy(chrom)
        
        for _ in range(self.generations):
            # 变异生成中间种群
            mutated_pop = self.mutation(population)
            
            # 交叉与选择
            new_population = []
            for i in range(self.population_size):
                target = population[i]
                trial = self.crossover(target, mutated_pop[i])
                
                target_fitness = self.fitness(target)
                trial_fitness = self.fitness(trial)
                
                # 更新全局最优
                if target_fitness > best_fitness:
                    best_fitness = target_fitness
                    best_solution = copy.deepcopy(target)
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_solution = copy.deepcopy(trial)
                
                # 贪婪选择
                if trial_fitness > target_fitness:
                    new_population.append(trial)
                else:
                    new_population.append(target)
            population = new_population
        
        return best_solution
                    