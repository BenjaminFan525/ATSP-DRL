import numpy as np
import random
import copy


class GeneticAlgorithm():
    def __init__(self, env, population_size=50, generations=20, mutation_rate=0.1):
        # 初始化环境
        self.env = env
        self.planes_num = len(env.planes)
        self.sites_num = len(env.sites)

        # 遗传算法参数
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        """初始化满足资源约束的种群"""
        population = []
        for _ in range(self.population_size):
            chromosome = []
            available_sites = list(range(self.sites_num))  # 可用的站位ID（0~20）
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

    def select(self, population, fitnesses):
        """锦标赛选择"""
        tournament_size = 3
        candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
        return max(candidates, key=lambda x: x[1])[0]  # 返回适应度最高的染色体

    def crossover(self, parent1, parent2):
        """单点交叉"""
        point = random.randint(1, self.planes_num - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        child1 = self.repair_chromosome(child1)
        child2 = self.repair_chromosome(child2)
        return child1, child2

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
    
    def mutate(self, chromosome):
        """变异"""
        for i in range(self.planes_num):
            if random.random() < self.mutation_rate:
                plane_id = i
                avail_actions = self.env.get_avail_agent_actions(plane_id)
                if avail_actions[-2] == 1:  # 正忙
                    chromosome[i] = self.sites_num+1
                elif avail_actions[-1] == 1:  # 已完成
                    chromosome[i] = self.sites_num+2
                else:
                    candidate_actions = [a for a in range(self.sites_num) if avail_actions[a] == 1]
                    if candidate_actions:
                        new_action = random.choice(candidate_actions)
                        # 确保站位不重复
                        if new_action not in chromosome:
                            chromosome[i] = new_action
                        else:
                            chromosome[i] = self.sites_num
                    else:
                        chromosome[i] = self.sites_num
        return chromosome
    
    def run(self):
        """运行遗传算法"""
        population = self.initialize_population()
        for _ in range(self.generations):
            fitnesses = [self.fitness(chrom) for chrom in population]
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.select(population, fitnesses)
                parent2 = self.select(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population
        fitnesses = [self.fitness(chrom) for chrom in population]
        best_idx = np.argmax(fitnesses)
        return population[best_idx]  # 返回最佳染色体
                    