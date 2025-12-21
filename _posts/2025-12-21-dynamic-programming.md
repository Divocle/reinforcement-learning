---
layout: post
title: "强化学习基础：状态价值函数与动作价值函数的定义与关系"
date: 2025-12-21  # 新增：Jekyll必需的日期字段，确保文章被正确识别
---
# 强化学习基础：状态价值函数与动作价值函数的定义与关系

> **摘要**：本文系统梳理强化学习中两个核心概念——状态价值函数 $v_\pi(s)$ 与动作价值函数 $q_\pi(s, a)$ 的数学定义、直观含义及其相互关系，并通过贝尔曼方程揭示其递归结构，为理解策略评估与优化奠定理论基础。

---

## 1. 背景：马尔可夫决策过程（MDP）

强化学习通常建模为 **马尔可夫决策过程**（Markov Decision Process, MDP），记为五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$：

- $\mathcal{S}$：状态空间  
- $\mathcal{A}$：动作空间  
- $P(s' \mid s, a)$：状态转移概率，即在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率  
- $R(s, a) = \mathbb{E}[r_{t+1} \mid s_t = s, a_t = a]$：期望即时奖励  
- $\gamma \in [0, 1]$：折扣因子，衡量未来奖励的重要性  

智能体遵循一个 **策略**（policy）$\pi(a \mid s)$，表示在状态 $s$ 下选择动作 $a$ 的概率。

目标是最大化从任意时刻 $t$ 开始的 **折扣回报**（discounted return）：
$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

---

## 2. 状态价值函数 $v_\pi(s)$

### 2.1 定义

状态价值函数衡量在策略 $\pi$ 下，从状态 $s$ 出发所能获得的**期望折扣回报**：

$$
v_\pi(s) \triangleq \mathbb{E}_\pi \left[ G_t \,\big|\, S_t = s \right]
= \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\bigg|\, S_t = s \right]
$$

> **关键点**：$v_\pi(s)$ 不指定具体动作，而是对策略 $\pi$ 在该状态下所有可能行为路径的平均表现。

---

## 3. 动作价值函数 $q_\pi(s, a)$

### 3.1 定义

动作价值函数衡量在策略 $\pi$ 下，从状态 $s$ **先执行动作 $a$**，然后继续遵循 $\pi$ 所能获得的期望折扣回报：

$$
q_\pi(s, a) \triangleq \mathbb{E}_\pi \left[ G_t \,\big|\, S_t = s, A_t = a \right]
= \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\bigg|\, S_t = s, A_t = a \right]
$$

> **关键点**：$q_\pi(s, a)$ 显式指定了第一个动作，因此可用于**比较不同动作的优劣**，是策略改进的关键工具。

---

## 4. 两者的核心关系

### 4.1 状态价值是动作价值的期望（按策略加权）

由于策略 $\pi(a \mid s)$ 决定了在状态 $s$ 下选择各动作的概率，状态价值可表示为动作价值的加权平均：

$$
\boxed{
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, q_\pi(s, a)
}
\tag{1}
$$

**推导**：  
由全期望公式（Law of Total Expectation）：

$$
\begin{aligned}
v_\pi(s) 
&= \mathbb{E}_\pi[G_t \mid S_t = s] \\
&= \sum_{a} \pi(a \mid s) \cdot \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] \\
&= \sum_{a} \pi(a \mid s) \, q_\pi(s, a)
\end{aligned}
$$

此式表明：**知道所有 $q_\pi(s, a)$ 和策略 $\pi$，即可计算 $v_\pi(s)$**。

---

### 4.2 动作价值可由状态价值递归表达（贝尔曼方程）

执行动作 $a$ 后，环境给出即时奖励 $r$ 并转移到新状态 $s'$。后续回报由 $v_\pi(s')$ 描述。因此：

$$
\boxed{
q_\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \, v_\pi(s')
}
\tag{2}
$$

**推导**： 

$$
\begin{aligned}
q_\pi(s, a) 
&= \mathbb{E}_\pi \left[ r_{t+1} + \gamma G_{t+1} \,\big|\, s_t = s, a_t = a \right] \\
&= \underbrace{\mathbb{E}[r_{t+1} \mid s, a]}_{R(s,a)} 
+ \gamma \, \mathbb{E}_\pi \left[ G_{t+1} \,\big|\, s_t = s, a_t = a \right] \\
&= R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \cdot \mathbb{E}_\pi[G_{t+1} \mid s_{t+1} = s'] \\
&= R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, v_\pi(s')
\end{aligned}
$$

此式是 **贝尔曼期望方程**（Bellman Expectation Equation）的核心形式之一。

---

### 4.3 联立得状态价值的贝尔曼方程

将式 (2) 代入式 (1)，得到经典的 **状态价值贝尔曼方程**：

$$
\begin{aligned}
v_\pi(s) 
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \, v_\pi(s') \right] \\
&= \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ R(s, a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot|s,a)} [v_\pi(s')] \right]
\end{aligned}
\tag{3}
$$

该方程是 **动态规划**（Dynamic Programming）中策略评估（Policy Evaluation）算法的基础。

---

## 5. 直观类比与应用场景

| 概念 | 生活类比 |
|------|--------|
| $v_\pi(s)$ | “我在北京生活，按我的日常习惯（策略），平均幸福感是多少？” |
| $q_\pi(s, a)$ | “如果我现在在北京选择‘去爬香山’这个动作，预期会有多开心？” |

### 应用意义：

- **策略评估**：通过式 (3) 迭代计算 $v_\pi(s)$  
- **策略改进**：利用 $q_\pi(s, a)$ 构造更优策略：
  $$
  \pi'(s) = \arg\max_{a} q_\pi(s, a)
  $$
- **最优性条件**：当对所有 $s \in \mathcal{S}$ 有
  $$
  v_\pi(s) = \max_{a \in \mathcal{A}} q_\pi(s, a)
  $$
  则 $\pi$ 为最优策略 $\pi^*$，对应 **贝尔曼最优方程**。

---

## 6. 总结

| 项目 | 状态价值函数 $v_\pi(s)$ | 动作价值函数 $q_\pi(s, a)$ |
|------|------------------------|--------------------------|
| **输入** | 状态 $s$ | 状态 $s$ + 动作 $a$ |
| **含义** | 状态的整体好坏 | 状态-动作对的好坏 |
| **关系** | $v_\pi(s) = \mathbb{E}_{a \sim \pi}[q_\pi(s, a)]$ | $q_\pi(s, a) = R(s, a) + \gamma \mathbb{E}_{s'}[v_\pi(s')]$ |
| **用途** | 评估策略整体性能 | 比较动作、改进策略 |

---

> **参考文献**  
> - Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.  
> - Bertsekas, D. P. (2019). *Reinforcement Learning and Optimal Control*. Athena Scientific.