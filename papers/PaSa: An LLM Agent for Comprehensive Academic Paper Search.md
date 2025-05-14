[PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/html/2501.10120?_immersive_translate_auto_translate=1) 提出了一个基于大语言模型（LLM）的学术文献搜索代理系统 **PaSa**，用以解决复杂学术查询中信息检索的难题。以下是对论文的详细解读：

---

## **论文背景与动机**

学术论文搜索是科研工作的核心，但它也面临诸多挑战：
- **长尾需求**：许多学术问题非常专业，现有工具（如 Google Scholar）难以满足。
- **复杂查询**：研究者需要的不仅是检索结果，还需要深入的文献调研。
- **时间消耗**：调研复杂问题常常需要耗费大量时间。

虽然大语言模型（LLMs）已经在信息检索和查询优化上取得了一些进展，但这些方法仅限于“简单检索”，而无法进行类似人类的深入文献分析。因此，论文提出了 PaSa（Paper Search Agent），一个能够自动化完成复杂学术文献搜索任务的智能代理系统。

---

## **核心贡献**

1. **提出了 PaSa 系统**：
   - PaSa 由两个 LLM 代理组成：**Crawler**（爬取器）和 **Selector**（选择器）。
   - 模拟人类学术调研行为，能够自主检索、阅读论文并通过引用网络扩展文献范围。
   
2. **构建了两个高质量数据集**：
   - **AutoScholarQuery**: 一个合成学术搜索数据集，包含 35,511 条精细化学术查询及对应文献。
   - **RealScholarQuery**: 一个基于真实复杂学术查询的测试集，包含 50 个实际问题及人工标注的答案。

3. **提出新方法训练和优化 PaSa**：
   - 使用强化学习（RL）框架 AGILE 对 Crawler 和 Selector 进行联合优化。
   - 设计了**会话级别 PPO 算法**，解决论文搜索任务中稀疏奖励和长轨迹问题。

4. **实验结果表明 PaSa 性能显著优于传统方法**：
   - 在 AutoScholarQuery 和 RealScholarQuery 上均显著超过 Google Scholar、GPT-4o 以及基于 Google 的多种检索方法。

---

## **PaSa 系统架构**

PaSa 系统的架构如论文中 **Figure 1** 和 **Figure 2** 所示，具体流程如下：

**Figure 1**
![image](https://github.com/user-attachments/assets/0979eb1f-f7ac-41d3-bac1-a5c16e547ac6)

**Figure 2**
![image](https://github.com/user-attachments/assets/5c0db16e-8f85-452b-918a-1f2dba2e36a4)

### **1. 两个 LLM 代理**
- **Crawler（爬取器）**：
  - 负责接收用户查询，生成搜索关键词，调用搜索工具（如 Google），获取相关论文。
  - 通过分析论文引用网络，进一步扩展文献范围。
  - 核心目标：**最大化召回率**（Recall）。
  
- **Selector（选择器）**：
  - 对 Crawler 收集的论文逐一进行阅读和分析，判断是否满足用户查询需求。
  - 核心目标：**提升精确率**（Precision）。

### **2. 工作流程**
- 用户输入学术查询，Crawler 开始搜索，生成多个搜索关键词并调用搜索工具。
- Crawler 收集初始论文后，通过论文的引用网络进一步扩展论文队列。
- Selector 对所有论文进行评估，筛选出最符合查询需求的论文。
- 输出最终的检索结果。

### **3. 强化学习优化**
- **Crawler 的 MDP（马尔可夫决策过程）建模**：
  - 状态：当前上下文和论文队列。
  - 动作：搜索、引用扩展、停止处理等操作。
  - 奖励：找到相关论文的奖励扣除动作成本。
  
- **训练细节**：
  - **模仿学习**：通过 GPT-4o 生成初始训练数据，进行模仿学习。
  - **强化学习（PPO）**：基于 AutoScholarQuery 数据集，设计稀疏奖励函数，并引入会话级别的 PPO 算法，优化 Crawler。

---


### Reward Design 详解

Reward Design 位于论文的 4.2 节（第5页），主要描述了 PaSa 系统中 **Crawler** 组件在强化学习训练时奖励函数的设计。

#### 1. 背景

- **Crawler** 的任务是自动搜索与用户查询相关的论文，通过调用搜索工具或扩展引用来最大化候选论文的召回率。
- 在强化学习（RL）框架中，Crawler 执行的每一步（如搜索、扩展引用、停止）都被建模为一个 **Markov Decision Process (MDP)**，即每个动作都会影响后续状态和奖励。

#### 2. 基础奖励函数

Crawler 的奖励函数如下：

$$
\
r(s_t, a_t) = \alpha \times \sum_{i=1}^{n_t} I(q, p_i, t) - c(a_t)
\
$$

- $$\( s_t \)$$：当前状态（包括上下文和论文队列）
- $$\( a_t \)$$：当前动作（如 [Search]、[Expand]、[Stop]）
- $$\( n_t \)$$：当前动作带来的新增论文数
- $$\( p_i \)$$：新增论文
- $$\( q \)$$：用户查询
- $$\( \alpha \)$$：奖励系数（调节奖励强度）
- $$\( c(a_t) \)$$：执行动作的成本（如 [Search]、[Expand] 动作有轻微惩罚，防止无意义扩展）
- $$\( I(q, p_i, t) \)$$：指标函数，判断新增论文是否“有用”

指标函数的基础定义：

$$
\
I(q, p_i, t) = 
\begin{cases}
1 & \text{如果 } p_i \text{ 符合查询且不在队列中} \\
0 & \text{否则}
\end{cases}
\
$$

但是！**AutoScholarQuery** 数据集中的答案集合仅是相关论文的一个子集（因为只收录了被引用的部分），直接用它判定会造成奖励稀疏和不准确。

#### 3. 稀疏奖励问题与辅助奖励模型

为了解决奖励稀疏（Sparse Reward）的问题，作者引入了 **Selector** 作为辅助奖励模型：

- Selector 是独立训练的 LLM，用于判断某篇论文是否与查询匹配。
- 于是，指标函数 $$\( I(q, p_i, t) \)$$ 被改进为：

$$
\
I(q, p_i, t) = 
\begin{cases}
1, & \text{如果 Selector}(q, p_i) = 1 \text{ 或 } p_i \in P \\
& \text{且 } p_i \notin Q_t \\
0, & \text{否则}
\end{cases}
\
$$

其中：
- $$\( P \)$$：AutoScholarQuery 中该查询的答案论文集合
- $$\( Q_t \)$$：当前已经在队列中的论文集合
- $$\( Selector(q, $p_i$) = 1 \)$$：Selector 判断该论文满足查询

**这样做的好处：**
- 只要 Selector 判断论文是相关的，就能获得奖励，不再仅依赖稀疏的标注答案。
- 有效缓解了 RL 训练中的奖励稀疏性问题，使 Crawler 在训练中获得更多正向反馈，能探索更多有潜力的论文。

#### 4. 动作成本

- [Search]、[Expand] 动作都有小的负奖励（如 0.1），[Stop] 没有惩罚。这样能防止模型无脑扩展，提高检索效率。

#### 5. 奖励设计小结

- 奖励=新发现的相关论文数×奖励系数-动作成本
- 相关的判定由“人工标注答案+Selector辅助判定”共同决定
- 有效解决了论文检索 RL 任务中的稀疏奖励和长轨迹问题

---


## **数据集构建**

### **1. AutoScholarQuery**
- 来源：从顶级 AI 会议（如 ICLR、ICML、NeurIPS 等）的论文中提取相关文献引用，生成学术查询和答案对。
- 数据规模：
  - 训练集：33,511 条查询-文献对。
  - 开发集和测试集：各 1,000 条。
- 数据质量：通过人工评估，94.0% 的查询和 93.7% 的文献答案与查询高度相关。

### **2. RealScholarQuery**
- 来源：从真实用户（AI 研究者）提供的学术查询中随机抽样，经过人工过滤后生成。
- 数据构建：结合多种检索方法（Google、GPT-4o、PaSa 等）生成候选文献，人工标注筛选最终答案。
- 数据规模：包含 50 个真实学术查询及其标注的答案文献。

---

## **实验与结果分析**

### **1. 实验设置**
- **对比方法**：
  - Google、Google Scholar、Google 配合 GPT-4o。
  - ChatGPT（搜索增强版 GPT-4o）。
  - 基线 LLM：GPT-o1 和 PaSa-GPT-4o。
- **评价指标**：
  - **Recall@k**：检索结果前 k 篇文献的召回率。
  - **Precision & Recall**：最终检索结果的精确率与召回率。

### **2. 实验结果**
#### **AutoScholarQuery 测试集**
- PaSa-7b 超越所有基线方法，提升显著：
  - 相比 Google with GPT-4o，在 Recall@20 上提升 **33.80%**，Recall@50 提升 **38.83%**。
  - 相比 PaSa-GPT-4o，Recall 提升 **9.64%**，精确率相当。

#### **RealScholarQuery 测试集**
- 在真实场景中，PaSa-7b 的表现更为突出：
  - 相比 Google with GPT-4o，在 Recall@20 上提升 **37.78%**，Recall@50 提升 **39.90%**。
  - 相比 PaSa-GPT-4o，Recall 提升 **30.36%**，精确率提升 **4.25%**。

### **3. 消融研究**
- 去掉引用网络扩展（[Expand]）：召回率大幅下降（AutoScholarQuery 降低 22.98%，RealScholarQuery 降低 32.21%）。
- 去掉强化学习：召回率下降（AutoScholarQuery 降低 6.24%，RealScholarQuery 降低 19.96%）。
- 不使用 Selector 作为奖励模型：召回率下降（AutoScholarQuery 降低 3.76%，RealScholarQuery 降低 9.63%）。

---

## **结论与意义**

该论文提出的 PaSa 系统在复杂学术检索任务中表现出色，显著提升了检索效率和结果准确性。其主要特点包括：
1. 模拟人类行为，结合引用网络扩展和深度文献分析。
2. 依靠强化学习优化，尤其是会话级 PPO 设计。
3. 尽管仅使用合成数据训练，但 PaSa 在真实场景中的性能依然优异，显著超过 Google Scholar 和 GPT-4o。

未来的研究可以进一步扩展 PaSa 的适用领域（如跨学科检索）以及提升模型对未见领域的泛化能力。

--- 
