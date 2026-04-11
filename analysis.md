我会直接把可用的 `planning.md` 成稿写出来（含你要的数据集详细描述），并同步给你一个可直接落地的版本。先读取当前会话里的计划草稿后输出最终文档。

Read memory [](file:///c%3A/Users/CosmosMount/AppData/Roaming/Code/User/workspaceStorage/4cd4a8da65ea99be1762985781bf5661/GitHub.copilot-chat/memory-tool/memories/NmE1YTg2NTEtMWQzYi00NTJlLThiNGQtMDNkNDM2MGU0ZjE4/plan.md)

# DSAA2011 Project Planning

## 1. 项目目标
本项目围绕学生学业状态预测展开，目标是构建一个可解释、可复现、评估完整的三分类模型，预测学生最终状态为 Dropout、Graduate 或 Enrolled。  
项目同时包含无监督分析，用聚类与低维可视化辅助理解数据结构与类别可分性。

## 2. 数据集详细描述
1. 数据来源：data.csv
2. 文件格式：分隔符为分号 ;
3. 数据规模：4424 条样本，37 列
4. 字段构成：36 个输入特征 + 1 个目标列 Target
5. 目标分布：
1. Graduate：2209，占比 49.9%
2. Dropout：1421，占比 32.1%
3. Enrolled：794，占比 18.0%
6. 缺失与重复：
1. 缺失值总量为 0
2. 重复样本数为 0
7. 类型结构：
1. 数值型特征约 11 个
2. 类别型特征约 25 个（多数字段已被整数编码）
8. 关键数值范围：
1. Age at enrollment：17 到 70
2. Admission grade：95.0 到 190.0
3. Semester grade：0.0 到 18.57
4. Unemployment rate：7.6 到 16.2
5. Inflation rate：-0.8 到 3.7
6. GDP：-4.06 到 3.51

## 3. 特征分组与业务含义
1. Demographic：Marital status、Nacionality、Gender、Age at enrollment、Displaced、International
2. Socioeconomic：父母学历与职业、Scholarship holder、Debtor、Tuition fees up to date
3. Academic progression：Application mode/order、Previous qualification、Admission grade、1st/2nd semester 学业表现
4. Macroeconomic：Unemployment rate、Inflation rate、GDP

## 4. 数据质量与风险点
1. 列名 Nacionality 存在拼写问题，语义上为 Nationality。
2. 多个整数编码字段本质是类别变量，不应按连续变量直接处理。
3. Enrolled 占比最低，存在类别不均衡风险。
4. Semester grade 中的 0 可能代表挂科或未参加评估，需结合 approved 与 without evaluations 字段解释。

## 5. 技术路线
1. 预处理：
1. 数值特征使用中位数填补并标准化。
2. 类别特征使用众数填补并 One-Hot 编码。
3. 使用统一 Pipeline 保证训练与测试处理一致，避免数据泄露。
2. 可视化与结构分析：
1. 目标类别分布可视化。
2. 关键变量分布与相关性分析。
3. t-SNE 二维映射观察类别重叠与边界。
3. 聚类分析：
1. 训练 K-Means 与 Agglomerative。
2. 评估 Silhouette、Calinski-Harabasz、Davies-Bouldin。
3. 使用 ARI 与 NMI 对照真实标签，判断天然可分性。
4. 监督学习建模：
1. 70/30 分层划分训练集与测试集。
2. 基线模型为 Logistic Regression 与 Decision Tree。
3. 输出 Accuracy、Precision、Recall、F1。
4. 使用混淆矩阵分析误分类。
5. 模型能力与调优：
1. 绘制多分类 ROC 曲线并计算 macro AUC。
2. 采用 GridSearchCV 与 StratifiedKFold，以 F1-macro 为主指标调参。
6. 开放探索：
1. 增加 Random Forest 作为扩展模型。
2. 使用 5 折交叉验证比较均值与方差。
3. 结合特征重要性做可解释分析与最终选型。

## 6. 评估标准
1. 主指标：F1-macro
2. 辅助指标：Accuracy、Precision-macro、Recall-macro、AUC-macro
3. 稳定性指标：交叉验证均值与标准差
4. 诊断工具：混淆矩阵、分类报告、ROC 曲线

## 7. 交付物
1. 完整实现 Notebook：project_tasks.ipynb
2. 数据文件：data.csv
3. 依赖说明：requirements.txt
4. 任务说明来源：DSAA2011-26sp-project_announce-L01.pdf
5. 项目计划文档：本文件

## 8. 范围边界
1. 当前范围不引入深度学习模型。
2. 当前范围不引入外部额外数据源。
3. 优先完成课程强制要求，再扩展提升项。

## 9. 执行时间建议
1. Day 1：数据审计、预处理流水线、基础 EDA。
2. Day 2：t-SNE 与聚类实验，完成聚类指标对比。
3. Day 3：监督模型训练、混淆矩阵、ROC/AUC。
4. Day 4：网格调参、交叉验证、模型对比与最终结论。
5. Day 5：整理报告叙述与图表，完成提交版本。