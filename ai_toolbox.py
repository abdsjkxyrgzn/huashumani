"""
===================================================================
AI Toolbox for Mathematical Modeling Competition (MCM)
===================================================================

Version: 1.0
Author: 俊宇
Creation Date: 2025-08-XX

Description:
这是一个为数学建模竞赛量身打造的、标准化的Python工具箱。
它封装了竞赛中最高频使用的数据处理、预测、机器学习及优化求解算法，
旨在最大化地提升建模效率与代码复用性。

Modules:
- Prediction Models (ARIMA, Grey Forecast)
- Machine Learning Models (K-Means, RandomForest, XGBoost)
- Evaluation & Visualization
- Optimization Solvers (Linear/Integer Programming)

"""
"""
俊宇的数学建模与AI工具箱
版本: v0.1 - 预测模型模块
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
# --- 预测模型 (Prediction Models) ---

def run_grey_forecast(data, n_preds):
    """
    执行灰色预测GM(1,1)模型。

    灰色预测适用于数据量较少、呈现指数增长趋势的时间序列。

    参数:
    - data (list or pd.Series): 输入的原始时间序列数据，必须是一维的。
    - n_preds (int): 需要向未来预测的步数。

    返回:
    - np.ndarray: 包含未来n_preds个预测值的Numpy数组。
    """
    print(f"--- 正在执行灰色预测 (GM(1,1))，预测未来 {n_preds} 步 ---")
    
    # 1. 累加生成 (AGO)
    x0 = np.array(data)
    x1 = np.cumsum(x0)

    # 2. 构造紧邻均值生成序列 (Z)
    z = (x1[:-1] + x1[1:]) / 2.0

    # 3. 构造数据矩阵B和数据向量Y
    B = np.vstack((-z, np.ones_like(z))).T
    Y = x0[1:].reshape(-1, 1)

    # 4. 最小二乘法求解参数 [a, u]^T
    try:
        # [[a], [u]] = (B^T * B)^-1 * B^T * Y
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        a, u = params.flatten()
    except np.linalg.LinAlgError:
        print("❌ 灰色预测失败：矩阵不可逆。请检查输入数据。")
        return None

    # 5. 建立预测模型
    def predict_value(k):
        return (x0[0] - u / a) * np.exp(-a * k) + u / a

    # 6. 累减还原，得到预测值
    f = np.zeros(len(x0) + n_preds)
    f[0] = x0[0]
    for k in range(1, len(x0) + n_preds):
        f[k] = predict_value(k) - predict_value(k - 1)
    
    predictions = f[-n_preds:]
    print(f"✅ 灰色预测完成！")
    return predictions


def run_arima_forecast(data, order, n_preds):
    """
    执行ARIMA时间序列预测模型。

    ARIMA适用于具有趋势性、季节性、周期性的时间序列数据。

    参数:
    - data (list or pd.Series): 输入的原始时间序列数据。
    - order (tuple): ARIMA模型的(p, d, q)参数。
        - p: 自回归项数
        - d: 差分阶数
        - q: 移动平均项数
    - n_preds (int): 需要向未来预测的步数。

    返回:
    - pd.Series: 包含未来n_preds个预测值的Pandas Series。
    """
    print(f"--- 正在执行ARIMA模型 (order={order})，预测未来 {n_preds} 步 ---")
    
    # 忽略ARIMA模型可能产生的警告信息，保持输出整洁
    warnings.filterwarnings("ignore")
    
    try:
        # 将数据转换为Pandas Series，增强模型的稳定性
        ts = pd.Series(data)
        
        # 训练ARIMA模型
        model = ARIMA(ts, order=order)
        model_fit = model.fit()

        # 预测未来n_preds步
        forecast = model_fit.forecast(steps=n_preds)
        
        print(f"✅ ARIMA预测完成！")
        return forecast
    except Exception as e:
        print(f"❌ ARIMA预测失败: {e}")
        return None
# ai_toolbox.py
# (文件顶部是你已经写好的灰色预测和ARIMA代码)
# ...

# --- 机器学习模型 (Machine Learning Models) - V1.1 (鲁棒性增强版) ---

import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

# 配置一个简单而专业的日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_kmeans(data, n_clusters, random_state=42):
    """
    执行K-Means聚类算法 (鲁棒性增强版)。

    参数:
    - data (pd.DataFrame or np.ndarray): 输入的数据。
    - n_clusters (int): 期望划分出的簇的数量。
    - random_state (int): 随机种子。

    返回:
    - dict or None: 成功则返回包含'labels'和'centers'的字典，失败则返回None。
    """
    logger.info(f"开始执行K-Means聚类，目标簇数: {n_clusters}")
    
    # 1. 严格的输入检查 (Input Validation)
    try:
        assert isinstance(data, (np.ndarray, pd.DataFrame)), "输入数据必须是Numpy数组或Pandas DataFrame！"
        assert data.ndim == 2, f"输入数据必须是二维的，现在是 {data.ndim} 维！"
        assert isinstance(n_clusters, int) and n_clusters > 0, "簇的数量必须是正整数！"
        if isinstance(data, pd.DataFrame):
            assert not data.isnull().values.any(), "输入数据中存在NaN值，请先处理！"
    except AssertionError as e:
        logger.error(f"K-Means 输入参数错误: {e}")
        return None

    # 2. 精细的异常捕获与核心逻辑
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        kmeans.fit(data)
        
        results = {
            'labels': kmeans.labels_,
            'centers': kmeans.cluster_centers_
        }
        logger.info("K-Means聚类成功！")
        return results
    except ConvergenceWarning as w:
        logger.warning(f"K-Means警告：算法可能未完全收敛。{w}")
        return None # 或者返回部分结果，取决于你的需求
    except Exception as e:
        logger.error("K-Means发生未知错误。", exc_info=True) # exc_info=True 会打印详细的错误栈
        return None

def run_random_forest_classifier(X, y, test_size=0.3, random_state=42, **kwargs):
    """
    执行随机森林分类任务 (鲁棒性增强版)。
    ... (参数说明不变)
    """
    logger.info("开始执行随机森林分类任务...")

    # 1. 严格的输入检查
    try:
        assert len(X) == len(y), f"特征(X)和标签(y)的长度不一致！({len(X)} vs {len(y)})"
    except AssertionError as e:
        logger.error(f"随机森林分类器输入参数错误: {e}")
        return None

    # 2. 精细的异常捕获与核心逻辑
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if hasattr(y, 'unique') else None)
        
        n_estimators = kwargs.get('n_estimators', 100)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)
        
        results = {
            'model': clf, 'X_test': X_test, 'y_test': y_test,
            'predictions': predictions, 'probabilities': probabilities
        }
        logger.info("随机森林分类成功！")
        return results
    except ValueError as e:
        logger.error(f"随机森林分类失败：请检查数据中是否包含NaN或无限值。错误详情: {e}")
        return None
    except Exception as e:
        logger.error("随机森林发生未知错误。", exc_info=True)
        return None
        

def run_xgboost_regressor(X, y, test_size=0.3, random_state=42, **kwargs):
    """
    执行XGBoost回归任务 (鲁棒性增强版)。
    ... (参数说明不变)
    """
    logger.info("开始执行XGBoost回归任务...")

    # 1. 严格的输入检查
    try:
        assert len(X) == len(y), f"特征(X)和目标(y)的长度不一致！({len(X)} vs {len(y)})"
    except AssertionError as e:
        logger.error(f"XGBoost回归器输入参数错误: {e}")
        return None
    
    # 2. 精细的异常捕获与核心逻辑
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        n_estimators = kwargs.get('n_estimators', 100)
        learning_rate = kwargs.get('learning_rate', 0.1)
        
        reg = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        reg.fit(X_train, y_train)

        predictions = reg.predict(X_test)

        results = {
            'model': reg, 'X_test': X_test, 'y_test': y_test,
            'predictions': predictions
        }
        logger.info("XGBoost回归成功！")
        return results
    except ImportError:
        logger.error("XGBoost回归失败: 请先通过`pip install xgboost`安装xgboost库。")
        return None
    except Exception as e:
        logger.error("XGBoost发生未知错误。", exc_info=True)
        return None
    # ai_toolbox.py
# (文件顶部是之前的所有代码)
# ...

# --- 模型评估与可视化 (Evaluation & Visualization) ---

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_regression(y_true, y_pred):
    """
    计算并返回回归模型的全套评估指标。

    参数:
    - y_true: 真实的数值标签。
    - y_pred: 模型预测的数值标签。

    返回:
    - dict: 包含MSE, RMSE, MAE, R2四个核心指标的字典。
    """
    logger.info("开始计算回归模型评估指标...")
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE (均方误差)': mse,
            'RMSE (均方根误差)': rmse,
            'MAE (平均绝对误差)': mae,
            'R2 Score (决定系数)': r2
        }
        logger.info(f"回归模型评估完成: RMSE={rmse:.4f}, R2 Score={r2:.4f}")
        return metrics
    except Exception as e:
        logger.error("回归模型评估失败。", exc_info=True)
        return None

def evaluate_classification(y_true, y_pred, labels=None):
    """
    计算并返回分类模型的全套评估指标，包括混淆矩阵。

    参数:
    - y_true: 真实的类别标签。
    - y_pred: 模型预测的类别标签。
    - labels (list, optional): 混淆矩阵的标签顺序。

    返回:
    - dict: 包含Accuracy, Precision, Recall, F1-Score和混淆矩阵的字典。
    """
    logger.info("开始计算分类模型评估指标...")
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        metrics = {
            'Accuracy (准确率)': accuracy,
            'Precision (精确率)': precision,
            'Recall (召回率)': recall,
            'F1-Score': f1,
            'Confusion Matrix (混淆矩阵)': cm
        }
        logger.info(f"分类模型评估完成: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}")
        return metrics
    except Exception as e:
        logger.error("分类模型评估失败。", exc_info=True)
        return None


# ai_toolbox.py (部分升级)

# (注意，你只需要替换这一个函数)
def plot_regression_results(y_true, y_pred, title="模型预测结果", 
                             x_true=None, x_pred=None, 
                             y_true_label="真实值", y_pred_label="预测值",
                             plot_type='scatter'):
    """
    可视化回归/预测模型的结果。 V1.1 新增时序绘图功能

    参数:
    ... (原有参数不变)
    - x_true, x_pred: 绘制折线图时的X轴坐标。
    - y_true_label, y_pred_label: 图例标签。
    - plot_type (str): 'scatter' (散点图) 或 'line' (折线图)。
    """
    plt.figure(figsize=(15, 7))

    if plot_type == 'scatter':
        # 绘制散点图，适合通用的回归问题
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, label="预测值 vs. 真实值")
        perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
        plt.plot(perfect_line, perfect_line, color='red', linestyle='--', label="完美预测 (y=x)")
        plt.xlabel("真实值 (True Values)", fontsize=12)
        plt.ylabel("预测值 (Predictions)", fontsize=12)
        
    elif plot_type == 'line':
        # 绘制折线图，适合时间序列问题
        if x_true is None: x_true = np.arange(len(y_true))
        if x_pred is None: x_pred = np.arange(len(y_true), len(y_true) + len(y_pred))
            
        plt.plot(x_true, y_true, marker='o', label=y_true_label)
        plt.plot(x_pred, y_pred, marker='o', linestyle='--', color='red', label=y_pred_label)
        plt.xlabel("时间索引", fontsize=12)
        plt.ylabel("数值", fontsize=12)
        
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm, class_names, title='混淆矩阵'):
    """
    可视化混淆矩阵。

    参数:
    - cm (np.ndarray): `evaluate_classification`函数返回的混淆矩阵。
    - class_names (list): 类别名称列表。
    - title (str): 图表标题。
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('真实标签 (True Label)', fontsize=12)
    plt.xlabel('预测标签 (Predicted Label)', fontsize=12)
    plt.show()
# ai_toolbox.py
# (文件顶部是之前的所有代码)
# ...

# --- 运筹优化求解器 (Optimization Solvers) ---

import pulp

def solve_lp_problem(objective_coeffs, constraint_coeffs, constraint_rhs, 
                     constraint_senses, var_names=None, 
                     problem_name="LP_Problem", sense=pulp.LpMaximize,
                     var_cat='Continuous', var_low_bound=0):
    """
    一个通用的线性/整数规划问题求解器。
    
    采用矩阵/向量化的方式定义问题，更加高效和通用。
    最大化/最小化: c^T * x
    约束: A * x <= b  (或 >=, =)

    参数:
    - objective_coeffs (list or np.array): 目标函数系数向量 (c)。
    - constraint_coeffs (list of lists or 2D np.array): 约束系数矩阵 (A)。
    - constraint_rhs (list or np.array): 约束右侧的值向量 (b)。
    - constraint_senses (list of pulp constants): 约束的类型 (pulp.LpConstraintLE for <=, 
                                                              pulp.LpConstraintGE for >=, 
                                                              pulp.LpConstraintEQ for =)。
    - var_names (list of strings, optional): 决策变量的名称。
    - problem_name (str): 问题名称。
    - sense (pulp constant): 求解目标 (pulp.LpMaximize or pulp.LpMinimize)。
    - var_cat (str): 变量类型 ('Continuous' or 'Integer')。
    - var_low_bound: 变量的下界。

    返回:
    - dict or None: 包含求解状态、目标函数最优值、各变量最优取值的字典。
    """
    logger.info(f"开始求解规划问题: {problem_name}")
    
    try:
        # 1. 初始化问题
        prob = pulp.LpProblem(problem_name, sense)

        # 2. 定义决策变量
        num_vars = len(objective_coeffs)
        if var_names is None:
            var_names = [f'x{i}' for i in range(num_vars)]
        
        variables = pulp.LpVariable.dicts("Var", var_names, lowBound=var_low_bound, cat=var_cat)
        
        # 将字典转换为有序列表，确保顺序一致
        var_list = [variables[name] for name in var_names]

        # 3. 定义目标函数
        prob += pulp.lpDot(objective_coeffs, var_list), "Objective_Function"

        # 4. 添加约束条件
        num_constraints = len(constraint_rhs)
        for i in range(num_constraints):
            constraint_expr = pulp.lpDot(constraint_coeffs[i], var_list)
            sense = constraint_senses[i]
            rhs = constraint_rhs[i]
            
            if sense == pulp.LpConstraintLE:
                prob += constraint_expr <= rhs, f"Constraint_{i}"
            elif sense == pulp.LpConstraintGE:
                prob += constraint_expr >= rhs, f"Constraint_{i}"
            elif sense == pulp.LpConstraintEQ:
                prob += constraint_expr == rhs, f"Constraint_{i}"

        # 5. 求解问题
        prob.solve()

        # 6. 整理并返回结果
        solution = {'status': pulp.LpStatus[prob.status]}
        if prob.status == pulp.LpStatusOptimal:
            solution['objective_value'] = pulp.value(prob.objective)
            solution['variables'] = {v.name: v.varValue for v in prob.variables()}
        
        logger.info(f"规划问题求解完成，状态: {solution['status']}")
        return solution
        
    except Exception as e:
        logger.error("规划问题求解失败。", exc_info=True)
        return None