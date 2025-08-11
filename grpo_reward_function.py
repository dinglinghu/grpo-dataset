#!/usr/bin/env python3
"""
GRPO卫星任务冲突消解奖励函数
符合VERL开源训练平台GRPO算法要求
参考文档：https://verl.readthedocs.io/en/latest/preparation/reward_function.html

GRPO算法特点：直接从模型输出计算优势函数估计值，无需真实值对比
"""

import json
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_optimization_result(solution_str: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取优化结果"""
    
    try:
        # 尝试提取JSON格式的结果
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, solution_str, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
            return result.get('optimization_result', result)
        
        # 如果没有找到JSON块，尝试直接解析整个字符串中的JSON
        json_pattern2 = r'\{[^{}]*"optimization_result"[^{}]*\{.*?\}[^{}]*\}'
        json_match2 = re.search(json_pattern2, solution_str, re.DOTALL)
        
        if json_match2:
            result = json.loads(json_match2.group(0))
            return result.get('optimization_result', result)
        
        # 如果都没找到，返回None
        logger.warning("无法从模型输出中提取有效的JSON结果")
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}")
        return None
    except Exception as e:
        logger.error(f"提取优化结果时出错: {e}")
        return None

def calculate_coverage_score(result: Dict[str, Any]) -> float:
    """计算覆盖率得分"""

    try:
        total_tasks = result.get('total_meta_tasks', 0)
        successfully_covered = result.get('successfully_covered', 0)

        # 确保数据类型正确
        if not isinstance(total_tasks, (int, float)) or not isinstance(successfully_covered, (int, float)):
            logger.warning("覆盖率数据类型错误，返回0分")
            return 0.0

        if total_tasks == 0:
            return 0.0

        coverage_rate = successfully_covered / total_tasks

        # 覆盖率得分：使用指数函数奖励高覆盖率
        coverage_score = np.power(coverage_rate, 0.5) * 100

        return min(coverage_score, 100.0)

    except Exception as e:
        logger.error(f"计算覆盖率得分时出错: {e}")
        return 0.0

def calculate_gdop_score(result: Dict[str, Any]) -> float:
    """计算GDOP质量得分"""

    try:
        assignments = result.get('assignments', [])
        if not assignments:
            return 0.0

        gdop_values = []
        for assignment in assignments:
            if assignment.get('is_successfully_covered', False):
                for satellite in assignment.get('assigned_satellites', []):
                    gdop_value = satellite.get('gdop_value', 2.0)
                    # 确保GDOP值是数字类型
                    if isinstance(gdop_value, (int, float)):
                        gdop_values.append(float(gdop_value))

        if not gdop_values:
            return 0.0

        avg_gdop = np.mean(gdop_values)

        # GDOP得分：值越小越好，使用反比例函数
        # 理想GDOP值在1.0-1.5之间，超过2.0认为较差
        if avg_gdop <= 1.0:
            gdop_score = 100.0
        elif avg_gdop <= 1.5:
            gdop_score = 100.0 - (avg_gdop - 1.0) * 40  # 1.0-1.5之间线性下降
        elif avg_gdop <= 2.0:
            gdop_score = 60.0 - (avg_gdop - 1.5) * 80   # 1.5-2.0之间快速下降
        else:
            gdop_score = max(0.0, 20.0 - (avg_gdop - 2.0) * 10)  # 超过2.0大幅惩罚

        return max(gdop_score, 0.0)

    except Exception as e:
        logger.error(f"计算GDOP得分时出错: {e}")
        return 0.0

def calculate_quality_score(result: Dict[str, Any]) -> float:
    """计算分配质量得分"""
    
    assignments = result.get('assignments', [])
    if not assignments:
        return 0.0
    
    quality_scores = []
    quality_weights = {
        'excellent': 100.0,
        'good': 75.0,
        'fair': 50.0,
        'poor': 20.0
    }
    
    for assignment in assignments:
        quality = assignment.get('coverage_quality', 'poor')
        score = quality_weights.get(quality, 0.0)
        quality_scores.append(score)
    
    return np.mean(quality_scores) if quality_scores else 0.0

def calculate_efficiency_score(result: Dict[str, Any]) -> float:
    """计算资源利用效率得分"""
    
    assignments = result.get('assignments', [])
    if not assignments:
        return 0.0
    
    # 统计卫星使用情况
    satellite_usage = {}
    total_assignments = 0
    
    for assignment in assignments:
        if assignment.get('is_successfully_covered', False):
            for satellite in assignment.get('assigned_satellites', []):
                sat_id = satellite.get('satellite_id', '')
                if sat_id:
                    satellite_usage[sat_id] = satellite_usage.get(sat_id, 0) + 1
                    total_assignments += 1
    
    if not satellite_usage:
        return 0.0
    
    # 计算负载均衡度
    usage_values = list(satellite_usage.values())
    avg_usage = np.mean(usage_values)
    usage_std = np.std(usage_values)
    
    # 负载越均衡，效率得分越高
    if avg_usage == 0:
        return 0.0
    
    balance_ratio = 1.0 - (usage_std / avg_usage) if avg_usage > 0 else 0.0
    efficiency_score = balance_ratio * 100.0
    
    return max(efficiency_score, 0.0)

def compute_score(data_source: str, solution_str: str, ground_truth: Dict[str, Any], 
                 extra_info: Optional[Dict] = None) -> float:
    """
    GRPO专用奖励函数：直接从模型输出计算优势函数估计值
    
    Args:
        data_source: 数据源标识
        solution_str: 模型输出的解决方案字符串
        ground_truth: 真实值（用于参考，但GRPO不需要对比）
        extra_info: 额外信息
    
    Returns:
        float: 优势函数估计值 (0-100分)
    """
    
    # 提取模型输出的优化结果
    result = extract_optimization_result(solution_str)
    
    if result is None:
        # 如果无法解析结果，给予低分
        logger.warning("无法解析模型输出，给予低分")
        return 10.0
    
    # 计算各项得分
    coverage_score = calculate_coverage_score(result)      # 覆盖率得分 (40%)
    gdop_score = calculate_gdop_score(result)             # GDOP质量得分 (30%)
    quality_score = calculate_quality_score(result)       # 分配质量得分 (20%)
    efficiency_score = calculate_efficiency_score(result) # 效率得分 (10%)
    
    # 加权计算总分
    total_score = (
        coverage_score * 0.4 +
        gdop_score * 0.3 +
        quality_score * 0.2 +
        efficiency_score * 0.1
    )
    
    # 确保得分在合理范围内
    final_score = max(0.0, min(100.0, total_score))
    
    # 记录详细信息（用于调试）
    logger.info(f"得分详情 - 覆盖率: {coverage_score:.1f}, GDOP: {gdop_score:.1f}, "
               f"质量: {quality_score:.1f}, 效率: {efficiency_score:.1f}, "
               f"总分: {final_score:.1f}")
    
    return final_score

def compute_detailed_score(data_source: str, solution_str: str, ground_truth: Dict[str, Any],
                          extra_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    计算详细的得分信息（用于分析和调试）
    
    Returns:
        Dict包含各项得分的详细信息
    """
    
    result = extract_optimization_result(solution_str)
    
    if result is None:
        return {
            "total_score": 10.0,
            "coverage_score": 0.0,
            "gdop_score": 0.0,
            "quality_score": 0.0,
            "efficiency_score": 0.0,
            "error": "无法解析模型输出"
        }
    
    coverage_score = calculate_coverage_score(result)
    gdop_score = calculate_gdop_score(result)
    quality_score = calculate_quality_score(result)
    efficiency_score = calculate_efficiency_score(result)
    
    total_score = (
        coverage_score * 0.4 +
        gdop_score * 0.3 +
        quality_score * 0.2 +
        efficiency_score * 0.1
    )
    
    return {
        "total_score": max(0.0, min(100.0, total_score)),
        "coverage_score": coverage_score,
        "gdop_score": gdop_score,
        "quality_score": quality_score,
        "efficiency_score": efficiency_score,
        "parsed_result": result,
        "ground_truth_comparison": {
            "gt_coverage_rate": ground_truth.get('coverage_rate', 0.0),
            "gt_average_gdop": ground_truth.get('average_gdop', 0.0),
            "gt_total_tasks": ground_truth.get('total_meta_tasks', 0)
        }
    }
