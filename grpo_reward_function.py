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
    """
    从模型输出中提取优化结果 - 增强版
    支持多种输出格式，从差到好的输出都能处理
    """

    if not solution_str or not solution_str.strip():
        logger.warning("输入为空，返回默认结果")
        return create_default_result()

    # 清理输入文本
    cleaned_str = clean_solution_text(solution_str)

    # 策略1: 尝试提取标准JSON格式
    result = extract_standard_json(cleaned_str)
    if result:
        logger.info("成功提取标准JSON格式")
        return result

    # 策略2: 尝试提取嵌套JSON格式
    result = extract_nested_json(cleaned_str)
    if result:
        logger.info("成功提取嵌套JSON格式")
        return result

    # 策略3: 尝试提取不完整的JSON
    result = extract_partial_json(cleaned_str)
    if result:
        logger.info("成功提取部分JSON格式")
        return result

    # 策略4: 尝试从文本中提取关键信息
    result = extract_from_text_patterns(cleaned_str)
    if result:
        logger.info("成功从文本模式提取信息")
        return result

    # 策略5: 尝试从表格或列表格式提取
    result = extract_from_structured_text(cleaned_str)
    if result:
        logger.info("成功从结构化文本提取信息")
        return result

    # 策略6: 最后的兜底策略 - 智能猜测
    result = intelligent_fallback_extraction(cleaned_str)
    if result:
        logger.info("使用智能兜底策略提取信息")
        return result

    # 如果所有策略都失败，返回最小可用结果
    logger.warning("所有提取策略失败，返回最小可用结果")
    return create_minimal_result(cleaned_str)

def clean_solution_text(text: str) -> str:
    """清理和标准化输入文本"""
    if not text:
        return ""

    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text.strip())

    # 修复常见的JSON格式问题
    text = text.replace('```json', '```json\n')
    text = text.replace('```', '\n```')
    text = re.sub(r'([}\]])\s*([{\[])', r'\1,\2', text)  # 添加缺失的逗号

    # 修复引号问题
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)  # 添加缺失的引号
    text = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', text)  # 为值添加引号

    return text

def create_default_result() -> Dict[str, Any]:
    """创建默认的优化结果"""
    return {
        "total_meta_tasks": 0,
        "successfully_covered": 0,
        "coverage_rate": 0.0,
        "average_gdop": 10.0,  # 最差的GDOP值
        "assignments": []
    }

def create_minimal_result(text: str) -> Dict[str, Any]:
    """基于文本内容创建最小可用结果"""
    # 尝试从文本中提取数字信息
    numbers = re.findall(r'\d+\.?\d*', text)

    # 估算任务数量
    task_count = 0
    if '任务' in text or 'task' in text.lower():
        task_matches = re.findall(r'(\d+).*?任务|task.*?(\d+)', text, re.IGNORECASE)
        if task_matches:
            task_count = max([int(m[0] or m[1]) for m in task_matches if m[0] or m[1]])

    if task_count == 0 and numbers:
        task_count = int(float(numbers[0])) if numbers else 1

    # 估算覆盖率
    coverage_rate = 0.0
    coverage_matches = re.findall(r'(\d+\.?\d*)%|覆盖.*?(\d+\.?\d*)', text)
    if coverage_matches:
        for match in coverage_matches:
            val = float(match[0] or match[1])
            if 0 <= val <= 100:
                coverage_rate = val / 100
                break

    successfully_covered = int(task_count * coverage_rate)

    return {
        "total_meta_tasks": max(1, task_count),
        "successfully_covered": successfully_covered,
        "coverage_rate": coverage_rate,
        "average_gdop": 5.0,  # 中等GDOP值
        "assignments": []
    }

def extract_standard_json(text: str) -> Optional[Dict[str, Any]]:
    """提取标准JSON格式"""
    try:
        # 策略1: 提取```json```块中的内容
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```JSON\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?"optimization_result".*?\})\s*```',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    result = json.loads(match)
                    if 'optimization_result' in result:
                        return result['optimization_result']
                    elif any(key in result for key in ['total_meta_tasks', 'coverage_rate', 'assignments']):
                        return result
                except json.JSONDecodeError:
                    continue

        return None

    except Exception as e:
        logger.debug(f"标准JSON提取失败: {e}")
        return None

def extract_nested_json(text: str) -> Optional[Dict[str, Any]]:
    """提取嵌套JSON格式"""
    try:
        # 查找包含optimization_result的JSON结构
        patterns = [
            r'\{[^{}]*"optimization_result"[^{}]*:\s*(\{.*?\})[^{}]*\}',
            r'"optimization_result"\s*:\s*(\{.*?\})',
            r'optimization_result["\']?\s*:\s*(\{.*?\})',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # 尝试解析嵌套的JSON
                    result = json.loads(match)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    # 尝试修复常见的JSON错误
                    fixed_json = fix_json_format(match)
                    if fixed_json:
                        try:
                            result = json.loads(fixed_json)
                            if isinstance(result, dict):
                                return result
                        except json.JSONDecodeError:
                            continue

        return None

    except Exception as e:
        logger.debug(f"嵌套JSON提取失败: {e}")
        return None

def extract_partial_json(text: str) -> Optional[Dict[str, Any]]:
    """提取不完整的JSON"""
    try:
        # 查找JSON片段并尝试重构
        json_fragments = re.findall(r'\{[^{}]*(?:"total_meta_tasks"|"coverage_rate"|"assignments")[^{}]*\}', text, re.DOTALL)

        for fragment in json_fragments:
            # 尝试修复和解析片段
            fixed_fragment = fix_json_format(fragment)
            if fixed_fragment:
                try:
                    result = json.loads(fixed_fragment)
                    if isinstance(result, dict) and any(key in result for key in ['total_meta_tasks', 'coverage_rate']):
                        return complete_partial_result(result)
                except json.JSONDecodeError:
                    continue

        return None

    except Exception as e:
        logger.debug(f"部分JSON提取失败: {e}")
        return None

def fix_json_format(json_str: str) -> Optional[str]:
    """修复常见的JSON格式错误"""
    try:
        # 移除多余的逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # 添加缺失的引号
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)

        # 修复布尔值
        json_str = re.sub(r':\s*true\b', ': true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r':\s*false\b', ': false', json_str, flags=re.IGNORECASE)

        # 修复null值
        json_str = re.sub(r':\s*null\b', ': null', json_str, flags=re.IGNORECASE)

        # 确保字符串值有引号
        json_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', json_str)

        return json_str

    except Exception:
        return None

def complete_partial_result(partial: Dict[str, Any]) -> Dict[str, Any]:
    """补全不完整的结果"""
    result = create_default_result()

    # 更新已有的字段
    for key, value in partial.items():
        if key in result:
            result[key] = value

    # 计算缺失的字段
    if 'total_meta_tasks' in result and 'successfully_covered' in result:
        if result['total_meta_tasks'] > 0:
            result['coverage_rate'] = result['successfully_covered'] / result['total_meta_tasks']

    if 'coverage_rate' in result and 'total_meta_tasks' in result:
        result['successfully_covered'] = int(result['coverage_rate'] * result['total_meta_tasks'])

    return result

def extract_from_text_patterns(text: str) -> Optional[Dict[str, Any]]:
    """从文本模式中提取关键信息"""
    try:
        result = create_default_result()
        found_data = False

        # 提取任务总数
        task_patterns = [
            r'总.*?任务.*?(\d+)',
            r'total.*?tasks?.*?(\d+)',
            r'(\d+).*?个.*?任务',
            r'任务.*?数量.*?(\d+)',
        ]

        for pattern in task_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['total_meta_tasks'] = int(match.group(1))
                found_data = True
                break

        # 提取成功覆盖数
        covered_patterns = [
            r'成功.*?覆盖.*?(\d+)',
            r'successfully.*?covered.*?(\d+)',
            r'覆盖.*?(\d+).*?个',
            r'完成.*?(\d+).*?任务',
        ]

        for pattern in covered_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['successfully_covered'] = int(match.group(1))
                found_data = True
                break

        # 提取覆盖率
        coverage_patterns = [
            r'覆盖率.*?(\d+\.?\d*)%',
            r'coverage.*?(\d+\.?\d*)%',
            r'(\d+\.?\d*)%.*?覆盖',
            r'覆盖.*?(\d+\.?\d*)',
        ]

        for pattern in coverage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rate = float(match.group(1))
                if rate > 1:  # 如果是百分比形式
                    rate = rate / 100
                result['coverage_rate'] = rate
                found_data = True
                break

        # 提取GDOP值
        gdop_patterns = [
            r'GDOP.*?(\d+\.?\d*)',
            r'gdop.*?(\d+\.?\d*)',
            r'几何.*?精度.*?(\d+\.?\d*)',
            r'平均.*?GDOP.*?(\d+\.?\d*)',
        ]

        for pattern in gdop_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['average_gdop'] = float(match.group(1))
                found_data = True
                break

        # 计算缺失字段
        if result['total_meta_tasks'] > 0 and result['coverage_rate'] > 0:
            result['successfully_covered'] = int(result['total_meta_tasks'] * result['coverage_rate'])
        elif result['total_meta_tasks'] > 0 and result['successfully_covered'] > 0:
            result['coverage_rate'] = result['successfully_covered'] / result['total_meta_tasks']

        return result if found_data else None

    except Exception as e:
        logger.debug(f"文本模式提取失败: {e}")
        return None

def extract_from_structured_text(text: str) -> Optional[Dict[str, Any]]:
    """从结构化文本（表格、列表）中提取信息"""
    try:
        result = create_default_result()
        found_data = False

        # 查找类似表格的结构
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 查找键值对模式
            kv_patterns = [
                (r'总任务|total.*tasks?', 'total_meta_tasks'),
                (r'成功覆盖|successfully.*covered', 'successfully_covered'),
                (r'覆盖率|coverage.*rate', 'coverage_rate'),
                (r'平均.*GDOP|average.*gdop', 'average_gdop'),
            ]

            for pattern, key in kv_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # 提取数值
                    numbers = re.findall(r'(\d+\.?\d*)', line)
                    if numbers:
                        value = float(numbers[-1])  # 取最后一个数字
                        if key == 'coverage_rate' and value > 1:
                            value = value / 100
                        result[key] = int(value) if key in ['total_meta_tasks', 'successfully_covered'] else value
                        found_data = True

        # 查找分配信息
        assignment_patterns = [
            r'任务.*?(\w+).*?卫星.*?(\w+)',
            r'(\w+).*?分配.*?(\w+)',
            r'satellite.*?(\w+).*?task.*?(\w+)',
        ]

        assignments = []
        for pattern in assignment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                assignments.append({
                    'meta_task_id': match[0],
                    'assigned_satellites': [{'satellite_id': match[1]}],
                    'is_successfully_covered': True,
                    'coverage_quality': 'fair'
                })

        if assignments:
            result['assignments'] = assignments[:10]  # 限制数量
            found_data = True

        return result if found_data else None

    except Exception as e:
        logger.debug(f"结构化文本提取失败: {e}")
        return None

def intelligent_fallback_extraction(text: str) -> Optional[Dict[str, Any]]:
    """智能兜底提取策略"""
    try:
        result = create_default_result()

        # 统计文本中的数字
        numbers = [float(n) for n in re.findall(r'\d+\.?\d*', text)]
        if not numbers:
            return None

        # 智能猜测任务数量（通常是较大的整数）
        integers = [int(n) for n in numbers if n == int(n) and 1 <= n <= 1000]
        if integers:
            result['total_meta_tasks'] = max(integers)

        # 智能猜测覆盖率（0-1之间的小数或0-100的百分比）
        percentages = [n for n in numbers if 0 <= n <= 100]
        if percentages:
            rate = max(percentages)
            if rate > 1:
                rate = rate / 100
            result['coverage_rate'] = rate
            result['successfully_covered'] = int(result['total_meta_tasks'] * rate)

        # 智能猜测GDOP值（通常在1-10之间）
        gdop_candidates = [n for n in numbers if 1.0 <= n <= 10.0]
        if gdop_candidates:
            result['average_gdop'] = min(gdop_candidates)  # 选择最好的GDOP值

        # 检查是否有有效数据
        if (result['total_meta_tasks'] > 0 or
            result['coverage_rate'] > 0 or
            result['average_gdop'] < 10.0):
            return result

        return None

    except Exception as e:
        logger.debug(f"智能兜底提取失败: {e}")
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
