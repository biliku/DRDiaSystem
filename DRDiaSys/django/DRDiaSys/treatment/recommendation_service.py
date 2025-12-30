"""
治疗方案AI推荐服务
基于诊断报告和患者信息推荐治疗方案
"""
from typing import List, Dict, Optional
from .models import TreatmentPlanTemplate, TreatmentPlan
from diagnosis.models import DiagnosisReport, CaseRecord
from users.models import ConditionInfo


def get_dr_grade_from_report(report: DiagnosisReport) -> Optional[int]:
    """从诊断报告中提取DR分级"""
    # 从lesion_summary或doctor_conclusion中提取DR分级
    # 这里需要根据实际的数据结构来解析
    # 假设lesion_summary中包含分级信息，或者从其他字段提取
    lesion_summary = report.lesion_summary or {}
    
    # 如果报告中有明确的DR分级字段，直接返回
    # 否则根据病灶类型推断
    # 这里是一个简化的实现，实际需要根据具体的数据结构
    
    # 尝试从doctor_conclusion中提取
    if report.doctor_conclusion:
        conclusion = report.doctor_conclusion.lower()
        if 'dr0' in conclusion or '无dr' in conclusion or '正常' in conclusion:
            return 0
        elif 'dr1' in conclusion or '轻度' in conclusion:
            return 1
        elif 'dr2' in conclusion or '中度' in conclusion:
            return 2
        elif 'dr3' in conclusion or '重度' in conclusion:
            return 3
        elif 'dr4' in conclusion or '增殖性' in conclusion:
            return 4
    
    # 默认返回None，表示无法确定
    return None


def get_lesion_types_from_report(report: DiagnosisReport) -> List[str]:
    """从诊断报告中提取病灶类型"""
    lesion_summary = report.lesion_summary or {}
    lesion_types = []
    
    if isinstance(lesion_summary, list):
        for item in lesion_summary:
            if isinstance(item, dict):
                name = item.get('name', '')
                if name:
                    lesion_types.append(name)
    
    return lesion_types


def get_patient_diabetes_type(case: CaseRecord) -> Optional[str]:
    """获取患者糖尿病类型"""
    try:
        condition_info = ConditionInfo.objects.filter(
            user=case.patient
        ).order_by('-created_at').first()
        
        if condition_info and condition_info.has_diabetes:
            return condition_info.diabetes_type
    except:
        pass
    
    return None


def recommend_treatment_plans(
    case: CaseRecord,
    report: Optional[DiagnosisReport] = None
) -> List[Dict]:
    """
    推荐治疗方案
    
    Args:
        case: 病例记录
        report: 诊断报告（可选）
    
    Returns:
        推荐方案列表，每个方案包含模板信息和匹配度
    """
    recommendations = []
    
    # 获取DR分级
    dr_grade = None
    if report:
        dr_grade = get_dr_grade_from_report(report)
    
    # 获取病灶类型
    lesion_types = []
    if report:
        lesion_types = get_lesion_types_from_report(report)
    
    # 获取糖尿病类型
    diabetes_type = get_patient_diabetes_type(case)
    
    # 查询匹配的模板
    templates = TreatmentPlanTemplate.objects.filter(is_active=True)
    
    # 按优先级和匹配度排序
    scored_templates = []
    
    for template in templates:
        score = 0.0
        
        # DR分级匹配（权重最高）
        if template.dr_grade == -1:  # 适用于所有级别
            score += 50
        elif dr_grade is not None and template.dr_grade == dr_grade:
            score += 100
        elif dr_grade is not None:
            # 分级差异越小，分数越高
            diff = abs(template.dr_grade - dr_grade)
            score += max(0, 50 - diff * 20)
        
        # 病灶类型匹配
        if template.lesion_types:
            matched_lesions = set(template.lesion_types) & set(lesion_types)
            if matched_lesions:
                score += len(matched_lesions) * 10
        
        # 糖尿病类型匹配
        if template.diabetes_type:
            if diabetes_type and template.diabetes_type == diabetes_type:
                score += 20
        
        # 优先级加成
        score += template.priority
        
        if score > 0:
            scored_templates.append({
                'template': template,
                'score': score
            })
    
    # 按分数排序
    scored_templates.sort(key=lambda x: x['score'], reverse=True)
    
    # 转换为推荐格式
    for item in scored_templates[:5]:  # 返回前5个推荐
        template = item['template']
        recommendations.append({
            'template_id': template.id,
            'dr_grade': template.dr_grade,
            'medications': template.medications,
            'follow_up_plan': template.follow_up_plan,
            'lifestyle_advice': template.lifestyle_advice,
            'precautions': template.precautions,
            'score': item['score'],
            'match_reason': _generate_match_reason(template, dr_grade, lesion_types, diabetes_type)
        })
    
    return recommendations


def _generate_match_reason(
    template: TreatmentPlanTemplate,
    dr_grade: Optional[int],
    lesion_types: List[str],
    diabetes_type: Optional[str]
) -> str:
    """生成匹配原因说明"""
    reasons = []
    
    if dr_grade is not None:
        if template.dr_grade == -1:
            reasons.append("适用于所有DR分级")
        elif template.dr_grade == dr_grade:
            reasons.append(f"匹配DR{dr_grade}级")
    
    if template.lesion_types and lesion_types:
        matched = set(template.lesion_types) & set(lesion_types)
        if matched:
            reasons.append(f"匹配病灶类型: {', '.join(matched)}")
    
    if template.diabetes_type and diabetes_type == template.diabetes_type:
        reasons.append(f"匹配糖尿病类型: {template.diabetes_type}")
    
    return "; ".join(reasons) if reasons else "通用推荐"


def create_plan_from_template(
    template: TreatmentPlanTemplate,
    case: CaseRecord,
    report: Optional[DiagnosisReport] = None,
    created_by=None,
    title: Optional[str] = None
) -> TreatmentPlan:
    """
    从模板创建治疗方案
    
    Args:
        template: 治疗方案模板
        case: 病例记录
        report: 关联的诊断报告
        created_by: 创建医生
        title: 方案标题（可选）
    
    Returns:
        创建的治疗方案
    """
    plan = TreatmentPlan.objects.create(
        case=case,
        related_report=report,
        title=title or f"DR{template.dr_grade}级治疗方案",
        medications=template.medications,
        follow_up_plan=template.follow_up_plan,
        lifestyle_advice=template.lifestyle_advice,
        precautions=template.precautions,
        is_ai_recommended=True,
        template_used=template,
        created_by=created_by,
        status='draft'
    )
    
    # 生成方案编号
    plan.plan_number = plan.generate_plan_number()
    plan.save()
    
    return plan

