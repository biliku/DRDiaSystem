from django.db import models
from django.contrib.auth.models import User
from diagnosis.models import CaseRecord, DiagnosisReport


class TreatmentPlanTemplate(models.Model):
    """治疗方案模板，用于AI推荐"""

    dr_grade = models.IntegerField(verbose_name='DR分级', help_text='0-4级，-1表示适用于所有级别')
    lesion_types = models.JSONField(default=list, blank=True, verbose_name='适用病灶类型')
    diabetes_type = models.CharField(max_length=20, blank=True, verbose_name='适用糖尿病类型')

    # ========== 基础管理目标 ==========
    blood_sugar_target = models.JSONField(
        default=dict,
        blank=True,
        verbose_name='血糖控制目标',
        help_text='{"fasting": "空腹血糖", "postprandial": "餐后血糖", "hba1c": "糖化血红蛋白"}'
    )
    blood_pressure_target = models.JSONField(
        default=dict,
        blank=True,
        verbose_name='血压控制目标',
        help_text='{"systolic": "收缩压", "diastolic": "舒张压"}'
    )
    lipid_management = models.TextField(blank=True, verbose_name='血脂管理建议')

    # ========== 眼科治疗 ==========
    anti_vegf_treatment = models.JSONField(
        default=dict,
        blank=True,
        verbose_name='抗VEGF治疗方案',
        help_text='{"drug": "药物名称", "frequency": "注射频率", "course": "疗程", "notes": "注意事项"}'
    )
    laser_treatment = models.JSONField(
        default=dict,
        blank=True,
        verbose_name='激光治疗方案',
        help_text='{"type": "类型(PRP/格栅)", "sessions": "次数", "interval": "间隔时间"}'
    )
    surgical_treatment = models.TextField(
        blank=True,
        verbose_name='手术治疗指征',
        help_text='玻切手术适应症描述'
    )

    # ========== 药物治疗 ==========
    medications = models.JSONField(
        default=list,
        verbose_name='药物列表',
        help_text='[{"name": "药物名称", "dosage": "用法用量", "frequency": "频次", "duration": "疗程", "category": "类别", "notes": "备注"}]'
    )

    # ========== 生活方式干预 ==========
    diet_guidance = models.TextField(blank=True, verbose_name='饮食指导')
    exercise_guidance = models.TextField(blank=True, verbose_name='运动指导')
    lifestyle_advice = models.TextField(blank=True, verbose_name='综合生活方式建议')

    # ========== 随访监测 ==========
    follow_up_plan = models.JSONField(
        default=dict,
        verbose_name='复查计划',
        help_text='{"interval_days": "间隔天数", "check_items": ["检查项目"], "next_date": "下次日期"}'
    )
    monitoring_plan = models.JSONField(
        default=list,
        blank=True,
        verbose_name='监测计划',
        help_text='[{"item": "监测项目", "frequency": "频率", "target": "目标值"}]'
    )
    warning_symptoms = models.TextField(
        blank=True,
        verbose_name='预警症状',
        help_text='需要立即就医的症状描述'
    )

    # ========== 其他 ==========
    precautions = models.TextField(blank=True, verbose_name='注意事项')
    priority = models.IntegerField(default=0, verbose_name='优先级')
    is_active = models.BooleanField(default=True, verbose_name='是否启用')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '治疗方案模板'
        verbose_name_plural = '治疗方案模板'
        ordering = ['-priority', '-created_at']
    
    def __str__(self):
        return f"DR{self.dr_grade}级治疗方案模板"


class TreatmentPlan(models.Model):
    """治疗方案模型"""

    STATUS_CHOICES = (
        ('draft', '草稿'),
        ('confirmed', '已确认'),
        ('active', '执行中'),
        ('completed', '已完成'),
        ('cancelled', '已取消'),
    )

    case = models.ForeignKey(
        CaseRecord,
        on_delete=models.CASCADE,
        related_name='treatment_plans',
        verbose_name='病例'
    )
    related_report = models.ForeignKey(
        DiagnosisReport,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='treatment_plans',
        verbose_name='关联报告'
    )

    # 方案基本信息
    plan_number = models.CharField(max_length=100, unique=True, verbose_name='方案编号')
    title = models.CharField(max_length=200, verbose_name='方案标题')
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='draft',
        verbose_name='方案状态'
    )

    # ========== 基础管理目标 ==========
    blood_sugar_target = models.JSONField(
        default=dict,
        blank=True,
        verbose_name='血糖控制目标'
    )
    blood_pressure_target = models.JSONField(
        default=dict,
        blank=True,
        verbose_name='血压控制目标'
    )
    lipid_management = models.TextField(blank=True, verbose_name='血脂管理建议')

    # ========== 眼科治疗 ==========
    treatments = models.JSONField(
        default=list,
        blank=True,
        verbose_name='治疗项目列表',
        help_text='[{"category": "类别", "item": "具体项目", "frequency": "频率", "course": "疗程", "notes": "备注"}]'
    )
    anti_vegf_treatment = models.JSONField(
        default=dict,
        blank=True,
        verbose_name='抗VEGF治疗方案'
    )
    laser_treatment = models.JSONField(
        default=dict,
        blank=True,
        verbose_name='激光治疗方案'
    )
    surgical_treatment = models.TextField(blank=True, verbose_name='手术治疗指征')

    # ========== 药物治疗 ==========
    medications = models.JSONField(
        default=list,
        verbose_name='药物列表',
        help_text='[{"name": "药物名称", "dosage": "用法用量", "frequency": "频次", "duration": "疗程", "category": "类别", "notes": "备注"}]'
    )

    # ========== 生活方式干预 ==========
    diet_guidance = models.TextField(blank=True, verbose_name='饮食指导')
    exercise_guidance = models.TextField(blank=True, verbose_name='运动指导')
    lifestyle_advice = models.TextField(blank=True, verbose_name='综合生活方式建议')

    # ========== 随访监测 ==========
    follow_up_plan = models.JSONField(default=dict, verbose_name='复查计划')
    monitoring_plan = models.JSONField(default=list, blank=True, verbose_name='监测计划')
    warning_symptoms = models.TextField(blank=True, verbose_name='预警症状')

    # ========== 其他 ==========
    precautions = models.TextField(blank=True, verbose_name='注意事项')

    # AI推荐信息
    is_ai_recommended = models.BooleanField(default=False, verbose_name='是否AI推荐')
    ai_recommendation_score = models.FloatField(null=True, blank=True, verbose_name='推荐置信度')
    template_used = models.ForeignKey(
        TreatmentPlanTemplate,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='treatment_plans',
        verbose_name='使用的模板'
    )

    # 医生信息
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_treatment_plans',
        verbose_name='创建医生'
    )
    confirmed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='confirmed_treatment_plans',
        verbose_name='确认医生'
    )
    confirmed_at = models.DateTimeField(null=True, blank=True, verbose_name='确认时间')

    # 时间戳
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    start_date = models.DateField(null=True, blank=True, verbose_name='开始执行日期')
    end_date = models.DateField(null=True, blank=True, verbose_name='预计结束日期')
    
    class Meta:
        verbose_name = '治疗方案'
        verbose_name_plural = '治疗方案'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.plan_number} - {self.title}"
    
    def generate_plan_number(self):
        """生成方案编号"""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        case_id = self.case.id
        return f"TP{timestamp}{case_id:04d}"


class TreatmentPlanExecution(models.Model):
    """方案执行记录"""
    
    plan = models.ForeignKey(
        TreatmentPlan,
        on_delete=models.CASCADE,
        related_name='executions',
        verbose_name='治疗方案'
    )
    execution_date = models.DateField(verbose_name='执行日期')
    medication_taken = models.JSONField(default=dict, verbose_name='用药记录')
    medication_notes = models.TextField(blank=True, verbose_name='用药备注')
    follow_up_completed = models.BooleanField(default=False, verbose_name='是否完成复查')
    patient_feedback = models.TextField(blank=True, verbose_name='患者反馈')
    doctor_notes = models.TextField(blank=True, verbose_name='医生备注')
    
    # 血糖记录
    blood_sugar_fasting = models.CharField(max_length=20, blank=True, verbose_name='空腹血糖(mmol/L)')
    blood_sugar_postprandial = models.CharField(max_length=20, blank=True, verbose_name='餐后血糖(mmol/L)')
    blood_sugar_hba1c = models.CharField(max_length=20, blank=True, verbose_name='糖化血红蛋白(%)')
    
    # 血压记录
    blood_pressure_systolic = models.CharField(max_length=20, blank=True, verbose_name='收缩压(mmHg)')
    blood_pressure_diastolic = models.CharField(max_length=20, blank=True, verbose_name='舒张压(mmHg)')
    
    # 饮食记录
    diet_completed = models.BooleanField(default=False, verbose_name='是否按饮食计划执行')
    diet_notes = models.TextField(blank=True, verbose_name='饮食记录备注')
    
    # 运动记录
    exercise_completed = models.BooleanField(default=False, verbose_name='是否按运动计划执行')
    exercise_notes = models.TextField(blank=True, verbose_name='运动记录备注')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='treatment_executions',
        verbose_name='记录人'
    )
    
    class Meta:
        verbose_name = '方案执行记录'
        verbose_name_plural = '方案执行记录'
        ordering = ['-execution_date', '-created_at']
    
    def __str__(self):
        return f"{self.plan.plan_number} - {self.execution_date}"


class Conversation(models.Model):
    """医患会话"""
    
    patient = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='patient_conversations',
        verbose_name='患者'
    )
    doctor = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='doctor_conversations',
        verbose_name='医生'
    )
    
    # 关联信息
    related_case = models.ForeignKey(
        CaseRecord,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='conversations',
        verbose_name='关联病例'
    )
    
    # 会话状态
    is_active = models.BooleanField(default=True, verbose_name='是否活跃')
    patient_unread_count = models.IntegerField(default=0, verbose_name='患者未读数')
    doctor_unread_count = models.IntegerField(default=0, verbose_name='医生未读数')
    
    # 最后消息
    last_message = models.TextField(blank=True, verbose_name='最后消息内容')
    last_message_at = models.DateTimeField(null=True, blank=True, verbose_name='最后消息时间')
    last_message_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='last_message_conversations',
        verbose_name='最后消息发送者'
    )
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '医患会话'
        verbose_name_plural = '医患会话'
        unique_together = ['patient', 'doctor']
        ordering = ['-last_message_at']
    
    def __str__(self):
        return f"{self.patient.username} - {self.doctor.username}"


class Message(models.Model):
    """消息模型"""
    
    MESSAGE_TYPE_CHOICES = (
        ('text', '文字'),
        ('image', '图片'),
        ('file', '文件'),
        ('system', '系统消息'),
    )
    
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name='会话'
    )
    sender = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='sent_messages',
        verbose_name='发送者'
    )
    
    # 消息内容
    message_type = models.CharField(
        max_length=20,
        choices=MESSAGE_TYPE_CHOICES,
        default='text',
        verbose_name='消息类型'
    )
    content = models.TextField(verbose_name='消息内容')
    file_url = models.CharField(max_length=500, blank=True, verbose_name='文件URL')
    file_name = models.CharField(max_length=200, blank=True, verbose_name='文件名')
    
    # 关联信息
    related_report = models.ForeignKey(
        DiagnosisReport,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='messages',
        verbose_name='关联报告'
    )
    related_treatment_plan = models.ForeignKey(
        TreatmentPlan,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='messages',
        verbose_name='关联治疗方案'
    )
    
    # 消息状态
    is_read = models.BooleanField(default=False, verbose_name='是否已读')
    read_at = models.DateTimeField(null=True, blank=True, verbose_name='已读时间')
    is_important = models.BooleanField(default=False, verbose_name='重要消息')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '消息'
        verbose_name_plural = '消息'
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.sender.username} - {self.get_message_type_display()}"


class MessageTemplate(models.Model):
    """消息模板（医生快捷回复）"""
    
    doctor = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='message_templates',
        verbose_name='医生'
    )
    title = models.CharField(max_length=100, verbose_name='模板标题')
    content = models.TextField(verbose_name='模板内容')
    category = models.CharField(max_length=50, blank=True, verbose_name='分类')
    is_public = models.BooleanField(default=False, verbose_name='是否公开')
    usage_count = models.IntegerField(default=0, verbose_name='使用次数')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '消息模板'
        verbose_name_plural = '消息模板'
        ordering = ['-usage_count', '-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.doctor.username}"
