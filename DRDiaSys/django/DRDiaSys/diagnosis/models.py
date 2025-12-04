from django.db import models
from django.contrib.auth.models import User
from datasets.models import PatientImage


class DiagnosisTask(models.Model):
    """AI诊断任务模型"""
    
    STATUS_CHOICES = (
        ('pending', '待处理'),
        ('processing', '处理中'),
        ('completed', '已完成'),
        ('failed', '失败'),
    )
    
    TASK_TYPE_CHOICES = (
        ('lesion_segmentation', '病灶分割'),
        ('dr_grading', 'DR分级'),
        ('both', '病灶分割+DR分级'),
    )
    
    patient_image = models.ForeignKey(
        PatientImage,
        on_delete=models.CASCADE,
        related_name='diagnosis_tasks',
        verbose_name='患者影像'
    )
    task_type = models.CharField(
        max_length=20,
        choices=TASK_TYPE_CHOICES,
        default='lesion_segmentation',
        verbose_name='任务类型'
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name='任务状态'
    )
    progress = models.IntegerField(default=0, verbose_name='处理进度(0-100)')
    error_message = models.TextField(blank=True, null=True, verbose_name='错误信息')
    
    # 模型配置
    model_path = models.CharField(max_length=500, blank=True, null=True, verbose_name='模型路径')
    model_version = models.CharField(max_length=50, blank=True, null=True, verbose_name='模型版本')
    
    # 结果路径
    result_image_path = models.CharField(max_length=500, blank=True, null=True, verbose_name='结果图像路径')
    segmentation_mask_path = models.CharField(max_length=500, blank=True, null=True, verbose_name='分割掩码路径')
    
    # 诊断结果数据（JSON格式存储详细结果）
    lesion_statistics = models.JSONField(default=dict, blank=True, verbose_name='病灶统计信息')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    completed_at = models.DateTimeField(blank=True, null=True, verbose_name='完成时间')
    
    class Meta:
        verbose_name = '诊断任务'
        verbose_name_plural = '诊断任务'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.patient_image.owner.username} - {self.task_type} - {self.status}"


class DiagnosisReport(models.Model):
    """诊断报告模型"""
    
    STATUS_CHOICES = (
        ('draft', '草稿'),
        ('pending_review', '待复核'),
        ('reviewed', '已复核'),
        ('finalized', '已确认'),
    )
    
    diagnosis_task = models.OneToOneField(
        DiagnosisTask,
        on_delete=models.CASCADE,
        related_name='report',
        verbose_name='诊断任务'
    )
    patient_image = models.ForeignKey(
        PatientImage,
        on_delete=models.CASCADE,
        related_name='reports',
        verbose_name='患者影像'
    )
    
    # 报告基本信息
    report_number = models.CharField(max_length=100, unique=True, verbose_name='报告编号')
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='draft',
        verbose_name='报告状态'
    )
    
    # AI诊断结果摘要
    ai_summary = models.TextField(blank=True, null=True, verbose_name='AI诊断摘要')
    lesion_summary = models.JSONField(default=dict, blank=True, verbose_name='病灶摘要')
    
    # 医生复核信息
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_reports',
        verbose_name='复核医生'
    )
    doctor_notes = models.TextField(blank=True, null=True, verbose_name='医生备注')
    doctor_conclusion = models.TextField(blank=True, null=True, verbose_name='医生结论')
    
    # PDF报告路径
    pdf_path = models.CharField(max_length=500, blank=True, null=True, verbose_name='PDF报告路径')
    
    # 时间戳
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    reviewed_at = models.DateTimeField(blank=True, null=True, verbose_name='复核时间')
    
    class Meta:
        verbose_name = '诊断报告'
        verbose_name_plural = '诊断报告'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.report_number} - {self.patient_image.owner.username}"
    
    def generate_report_number(self):
        """生成报告编号"""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        patient_id = self.patient_image.owner.id
        return f"DR{timestamp}{patient_id:04d}"


class CaseRecord(models.Model):
    """病例记录（医生端管理）"""

    STATUS_CHOICES = (
        ('active', '进行中'),
        ('closed', '已结束'),
        ('archived', '已归档'),
    )

    patient = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='case_records',
        verbose_name='患者'
    )
    primary_report = models.OneToOneField(
        DiagnosisReport,
        on_delete=models.CASCADE,
        related_name='case_record',
        verbose_name='首份报告'
    )
    title = models.CharField(max_length=100, verbose_name='病例标题')
    summary = models.TextField(blank=True, verbose_name='病例摘要')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active', verbose_name='病例状态')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '病例记录'
        verbose_name_plural = '病例记录'
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.title} - {self.patient.username}"


class CaseEvent(models.Model):
    """病例事件 / 随访记录"""

    EVENT_CHOICES = (
        ('ai_report', 'AI诊断报告'),
        ('doctor_note', '医生记录'),
        ('follow_up', '随访计划'),
        ('treatment', '治疗方案'),
    )

    case = models.ForeignKey(
        CaseRecord,
        on_delete=models.CASCADE,
        related_name='events',
        verbose_name='病例'
    )
    event_type = models.CharField(max_length=30, choices=EVENT_CHOICES, default='doctor_note', verbose_name='事件类型')
    related_report = models.ForeignKey(
        DiagnosisReport,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='case_events',
        verbose_name='关联报告'
    )
    description = models.TextField(verbose_name='事件描述')
    next_followup_date = models.DateField(null=True, blank=True, verbose_name='下次随访时间')
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_case_events',
        verbose_name='记录医生'
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '病例事件'
        verbose_name_plural = '病例事件'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.case.title} - {self.get_event_type_display()}"
