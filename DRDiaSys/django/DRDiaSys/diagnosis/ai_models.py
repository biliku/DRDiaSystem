from django.db import models
from django.contrib.auth.models import User


class AIModel(models.Model):
    """AI模型管理"""

    MODEL_TYPE_CHOICES = (
        ('grading', '分级模型'),
        ('segmentation', '病变分割模型'),
    )

    DEPLOYMENT_STRATEGY_CHOICES = (
        ('manual', '手动部署'),
        ('automatic', '自动部署'),
        ('canary', '金丝雀发布'),
        ('blue_green', '蓝绿部署'),
    )

    STATUS_CHOICES = (
        ('development', '开发中'),
        ('testing', '测试中'),
        ('production', '生产环境'),
        ('deprecated', '已废弃'),
    )

    name = models.CharField(max_length=100, verbose_name='模型名称')
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES, verbose_name='模型类型')
    version = models.CharField(max_length=50, verbose_name='模型版本')

    # 模型文件信息
    model_path = models.CharField(max_length=500, blank=True, verbose_name='模型文件路径')
    model_size = models.BigIntegerField(default=0, verbose_name='模型大小(字节)')
    config_path = models.CharField(max_length=500, blank=True, verbose_name='配置文件路径')

    # 描述信息
    description = models.TextField(blank=True, verbose_name='模型描述')
    changelog = models.TextField(blank=True, verbose_name='版本更新日志')

    # 性能指标
    accuracy = models.FloatField(null=True, blank=True, verbose_name='准确率')
    precision = models.FloatField(null=True, blank=True, verbose_name='精确率')
    recall = models.FloatField(null=True, blank=True, verbose_name='召回率')
    f1_score = models.FloatField(null=True, blank=True, verbose_name='F1分数')
    auc = models.FloatField(null=True, blank=True, verbose_name='AUC值')

    # 适用场景
    applicable_dr_grades = models.JSONField(default=list, verbose_name='适用DR分级')
    applicable_lesion_types = models.JSONField(default=list, verbose_name='适用病灶类型')

    # 部署相关
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='development', verbose_name='状态')
    is_active = models.BooleanField(default=False, verbose_name='是否激活')
    is_default = models.BooleanField(default=False, verbose_name='是否默认模型')
    deployment_strategy = models.CharField(max_length=20, choices=DEPLOYMENT_STRATEGY_CHOICES, default='manual', verbose_name='部署策略')

    # 推理服务
    inference_endpoint = models.CharField(max_length=500, blank=True, verbose_name='推理服务地址')
    inference_params = models.JSONField(default=dict, blank=True, verbose_name='推理参数')

    # 元数据
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='created_models', verbose_name='创建人')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    deployed_at = models.DateTimeField(null=True, blank=True, verbose_name='部署时间')

    class Meta:
        verbose_name = 'AI模型'
        verbose_name_plural = 'AI模型'
        ordering = ['-created_at']
        unique_together = ['name', 'version']

    def __str__(self):
        return f"{self.name} v{self.version} ({self.get_model_type_display()})"


class ModelPerformanceLog(models.Model):
    """模型性能监控日志"""

    LOG_TYPE_CHOICES = (
        ('inference', '推理性能'),
        ('accuracy', '准确率评估'),
        ('drift', '模型漂移'),
    )

    model = models.ForeignKey(AIModel, on_delete=models.CASCADE, related_name='performance_logs', verbose_name='模型')
    log_type = models.CharField(max_length=20, choices=LOG_TYPE_CHOICES, verbose_name='日志类型')

    # 性能数据
    avg_inference_time = models.FloatField(null=True, blank=True, verbose_name='平均推理时间(ms)')
    p95_inference_time = models.FloatField(null=True, blank=True, verbose_name='P95推理时间(ms)')
    p99_inference_time = models.FloatField(null=True, blank=True, verbose_name='P99推理时间(ms)')
    throughput = models.FloatField(null=True, blank=True, verbose_name='吞吐量(请求/秒)')

    # 准确率数据
    sample_count = models.IntegerField(default=0, verbose_name='样本数量')
    correct_count = models.IntegerField(default=0, verbose_name='正确数量')
    accuracy = models.FloatField(null=True, blank=True, verbose_name='准确率')

    # 预测分布
    prediction_distribution = models.JSONField(default=dict, blank=True, verbose_name='预测分布')

    # 模型漂移
    drift_score = models.FloatField(null=True, blank=True, verbose_name='漂移分数')
    drift_status = models.CharField(max_length=20, blank=True, verbose_name='漂移状态')

    # 附加信息
    additional_info = models.JSONField(default=dict, blank=True, verbose_name='附加信息')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '模型性能日志'
        verbose_name_plural = '模型性能日志'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.model.name} - {self.get_log_type_display()} @ {self.created_at}"


class ModelDeploymentHistory(models.Model):
    """模型部署历史"""

    ACTION_CHOICES = (
        ('deploy', '部署'),
        ('rollback', '回滚'),
        ('activate', '激活'),
        ('deactivate', '停用'),
    )

    model = models.ForeignKey(AIModel, on_delete=models.CASCADE, related_name='deployment_history', verbose_name='模型')
    action = models.CharField(max_length=20, choices=ACTION_CHOICES, verbose_name='操作类型')
    from_version = models.CharField(max_length=50, blank=True, verbose_name='原版本')
    to_version = models.CharField(max_length=50, verbose_name='目标版本')

    # 部署详情
    deployment_strategy = models.CharField(max_length=20, verbose_name='部署策略')
    traffic_percentage = models.IntegerField(default=100, verbose_name='流量比例(%)')
    rollback_available = models.BooleanField(default=True, verbose_name='可回滚')

    # 状态
    status = models.CharField(max_length=20, default='success', verbose_name='状态')

    # 操作者
    operator = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name='操作人')

    # 时间
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='操作时间')
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name='完成时间')

    class Meta:
        verbose_name = '模型部署历史'
        verbose_name_plural = '模型部署历史'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.model.name} - {self.get_action_display()} to v{self.to_version}"
