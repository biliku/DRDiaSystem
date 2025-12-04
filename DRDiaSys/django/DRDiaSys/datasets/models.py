from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
    STATUS_CHOICES = [
        ('unprocessed', '原始'),
        ('processing', '处理中'),
        ('processed', '已处理'),
    ]

    TYPE_CHOICES = [
        ('public', '公开'),
        ('clinical', '临床'),
    ]

    name = models.CharField(max_length=255, unique=True, verbose_name='数据集名称')
    type = models.CharField(max_length=50, choices=TYPE_CHOICES, blank=True, null=True, verbose_name='数据集类型')
    description = models.TextField(blank=True, null=True, verbose_name='描述信息')
    image_count = models.IntegerField(default=0, verbose_name='图片数量')
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='unprocessed', verbose_name='处理状态')
    path = models.CharField(max_length=500, verbose_name='存储路径') # 存储相对于DATASET_ROOT的路径
    version = models.CharField(max_length=50, blank=True, null=True, verbose_name='版本信息')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '数据集'
        verbose_name_plural = '数据集'
        ordering = ['-created_at']

    def __str__(self):
        return self.name 


class PatientImage(models.Model):
    """患者自主影像记录，归档至数据集目录便于管理员集中管理"""

    EYE_CHOICES = (
        ('left', '左眼'),
        ('right', '右眼'),
        ('both', '双眼'),
        ('unknown', '未知'),
    )

    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='patient_images',
        verbose_name='所属患者'
    )
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='uploaded_patient_images',
        verbose_name='上传者'
    )
    original_name = models.CharField(max_length=255, verbose_name='原始文件名')
    stored_path = models.CharField(
        max_length=500,
        verbose_name='相对存储路径'
    )
    dataset_folder = models.CharField(
        max_length=255,
        default='patient_uploads',
        verbose_name='所属数据集目录'
    )
    eye_side = models.CharField(
        max_length=10,
        choices=EYE_CHOICES,
        default='unknown',
        verbose_name='眼别'
    )
    description = models.TextField(blank=True, null=True, verbose_name='备注')
    file_size = models.BigIntegerField(default=0, verbose_name='文件大小(Byte)')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='上传时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '患者影像'
        verbose_name_plural = '患者影像'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.owner.username} - {self.original_name}"

    @property
    def relative_path(self):
        """返回相对 datasets 根目录的路径"""
        return self.stored_path