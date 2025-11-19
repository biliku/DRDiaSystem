from django.db import models

class Dataset(models.Model):
    STATUS_CHOICES = [
        ('unprocessed', '原始'),
        ('processing', '处理中'),
        ('processed', '已处理'),
    ]

    TYPE_CHOICES = [
        ('public', '公开'),
        ('clinical', '临床'),
        ('private', '私有项目'),
    ]

    name = models.CharField(max_length=255, unique=True, verbose_name='数据集名称')
    type = models.CharField(max_length=50, choices=TYPE_CHOICES, default='public', verbose_name='数据集类型')
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