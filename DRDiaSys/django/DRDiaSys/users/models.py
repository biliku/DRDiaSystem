from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    ROLE_CHOICES = (
        ('admin', '管理員'),
        ('doctor', '医生'),
        ('patient', '患者'),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='patient')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()}"

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        # 根據用戶名判斷角色
        role = 'patient'  # 默認角色
        if 'admin' in instance.username.lower():
            role = 'admin'
        elif 'doctor' in instance.username.lower():
            role = 'doctor'
        
        UserProfile.objects.create(user=instance, role=role)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()


class PatientInfo(models.Model):
    """患者个人信息模型"""
    GENDER_CHOICES = (
        ('M', '男'),
        ('F', '女'),
        ('O', '其他'),
    )
    
    BLOOD_TYPE_CHOICES = (
        ('A', 'A型'),
        ('B', 'B型'),
        ('AB', 'AB型'),
        ('O', 'O型'),
        ('UNKNOWN', '未知'),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='patient_info', verbose_name='用户')
    
    # 基本信息
    real_name = models.CharField(max_length=50, blank=True, null=True, verbose_name='真实姓名')
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, blank=True, null=True, verbose_name='性别')
    birth_date = models.DateField(blank=True, null=True, verbose_name='出生日期')
    age = models.IntegerField(blank=True, null=True, verbose_name='年龄')
    id_card = models.CharField(max_length=18, blank=True, null=True, verbose_name='身份证号')
    phone = models.CharField(max_length=20, blank=True, null=True, verbose_name='联系电话')
    email = models.EmailField(blank=True, null=True, verbose_name='邮箱')
    
    # 地址信息
    address = models.TextField(blank=True, null=True, verbose_name='详细地址')
    province = models.CharField(max_length=50, blank=True, null=True, verbose_name='省份')
    city = models.CharField(max_length=50, blank=True, null=True, verbose_name='城市')
    district = models.CharField(max_length=50, blank=True, null=True, verbose_name='区县')
    
    # 医疗信息
    blood_type = models.CharField(max_length=10, choices=BLOOD_TYPE_CHOICES, blank=True, null=True, verbose_name='血型')
    emergency_contact = models.CharField(max_length=50, blank=True, null=True, verbose_name='紧急联系人')
    emergency_phone = models.CharField(max_length=20, blank=True, null=True, verbose_name='紧急联系电话')
    
    # 时间戳
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '患者信息'
        verbose_name_plural = '患者信息'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.real_name or self.user.username} - 患者信息"


class ConditionInfo(models.Model):
    """患者病情信息模型"""
    DIABETES_TYPE_CHOICES = (
        ('TYPE1', '1型糖尿病'),
        ('TYPE2', '2型糖尿病'),
        ('GESTATIONAL', '妊娠期糖尿病'),
        ('OTHER', '其他类型'),
        ('NONE', '无糖尿病'),
    )
    
    SYMPTOM_CHOICES = (
        ('BLURRED_VISION', '视力模糊'),
        ('FLOATERS', '飞蚊症'),
        ('DARK_SPOTS', '暗点'),
        ('POOR_NIGHT_VISION', '夜视能力差'),
        ('COLOR_VISION_LOSS', '色觉减退'),
        ('NONE', '无明显症状'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='condition_info', verbose_name='用户')
    
    # 糖尿病相关信息
    has_diabetes = models.BooleanField(default=False, verbose_name='是否患有糖尿病')
    diabetes_type = models.CharField(max_length=20, choices=DIABETES_TYPE_CHOICES, blank=True, null=True, verbose_name='糖尿病类型')
    diabetes_duration = models.IntegerField(blank=True, null=True, verbose_name='糖尿病病程（年）')
    blood_sugar_level = models.FloatField(blank=True, null=True, verbose_name='血糖水平（mmol/L）')
    hba1c = models.FloatField(blank=True, null=True, verbose_name='糖化血红蛋白（%）')
    
    # 症状信息
    symptoms = models.JSONField(default=list, blank=True, verbose_name='症状列表')
    symptom_description = models.TextField(blank=True, null=True, verbose_name='症状详细描述')
    symptom_duration = models.CharField(max_length=100, blank=True, null=True, verbose_name='症状持续时间')
    
    # 病史信息
    medical_history = models.TextField(blank=True, null=True, verbose_name='既往病史')
    family_history = models.TextField(blank=True, null=True, verbose_name='家族病史')
    medication_history = models.TextField(blank=True, null=True, verbose_name='用药史')
    allergy_history = models.TextField(blank=True, null=True, verbose_name='过敏史')
    
    # 其他信息
    other_conditions = models.TextField(blank=True, null=True, verbose_name='其他疾病')
    notes = models.TextField(blank=True, null=True, verbose_name='备注')
    
    # 时间戳
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '病情信息'
        verbose_name_plural = '病情信息'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - 病情信息 ({self.created_at.strftime('%Y-%m-%d')})"
