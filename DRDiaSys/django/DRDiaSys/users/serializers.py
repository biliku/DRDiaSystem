from rest_framework import serializers
from django.contrib.auth.models import User
from .models import PatientInfo, ConditionInfo


class PatientInfoSerializer(serializers.ModelSerializer):
    """患者个人信息序列化器"""
    username = serializers.CharField(source='user.username', read_only=True)
    user_email = serializers.EmailField(source='user.email', read_only=True)
    
    class Meta:
        model = PatientInfo
        fields = [
            'id',
            'user',
            'username',
            'user_email',
            'real_name',
            'gender',
            'birth_date',
            'age',
            'id_card',
            'phone',
            'email',
            'address',
            'province',
            'city',
            'district',
            'blood_type',
            'emergency_contact',
            'emergency_phone',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ('id', 'created_at', 'updated_at', 'user')


class ConditionInfoSerializer(serializers.ModelSerializer):
    """患者病情信息序列化器"""
    username = serializers.CharField(source='user.username', read_only=True)
    real_name = serializers.SerializerMethodField()
    
    class Meta:
        model = ConditionInfo
        fields = [
            'id',
            'user',
            'username',
            'real_name',
            'has_diabetes',
            'diabetes_type',
            'diabetes_duration',
            'blood_sugar_level',
            'hba1c',
            'symptoms',
            'symptom_description',
            'symptom_duration',
            'medical_history',
            'family_history',
            'medication_history',
            'allergy_history',
            'other_conditions',
            'notes',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ('id', 'created_at', 'updated_at', 'user')
    
    def get_real_name(self, obj):
        """获取患者真实姓名"""
        if hasattr(obj.user, 'patient_info') and obj.user.patient_info.real_name:
            return obj.user.patient_info.real_name
        return obj.user.username

