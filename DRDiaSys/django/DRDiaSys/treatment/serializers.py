from rest_framework import serializers
from django.contrib.auth.models import User
from .models import (
    TreatmentPlan,
    TreatmentPlanExecution,
    TreatmentPlanTemplate,
    Conversation,
    Message,
    MessageTemplate
)
from diagnosis.serializers import CaseRecordSerializer, DiagnosisReportSerializer


class TreatmentPlanTemplateSerializer(serializers.ModelSerializer):
    """治疗方案模板序列化器"""

    class Meta:
        model = TreatmentPlanTemplate
        fields = [
            'id',
            'dr_grade',
            'lesion_types',
            'diabetes_type',
            # 基础管理目标
            'blood_sugar_target',
            'blood_pressure_target',
            'lipid_management',
            # 眼科治疗
            'anti_vegf_treatment',
            'laser_treatment',
            'surgical_treatment',
            # 药物治疗
            'medications',
            # 生活方式干预
            'diet_guidance',
            'exercise_guidance',
            'lifestyle_advice',
            # 随访监测
            'follow_up_plan',
            'monitoring_plan',
            'warning_symptoms',
            # 其他
            'precautions',
            'priority',
            'is_active',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class TreatmentPlanSerializer(serializers.ModelSerializer):
    """治疗方案序列化器"""

    case_info = CaseRecordSerializer(source='case', read_only=True)
    related_report_info = DiagnosisReportSerializer(source='related_report', read_only=True)
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    confirmed_by_name = serializers.CharField(source='confirmed_by.username', read_only=True)
    template_info = TreatmentPlanTemplateSerializer(source='template_used', read_only=True)
    execution_count = serializers.IntegerField(source='executions.count', read_only=True)

    class Meta:
        model = TreatmentPlan
        fields = [
            'id',
            'case',
            'case_info',
            'related_report',
            'related_report_info',
            'plan_number',
            'title',
            'status',
            # 基础管理目标
            'blood_sugar_target',
            'blood_pressure_target',
            'lipid_management',
            # 眼科治疗
            'treatments',
            'anti_vegf_treatment',
            'laser_treatment',
            'surgical_treatment',
            # 药物治疗
            'medications',
            # 生活方式干预
            'diet_guidance',
            'exercise_guidance',
            'lifestyle_advice',
            # 随访监测
            'follow_up_plan',
            'monitoring_plan',
            'warning_symptoms',
            # 其他
            'precautions',
            # AI推荐信息
            'is_ai_recommended',
            'ai_recommendation_score',
            'template_used',
            'template_info',
            # 医生信息
            'created_by',
            'created_by_name',
            'confirmed_by',
            'confirmed_by_name',
            'confirmed_at',
            # 时间戳
            'created_at',
            'updated_at',
            'start_date',
            'end_date',
            'execution_count',
        ]
        read_only_fields = [
            'id',
            'plan_number',
            'created_at',
            'updated_at',
            'confirmed_at',
        ]


class TreatmentPlanExecutionSerializer(serializers.ModelSerializer):
    """方案执行记录序列化器"""
    
    plan_info = TreatmentPlanSerializer(source='plan', read_only=True)
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = TreatmentPlanExecution
        fields = [
            'id',
            'plan',
            'plan_info',
            'execution_date',
            'medication_taken',
            'medication_notes',
            'follow_up_completed',
            'patient_feedback',
            'doctor_notes',
            # 血糖记录
            'blood_sugar_fasting',
            'blood_sugar_postprandial',
            'blood_sugar_hba1c',
            # 血压记录
            'blood_pressure_systolic',
            'blood_pressure_diastolic',
            # 饮食记录
            'diet_completed',
            'diet_notes',
            # 运动记录
            'exercise_completed',
            'exercise_notes',
            # 记录信息
            'created_by',
            'created_by_name',
            'created_at',
        ]
        read_only_fields = ['id', 'plan', 'created_at', 'created_by', 'created_by_name']


class MessageSerializer(serializers.ModelSerializer):
    """消息序列化器"""
    
    sender_name = serializers.CharField(source='sender.username', read_only=True)
    related_report_info = DiagnosisReportSerializer(source='related_report', read_only=True)
    related_treatment_plan_info = TreatmentPlanSerializer(source='related_treatment_plan', read_only=True)
    
    class Meta:
        model = Message
        fields = [
            'id',
            'conversation',
            'sender',
            'sender_name',
            'message_type',
            'content',
            'file_url',
            'file_name',
            'related_report',
            'related_report_info',
            'related_treatment_plan',
            'related_treatment_plan_info',
            'is_read',
            'read_at',
            'is_important',
            'created_at',
        ]
        read_only_fields = [
            'id',
            'conversation',
            'sender',
            'sender_name',
            'related_report_info',
            'related_treatment_plan_info',
            'is_read',
            'read_at',
            'created_at',
        ]


class ConversationSerializer(serializers.ModelSerializer):
    """会话序列化器"""
    
    patient_name = serializers.CharField(source='patient.username', read_only=True)
    doctor_name = serializers.CharField(source='doctor.username', read_only=True)
    related_case_info = CaseRecordSerializer(source='related_case', read_only=True)
    last_message_by_name = serializers.CharField(source='last_message_by.username', read_only=True)
    message_count = serializers.IntegerField(source='messages.count', read_only=True)
    
    class Meta:
        model = Conversation
        fields = [
            'id',
            'patient',
            'patient_name',
            'doctor',
            'doctor_name',
            'related_case',
            'related_case_info',
            'is_active',
            'patient_unread_count',
            'doctor_unread_count',
            'last_message',
            'last_message_at',
            'last_message_by',
            'last_message_by_name',
            'message_count',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'patient_unread_count',
            'doctor_unread_count',
            'last_message',
            'last_message_at',
            'last_message_by',
            'created_at',
            'updated_at',
        ]


class ConversationDetailSerializer(ConversationSerializer):
    """会话详情序列化器（包含消息列表）"""
    
    messages = MessageSerializer(many=True, read_only=True)
    
    class Meta(ConversationSerializer.Meta):
        fields = ConversationSerializer.Meta.fields + ['messages']


class MessageTemplateSerializer(serializers.ModelSerializer):
    """消息模板序列化器"""
    
    doctor_name = serializers.CharField(source='doctor.username', read_only=True)
    
    class Meta:
        model = MessageTemplate
        fields = [
            'id',
            'doctor',
            'doctor_name',
            'title',
            'content',
            'category',
            'is_public',
            'usage_count',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'usage_count', 'created_at', 'updated_at']

