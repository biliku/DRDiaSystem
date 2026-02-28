from rest_framework import serializers
from .models import DiagnosisTask, DiagnosisReport, CaseRecord, CaseEvent
from .ai_models import AIModel, ModelPerformanceLog, ModelDeploymentHistory
from datasets.serializers import PatientImageSerializer
from treatment.models import TreatmentPlan


class DiagnosisTaskSerializer(serializers.ModelSerializer):
    patient_image_info = PatientImageSerializer(source='patient_image', read_only=True)
    patient_name = serializers.CharField(source='patient_image.owner.username', read_only=True)
    
    class Meta:
        model = DiagnosisTask
        fields = [
            'id',
            'patient_image',
            'patient_image_info',
            'patient_name',
            'task_type',
            'status',
            'progress',
            'error_message',
            'model_path',
            'model_version',
            'result_image_path',
            'segmentation_mask_path',
            'lesion_statistics',
            'dr_grade_result',
            'created_at',
            'updated_at',
            'completed_at',
        ]
        read_only_fields = [
            'id',
            'status',
            'progress',
            'error_message',
            'result_image_path',
            'segmentation_mask_path',
            'lesion_statistics',
            'dr_grade_result',
            'created_at',
            'updated_at',
            'completed_at',
        ]


class DiagnosisReportSerializer(serializers.ModelSerializer):
    diagnosis_task_info = DiagnosisTaskSerializer(source='diagnosis_task', read_only=True)
    patient_image_info = PatientImageSerializer(source='patient_image', read_only=True)
    patient_name = serializers.CharField(source='patient_image.owner.username', read_only=True)
    reviewed_by_name = serializers.CharField(source='reviewed_by.username', read_only=True)
    case_id = serializers.IntegerField(source='case_record.id', read_only=True)
    
    class Meta:
        model = DiagnosisReport
        fields = [
            'id',
            'diagnosis_task',
            'diagnosis_task_info',
            'patient_image',
            'patient_image_info',
            'patient_name',
            'report_number',
            'status',
            'ai_summary',
            'lesion_summary',
            'dr_grade_result',
            'reviewed_by',
            'reviewed_by_name',
            'doctor_notes',
            'doctor_conclusion',
            'pdf_path',
            'case_id',
            'created_at',
            'updated_at',
            'reviewed_at',
        ]
        read_only_fields = [
            'id',
            'report_number',
            'pdf_path',
            'created_at',
            'updated_at',
            'reviewed_at',
            'case_id',
        ]


class TreatmentPlanSimpleSerializer(serializers.ModelSerializer):
    """简化的治疗方案序列化器，用于嵌套显示"""
    class Meta:
        model = TreatmentPlan
        fields = ['id', 'plan_number', 'title', 'status', 'created_at']


class CaseEventSerializer(serializers.ModelSerializer):
    related_report = DiagnosisReportSerializer(read_only=True)
    related_report_id = serializers.PrimaryKeyRelatedField(
        queryset=DiagnosisReport.objects.filter(status='finalized'),
        source='related_report',
        write_only=True,
        allow_null=True,
        required=False
    )
    related_plan = TreatmentPlanSimpleSerializer(read_only=True)
    related_plan_id = serializers.PrimaryKeyRelatedField(
        queryset=TreatmentPlan.objects.all(),
        source='related_plan',
        write_only=True,
        allow_null=True,
        required=False
    )
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)

    class Meta:
        model = CaseEvent
        fields = [
            'id',
            'event_type',
            'description',
            'related_report',
            'related_report_id',
            'related_plan',
            'related_plan_id',
            'next_followup_date',
            'created_by',
            'created_by_name',
            'created_at',
        ]
        read_only_fields = ['id', 'related_report', 'related_plan', 'created_by', 'created_by_name', 'created_at']

    def validate(self, attrs):
        case = self.context.get('case')
        report = attrs.get('related_report')
        if report and report.patient_image.owner_id != case.patient_id:
            raise serializers.ValidationError('关联的报告与当前病例患者不一致')
        return attrs

    def create(self, validated_data):
        return CaseEvent.objects.create(**validated_data)


class CaseRecordSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.username', read_only=True)
    primary_report_info = DiagnosisReportSerializer(source='primary_report', read_only=True)
    events = CaseEventSerializer(many=True, read_only=True)

    class Meta:
        model = CaseRecord
        fields = [
            'id',
            'patient',
            'patient_name',
            'primary_report',
            'primary_report_info',
            'title',
            'summary',
            'status',
            'created_at',
            'updated_at',
            'events',
        ]


# ==================== AI模型管理序列化器 ====================

class AIModelSerializer(serializers.ModelSerializer):
    model_type_name = serializers.CharField(source='get_model_type_display', read_only=True)
    status_name = serializers.CharField(source='get_status_display', read_only=True)
    deployment_strategy_name = serializers.CharField(source='get_deployment_strategy_display', read_only=True)
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    model_size_mb = serializers.SerializerMethodField()

    class Meta:
        model = AIModel
        fields = [
            'id', 'name', 'model_type', 'model_type_name', 'version',
            'model_path', 'model_size', 'model_size_mb', 'config_path',
            'description', 'changelog',
            'accuracy', 'precision', 'recall', 'f1_score', 'auc',
            'applicable_dr_grades', 'applicable_lesion_types',
            'status', 'status_name', 'is_active', 'is_default',
            'deployment_strategy', 'deployment_strategy_name',
            'inference_endpoint', 'inference_params',
            'created_by', 'created_by_name', 'created_at', 'updated_at', 'deployed_at',
        ]
        read_only_fields = ['created_at', 'updated_at']

    def get_model_size_mb(self, obj):
        if obj.model_size:
            return round(obj.model_size / (1024 * 1024), 2)
        return 0


class AIModelListSerializer(serializers.ModelSerializer):
    """简化版的模型列表序列化器"""
    model_type_name = serializers.CharField(source='get_model_type_display', read_only=True)
    status_name = serializers.CharField(source='get_status_display', read_only=True)
    model_size_mb = serializers.SerializerMethodField()

    class Meta:
        model = AIModel
        fields = [
            'id', 'name', 'model_type', 'model_type_name', 'version',
            'status', 'status_name', 'is_active', 'is_default',
            'accuracy', 'model_size_mb', 'deployed_at', 'created_at',
        ]

    def get_model_size_mb(self, obj):
        if obj.model_size:
            return round(obj.model_size / (1024 * 1024), 2)
        return 0


class ModelPerformanceLogSerializer(serializers.ModelSerializer):
    model_name = serializers.CharField(source='model.name', read_only=True)
    log_type_name = serializers.CharField(source='get_log_type_display', read_only=True)

    class Meta:
        model = ModelPerformanceLog
        fields = [
            'id', 'model', 'model_name', 'log_type', 'log_type_name',
            'avg_inference_time', 'p95_inference_time', 'p99_inference_time',
            'throughput', 'sample_count', 'correct_count', 'accuracy',
            'prediction_distribution', 'drift_score', 'drift_status',
            'additional_info', 'created_at',
        ]


class ModelDeploymentHistorySerializer(serializers.ModelSerializer):
    model_name = serializers.CharField(source='model.name', read_only=True)
    action_name = serializers.CharField(source='get_action_display', read_only=True)
    operator_name = serializers.CharField(source='operator.username', read_only=True)

    class Meta:
        model = ModelDeploymentHistory
        fields = [
            'id', 'model', 'model_name', 'action', 'action_name',
            'from_version', 'to_version', 'deployment_strategy',
            'traffic_percentage', 'rollback_available', 'status',
            'operator', 'operator_name', 'created_at', 'completed_at',
        ]
        read_only_fields = ['id', 'patient', 'primary_report', 'created_at', 'updated_at', 'events']
