from rest_framework import serializers
from .models import DiagnosisTask, DiagnosisReport, CaseRecord, CaseEvent
from datasets.serializers import PatientImageSerializer


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


class CaseEventSerializer(serializers.ModelSerializer):
    related_report = DiagnosisReportSerializer(read_only=True)
    related_report_id = serializers.PrimaryKeyRelatedField(
        queryset=DiagnosisReport.objects.filter(status='finalized'),
        source='related_report',
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
            'next_followup_date',
            'created_by',
            'created_by_name',
            'created_at',
        ]
        read_only_fields = ['id', 'related_report', 'created_by', 'created_by_name', 'created_at']

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
        read_only_fields = ['id', 'patient', 'primary_report', 'created_at', 'updated_at', 'events']
