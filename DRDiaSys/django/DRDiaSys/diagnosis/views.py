import os
import threading

from django.conf import settings
from django.db.models import Count
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import DiagnosisTask, DiagnosisReport, CaseRecord, CaseEvent
from .serializers import (
    DiagnosisTaskSerializer,
    DiagnosisReportSerializer,
    CaseRecordSerializer,
    CaseEventSerializer,
)
from .diagnosis_service import perform_lesion_segmentation
from .report_generator import DiagnosisReportGenerator
from datasets.models import PatientImage
from datasets.views import DATASET_ROOT, _is_admin


def _is_doctor(user):
    profile = getattr(user, 'profile', None)
    return getattr(profile, 'role', None) == 'doctor'


def _create_ai_event(case, report, doctor=None, description=None):
    if not report or CaseEvent.objects.filter(case=case, related_report=report).exists():
        return
    CaseEvent.objects.create(
        case=case,
        event_type='ai_report',
        related_report=report,
        description=description or f"报告 {report.report_number} 已确认并归档",
        created_by=doctor if _is_doctor(doctor) else None
    )


def _merge_patient_cases(patient_id):
    cases = list(
        CaseRecord.objects.filter(patient_id=patient_id)
        .select_related('patient', 'primary_report__patient_image')
        .order_by('created_at')
    )
    if not cases:
        return None
    base = cases[0]
    for extra in cases[1:]:
        _create_ai_event(base, extra.primary_report, None, '合并历史病例记录')
        CaseEvent.objects.filter(case=extra).update(case=base)
        extra.delete()
    return base


def _get_or_create_case_for_patient(patient, report, doctor=None):
    case = _merge_patient_cases(patient.id)
    if not case:
        summary = (report.doctor_conclusion or report.ai_summary or '')[:500]
        case = CaseRecord.objects.create(
            patient=patient,
            primary_report=report,
            title=f"{patient.username} 病例",
            summary=summary,
            status='active'
        )
    return case


def _ensure_case_for_report(report, doctor):
    """确保已确认报告有病例记录，并记录AI事件"""
    patient = report.patient_image.owner
    case = _get_or_create_case_for_patient(patient, report, doctor)
    _create_ai_event(case, report, doctor)
    return case


def process_diagnosis_task(task_id):
    """后台处理诊断任务"""
    try:
        task = DiagnosisTask.objects.get(id=task_id)
        task.status = 'processing'
        task.progress = 10
        task.save()
        
        # 获取图像路径
        patient_image = task.patient_image
        image_path = os.path.join(DATASET_ROOT, patient_image.stored_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 创建结果输出目录
        result_dir = os.path.join(
            DATASET_ROOT,
            'diagnosis_results',
            f"patient_{patient_image.owner.id}_{patient_image.owner.username}"
        )
        os.makedirs(result_dir, exist_ok=True)
        
        task.progress = 30
        task.save()
        
        # 执行病灶分割
        result = perform_lesion_segmentation(
            image_path=image_path,
            model_path=task.model_path,
            output_dir=result_dir
        )
        
        if result.get('success'):
            # 更新任务结果
            task.result_image_path = result['result_image_path']
            task.segmentation_mask_path = result['mask_path']
            task.lesion_statistics = result['lesion_statistics']
            task.status = 'completed'
            task.progress = 100
            task.completed_at = timezone.now()
            task.save()
            
            # 自动创建诊断报告
            create_diagnosis_report(task)
        else:
            task.status = 'failed'
            task.error_message = result.get('error', '处理失败')
            task.save()
            
    except Exception as e:
        task = DiagnosisTask.objects.get(id=task_id)
        task.status = 'failed'
        task.error_message = str(e)
        task.save()


def create_diagnosis_report(diagnosis_task):
    """创建诊断报告"""
    try:
        # 检查是否已存在报告
        if hasattr(diagnosis_task, 'report'):
            report = diagnosis_task.report
        else:
            report = DiagnosisReport.objects.create(
                diagnosis_task=diagnosis_task,
                patient_image=diagnosis_task.patient_image
            )
            report.report_number = report.generate_report_number()
            report.save()
        
        # 生成AI摘要
        lesion_stats = diagnosis_task.lesion_statistics or {}
        lesion_summary = []
        for class_id, stat in lesion_stats.items():
            if class_id != 0 and stat.get('percentage', 0) > 0.01:  # 排除背景，且占比>0.01%
                lesion_summary.append({
                    'name': stat.get('name', ''),
                    'percentage': stat.get('percentage', 0)
                })
        
        if lesion_summary:
            report.ai_summary = f"检测到{len(lesion_summary)}种病灶类型"
            report.lesion_summary = lesion_summary
        else:
            report.ai_summary = "未检测到明显病灶"
            report.lesion_summary = []
        
        report.save()
        
        # 生成PDF报告
        try:
            pdf_path = generate_pdf_report(report)
            report.pdf_path = pdf_path
            report.save()
        except Exception as e:
            print(f"PDF生成失败: {e}")
        
        return report
    except Exception as e:
        print(f"创建报告失败: {e}")
        return None


def generate_pdf_report(report):
    """生成PDF报告"""
    generator = DiagnosisReportGenerator(report)
    pdf_path = generator.generate()
    return pdf_path


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_diagnosis_task(request):
    """创建诊断任务"""
    patient_image_id = request.data.get('patient_image_id')
    task_type = request.data.get('task_type', 'lesion_segmentation')
    model_path = request.data.get('model_path')
    
    if not patient_image_id:
        return Response({'message': '请提供患者影像ID'}, status=status.HTTP_400_BAD_REQUEST)
    
    patient_image = get_object_or_404(PatientImage, id=patient_image_id)
    
    # 检查权限：患者只能为自己的影像创建任务，医生和管理员可以为任何影像创建
    if patient_image.owner != request.user and not _is_admin(request.user):
        # 检查是否是医生
        profile = getattr(request.user, 'profile', None)
        if not (profile and profile.role == 'doctor'):
            return Response({'message': '无权为该影像创建诊断任务'}, status=status.HTTP_403_FORBIDDEN)
    
    # 创建诊断任务
    task = DiagnosisTask.objects.create(
        patient_image=patient_image,
        task_type=task_type,
        model_path=model_path,
        status='pending'
    )
    
    # 异步处理任务
    thread = threading.Thread(target=process_diagnosis_task, args=(task.id,))
    thread.daemon = True
    thread.start()
    
    serializer = DiagnosisTaskSerializer(task)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def diagnosis_task_detail(request, task_id):
    """获取诊断任务详情"""
    task = get_object_or_404(DiagnosisTask, id=task_id)
    
    # 检查权限
    if task.patient_image.owner != request.user and not _is_admin(request.user):
        profile = getattr(request.user, 'profile', None)
        if not (profile and profile.role == 'doctor'):
            return Response({'message': '无权访问该诊断任务'}, status=status.HTTP_403_FORBIDDEN)
    
    serializer = DiagnosisTaskSerializer(task)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def diagnosis_task_list(request):
    """获取诊断任务列表"""
    patient_id = request.GET.get('patient_id')
    status_filter = request.GET.get('status')
    
    if _is_admin(request.user):
        # 管理员可以查看所有任务
        queryset = DiagnosisTask.objects.all()
    elif hasattr(request.user, 'profile') and request.user.profile.role == 'doctor':
        # 医生可以查看所有任务
        queryset = DiagnosisTask.objects.all()
    else:
        # 患者只能查看自己的任务
        queryset = DiagnosisTask.objects.filter(patient_image__owner=request.user)
    
    if patient_id:
        queryset = queryset.filter(patient_image__owner_id=patient_id)
    
    if status_filter:
        queryset = queryset.filter(status=status_filter)
    
    queryset = queryset.order_by('-created_at')
    serializer = DiagnosisTaskSerializer(queryset, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def diagnosis_report_list(request):
    """获取诊断报告列表"""
    patient_id = request.GET.get('patient_id')
    status_filter = request.GET.get('status')
    unassigned = request.GET.get('unassigned')
    
    user = request.user

    if _is_admin(user) or _is_doctor(user):
        # 管理员 / 医生：可查看全部报告
        queryset = DiagnosisReport.objects.all()
    else:
        # 患者：仅能查看自己、且已确认的报告
        queryset = DiagnosisReport.objects.filter(
            patient_image__owner=user,
            status='finalized'
        )
    
    if patient_id:
        queryset = queryset.filter(patient_image__owner_id=patient_id)
    
    if status_filter:
        queryset = queryset.filter(status=status_filter)

    if unassigned == 'true' and (_is_admin(user) or _is_doctor(user)):
        queryset = queryset.filter(case_record__isnull=True, status='finalized')
    
    queryset = queryset.order_by('-created_at')
    serializer = DiagnosisReportSerializer(queryset, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def diagnosis_report_detail(request, report_id):
    """获取诊断报告详情"""
    report = get_object_or_404(DiagnosisReport, id=report_id)

    user = request.user
    # 管理员 / 医生：可查看任意报告
    if not (_is_admin(user) or _is_doctor(user)):
        # 患者：仅能查看自己的且已确认的报告
        if report.patient_image.owner != user or report.status != 'finalized':
            return Response({'message': '无权访问该报告'}, status=status.HTTP_403_FORBIDDEN)

    serializer = DiagnosisReportSerializer(report)
    return Response(serializer.data)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_diagnosis_report(request, report_id):
    """管理员删除诊断报告"""
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可删除报告'}, status=status.HTTP_403_FORBIDDEN)

    report = get_object_or_404(DiagnosisReport, id=report_id)

    # 删除PDF文件
    if report.pdf_path and os.path.exists(report.pdf_path):
        try:
            os.remove(report.pdf_path)
        except OSError:
            pass

    report.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def review_diagnosis_report(request, report_id):
    """医生复核诊断报告"""
    report = get_object_or_404(DiagnosisReport, id=report_id)
    
    # 只有医生可以复核
    if not _is_doctor(request.user):
        return Response({'message': '只有医生可以复核报告'}, status=status.HTTP_403_FORBIDDEN)
    
    doctor_notes = request.data.get('doctor_notes', '')
    doctor_conclusion = request.data.get('doctor_conclusion', '')
    report_status = request.data.get('status', 'reviewed')
    
    report.reviewed_by = request.user
    report.reviewed_at = timezone.now()
    report.doctor_notes = doctor_notes
    report.doctor_conclusion = doctor_conclusion
    report.status = report_status
    
    # 如果状态改为已确认，重新生成PDF
    case = None
    if report_status == 'finalized':
        try:
            pdf_path = generate_pdf_report(report)
            report.pdf_path = pdf_path
        except Exception as e:
            print(f"重新生成PDF失败: {e}")
        case = _ensure_case_for_report(report, request.user)
    
    report.save()
    
    serializer = DiagnosisReportSerializer(report)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def download_diagnosis_report(request, report_id):
    """下载诊断报告PDF"""
    report = get_object_or_404(DiagnosisReport, id=report_id)

    user = request.user
    # 管理员 / 医生：可下载任意报告
    if not (_is_admin(user) or _is_doctor(user)):
        # 患者：仅能下载自己的且已确认的报告
        if report.patient_image.owner != user or report.status != 'finalized':
            return Response({'message': '无权下载该报告'}, status=status.HTTP_403_FORBIDDEN)
    
    if not report.pdf_path or not os.path.exists(report.pdf_path):
        return Response({'message': '报告文件不存在'}, status=status.HTTP_404_NOT_FOUND)
    
    return FileResponse(
        open(report.pdf_path, 'rb'),
        content_type='application/pdf',
        filename=f"{report.report_number}.pdf"
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_result_image(request, task_id):
    """获取诊断结果图像"""
    task = get_object_or_404(DiagnosisTask, id=task_id)
    
    # 检查权限：仅管理员/医生
    if not (_is_admin(request.user) or _is_doctor(request.user)):
        return Response({'message': '仅管理员或医生可查看结果'}, status=status.HTTP_403_FORBIDDEN)
    
    if not task.result_image_path or not os.path.exists(task.result_image_path):
        return Response({'message': '结果图像不存在'}, status=status.HTTP_404_NOT_FOUND)
    
    return FileResponse(
        open(task.result_image_path, 'rb'),
        content_type='image/jpeg'
    )


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def case_records(request):
    """医生端病例列表 & 创建"""
    if not _is_doctor(request.user):
        return Response({'message': '仅医生可管理病例'}, status=status.HTTP_403_FORBIDDEN)

    if request.method == 'GET':
        # 自动补建历史已确认报告的病例
        missing_reports = DiagnosisReport.objects.filter(
            status='finalized',
            case_record__isnull=True
        )
        for report in missing_reports:
            reviewer = report.reviewed_by or request.user
            _ensure_case_for_report(report, reviewer)

        # 合并重复病例
        duplicate_patient_ids = (
            CaseRecord.objects.values('patient')
            .annotate(total=Count('id'))
            .filter(total__gt=1)
            .values_list('patient', flat=True)
        )
        for pid in duplicate_patient_ids:
            _merge_patient_cases(pid)

        patient_id = request.GET.get('patient_id')
        status_filter = request.GET.get('status')
        queryset = CaseRecord.objects.select_related(
            'patient',
            'primary_report__patient_image',
            'primary_report__diagnosis_task'
        ).prefetch_related('events__related_report')

        if patient_id:
            queryset = queryset.filter(patient_id=patient_id)

        if status_filter and status_filter != 'all':
            queryset = queryset.filter(status=status_filter)

        serializer = CaseRecordSerializer(queryset, many=True)
        return Response(serializer.data)

    # POST: create new case
    primary_report_id = request.data.get('primary_report_id')
    title = request.data.get('title')
    if not primary_report_id or not title:
        return Response({'message': '请提供病例标题和首份报告'}, status=status.HTTP_400_BAD_REQUEST)

    report = get_object_or_404(DiagnosisReport, pk=primary_report_id)
    if report.status != 'finalized':
        return Response({'message': '仅已确认的报告可建立病例'}, status=status.HTTP_400_BAD_REQUEST)
    if hasattr(report, 'case_record'):
        return Response({'message': '该报告已关联病例'}, status=status.HTTP_400_BAD_REQUEST)

    patient = report.patient_image.owner
    case = CaseRecord.objects.create(
        patient=patient,
        primary_report=report,
        title=title,
        summary=request.data.get('summary', ''),
        status=request.data.get('status', 'active')
    )

    CaseEvent.objects.create(
        case=case,
        event_type='ai_report',
        related_report=report,
        description=request.data.get('initial_event_desc', '首份AI诊断报告已确认并建立病例'),
        created_by=request.user
    )

    serializer = CaseRecordSerializer(case)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET', 'PATCH'])
@permission_classes([IsAuthenticated])
def case_record_detail(request, case_id):
    if not _is_doctor(request.user):
        return Response({'message': '仅医生可管理病例'}, status=status.HTTP_403_FORBIDDEN)

    case = get_object_or_404(
        CaseRecord.objects.select_related(
            'patient',
            'primary_report__patient_image',
            'primary_report__diagnosis_task'
        ).prefetch_related('events__related_report'),
        pk=case_id
    )

    if request.method == 'GET':
        serializer = CaseRecordSerializer(case)
        return Response(serializer.data)

    case.title = request.data.get('title', case.title)
    case.summary = request.data.get('summary', case.summary)
    case.status = request.data.get('status', case.status)
    case.save()
    serializer = CaseRecordSerializer(case)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_case_event(request, case_id):
    if not _is_doctor(request.user):
        return Response({'message': '仅医生可记录病例事件'}, status=status.HTTP_403_FORBIDDEN)

    case = get_object_or_404(CaseRecord, pk=case_id)
    serializer = CaseEventSerializer(data=request.data, context={'case': case})
    serializer.is_valid(raise_exception=True)
    event = serializer.save(case=case, created_by=request.user)
    return Response(CaseEventSerializer(event).data, status=status.HTTP_201_CREATED)
