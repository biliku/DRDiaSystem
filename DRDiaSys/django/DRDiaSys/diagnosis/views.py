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
from .ai_models import AIModel, ModelPerformanceLog, ModelDeploymentHistory
from .serializers import (
    DiagnosisTaskSerializer,
    DiagnosisReportSerializer,
    CaseRecordSerializer,
    CaseEventSerializer,
    AIModelSerializer,
    AIModelListSerializer,
    ModelPerformanceLogSerializer,
    ModelDeploymentHistorySerializer,
)
from .diagnosis_service import perform_lesion_segmentation, perform_dr_grading
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
        
        # 根据任务类型执行相应的诊断
        task_type = task.task_type
        
        # 用于存储结果
        segmentation_result = None
        dr_grade_result = None
        
        # 执行病灶分割（如果是分割任务或综合任务）
        if task_type in ['lesion_segmentation', 'both']:
            segmentation_result = perform_lesion_segmentation(
                image_path=image_path,
                model_path=task.model_path,
                output_dir=result_dir
            )
        
        # 执行DR分级（如果是分级任务或综合任务）
        if task_type in ['dr_grading', 'both']:
            dr_grade_result = perform_dr_grading(
                image_path=image_path,
                model_path=None,  # 使用默认的DR分级模型
                output_dir=result_dir
            )
        
        # 检查是否有成功的任务
        success = False
        if task_type == 'lesion_segmentation':
            success = segmentation_result and segmentation_result.get('success')
        elif task_type == 'dr_grading':
            success = dr_grade_result and dr_grade_result.get('success')
        elif task_type == 'both':
            success = (segmentation_result and segmentation_result.get('success')) or \
                      (dr_grade_result and dr_grade_result.get('success'))
        
        if success:
            # 更新任务结果 - 分割结果
            if segmentation_result and segmentation_result.get('success'):
                task.result_image_path = segmentation_result['result_image_path']
                task.segmentation_mask_path = segmentation_result['mask_path']
                task.lesion_statistics = segmentation_result['lesion_statistics']
            
            # 更新任务结果 - 分级结果
            if dr_grade_result and dr_grade_result.get('success'):
                # 移除不需要保存到数据库的临时路径
                grade_result_copy = dr_grade_result.copy()
                grade_result_copy.pop('result_json_path', None)
                task.dr_grade_result = grade_result_copy
            
            task.status = 'completed'
            task.progress = 100
            task.completed_at = timezone.now()
            task.save()
            
            # 自动创建诊断报告
            create_diagnosis_report(task)
        else:
            task.status = 'failed'
            task.error_message = '处理失败'
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
        
        # 保存DR分级结果
        dr_grade = diagnosis_task.dr_grade_result
        if dr_grade:
            report.dr_grade_result = dr_grade
            
            # 更新AI摘要，包含分级信息
            grade_class_name = dr_grade.get('class_name', '')
            grade_confidence = dr_grade.get('confidence', 0)
            if report.ai_summary:
                report.ai_summary += f"；DR分级为{grade_class_name}（置信度{grade_confidence*100:.1f}%）"
            else:
                report.ai_summary = f"DR分级为{grade_class_name}（置信度{grade_confidence*100:.1f}%）"
        
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
        ).prefetch_related('events__related_report', 'events__related_plan')

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
        ).prefetch_related('events__related_report', 'events__related_plan'),
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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def diagnosis_statistics(request):
    """获取诊断相关统计数据"""
    from django.db.models import Count, Q
    from django.utils import timezone
    from datetime import timedelta

    # 诊断任务统计
    task_stats = DiagnosisTask.objects.aggregate(
        total=Count('id'),
        pending=Count('id', filter=Q(status='pending')),
        processing=Count('id', filter=Q(status='processing')),
        completed=Count('id', filter=Q(status='completed')),
        failed=Count('id', filter=Q(status='failed'))
    )

    # 诊断报告统计
    report_stats = DiagnosisReport.objects.aggregate(
        total=Count('id'),
        draft=Count('id', filter=Q(status='draft')),
        confirmed=Count('id', filter=Q(status='confirmed')),
        archived=Count('id', filter=Q(status='archived'))
    )

    # 病例统计
    case_stats = CaseRecord.objects.aggregate(
        total=Count('id'),
        active=Count('id', filter=Q(status='active')),
        closed=Count('id', filter=Q(status='closed'))
    )

    # 病灶类型统计
    lesion_stats = {}
    for task in DiagnosisTask.objects.filter(lesion_statistics__isnull=False).exclude(lesion_statistics={}):
        task_lesions = task.lesion_statistics or {}
        for lesion_type, stats in task_lesions.items():
            if lesion_type not in lesion_stats:
                lesion_stats[lesion_type] = {'count': 0, 'total_pixels': 0}
            lesion_stats[lesion_type]['count'] += 1
            lesion_stats[lesion_type]['total_pixels'] += stats.get('pixel_count', 0)

    # 最近7天诊断趋势
    seven_days_ago = timezone.now() - timedelta(days=7)
    daily_tasks = []
    for i in range(7):
        day = seven_days_ago + timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        count = DiagnosisTask.objects.filter(created_at__gte=day_start, created_at__lt=day_end).count()
        daily_tasks.append({
            'date': day.strftime('%m-%d'),
            'count': count
        })

    # 最近7天报告趋势
    daily_reports = []
    for i in range(7):
        day = seven_days_ago + timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        count = DiagnosisReport.objects.filter(created_at__gte=day_start, created_at__lt=day_end).count()
        daily_reports.append({
            'date': day.strftime('%m-%d'),
            'count': count
        })

    return Response({
        'tasks': task_stats,
        'reports': report_stats,
        'cases': case_stats,
        'lesion_types': lesion_stats,
        'daily_tasks': daily_tasks,
        'daily_reports': daily_reports
    })


# ==================== AI模型管理 ====================

def _is_admin(user):
    """检查用户是否为管理员"""
    profile = getattr(user, 'profile', None)
    return getattr(profile, 'role', None) == 'admin'


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def ai_models(request):
    """获取或创建AI模型"""
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可管理AI模型'}, status=status.HTTP_403_FORBIDDEN)

    if request.method == 'GET':
        model_type = request.query_params.get('model_type')
        status_filter = request.query_params.get('status')
        is_active = request.query_params.get('is_active')

        queryset = AIModel.objects.all()

        if model_type:
            queryset = queryset.filter(model_type=model_type)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')

        serializer = AIModelListSerializer(queryset, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = AIModelSerializer(data=request.data)
        if serializer.is_valid():
            model = serializer.save(created_by=request.user)

            # 如果设为默认模型，取消其他默认模型
            if model.is_default:
                AIModel.objects.exclude(pk=model.pk).update(is_default=False)

            # 如果设为激活状态，取消其他激活状态
            if model.is_active:
                AIModel.objects.exclude(pk=model.pk).update(is_active=False)

            return Response(AIModelSerializer(model).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PUT', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def ai_model_detail(request, model_id):
    """AI模型详情"""
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可管理AI模型'}, status=status.HTTP_403_FORBIDDEN)

    model = get_object_or_404(AIModel, pk=model_id)

    if request.method == 'GET':
        serializer = AIModelSerializer(model)
        return Response(serializer.data)

    elif request.method in ['PUT', 'PATCH']:
        serializer = AIModelSerializer(model, data=request.data, partial=request.method == 'PATCH')
        if serializer.is_valid():
            model = serializer.save()

            # 如果设为默认模型，取消其他默认模型
            if model.is_default:
                AIModel.objects.exclude(pk=model.pk).update(is_default=False)

            # 如果设为激活状态，取消其他激活状态
            if model.is_active:
                AIModel.objects.exclude(pk=model.pk).update(is_active=False)

            return Response(AIModelSerializer(model).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        model.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def activate_model(request, model_id):
    """激活AI模型"""
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可操作AI模型'}, status=status.HTTP_403_FORBIDDEN)

    model = get_object_or_404(AIModel, pk=model_id)

    # 停用其他同类型模型
    AIModel.objects.filter(model_type=model.model_type).exclude(pk=model.pk).update(is_active=False)

    model.is_active = True
    model.save()

    # 记录部署历史
    ModelDeploymentHistory.objects.create(
        model=model,
        action='activate',
        to_version=model.version,
        deployment_strategy=model.deployment_strategy,
        operator=request.user
    )

    return Response(AIModelSerializer(model).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def deactivate_model(request, model_id):
    """停用AI模型"""
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可操作AI模型'}, status=status.HTTP_403_FORBIDDEN)

    model = get_object_or_404(AIModel, pk=model_id)
    model.is_active = False
    model.save()

    # 记录部署历史
    ModelDeploymentHistory.objects.create(
        model=model,
        action='deactivate',
        to_version=model.version,
        deployment_strategy=model.deployment_strategy,
        operator=request.user
    )

    return Response(AIModelSerializer(model).data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def set_default_model(request, model_id):
    """设置默认模型"""
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可操作AI模型'}, status=status.HTTP_403_FORBIDDEN)

    model = get_object_or_404(AIModel, pk=model_id)

    # 取消其他同类型默认模型
    AIModel.objects.filter(model_type=model.model_type).exclude(pk=model.pk).update(is_default=False)

    model.is_default = True
    model.save()

    return Response(AIModelSerializer(model).data)


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def model_performance_logs(request, model_id=None):
    """获取或创建模型性能日志"""
    if model_id:
        model = get_object_or_404(AIModel, pk=model_id)
        logs = model.performance_logs.all()
    else:
        logs = ModelPerformanceLog.objects.all()

    if request.method == 'GET':
        log_type = request.query_params.get('log_type')
        if log_type:
            logs = logs.filter(log_type=log_type)

        # 如果指定了model_id，只返回该模型的日志
        if model_id:
            logs = logs[:100]  # 限制返回数量

        serializer = ModelPerformanceLogSerializer(logs, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        if not _is_admin(request.user):
            return Response({'message': '仅管理员可记录性能日志'}, status=status.HTTP_403_FORBIDDEN)

        serializer = ModelPerformanceLogSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def model_deployment_history(request, model_id):
    """获取模型部署历史"""
    model = get_object_or_404(AIModel, pk=model_id)
    history = model.deployment_history.all()[:50]

    serializer = ModelDeploymentHistorySerializer(history, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def model_statistics(request):
    """获取模型统计信息"""
    from django.db.models import Count, Avg, Q

    # 检查是否为管理员
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可查看模型统计'}, status=status.HTTP_403_FORBIDDEN)

    # 模型数量统计
    model_stats = AIModel.objects.aggregate(
        total=Count('id'),
        grading=Count('id', filter=Q(model_type='grading')),
        segmentation=Count('id', filter=Q(model_type='segmentation')),
        active=Count('id', filter=Q(is_active=True)),
        development=Count('id', filter=Q(status='development')),
        testing=Count('id', filter=Q(status='testing')),
        production=Count('id', filter=Q(status='production')),
    )

    # 性能统计
    latest_logs = ModelPerformanceLog.objects.filter(
        log_type='inference'
    ).values('model__name', 'model__model_type').annotate(
        avg_time=Avg('avg_inference_time'),
        p95_time=Avg('p95_inference_time'),
        throughput=Avg('throughput')
    )

    # 部署历史统计
    deployment_stats = ModelDeploymentHistory.objects.aggregate(
        total_deploys=Count('id', filter=Q(action='deploy')),
        total_rollbacks=Count('id', filter=Q(action='rollback')),
    )

    return Response({
        'models': model_stats,
        'performance': list(latest_logs),
        'deployments': deployment_stats,
    })
