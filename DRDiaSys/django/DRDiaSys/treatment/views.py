import os
from django.conf import settings
from django.db.models import Q, Sum
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from diagnosis.models import CaseRecord, CaseEvent

from .models import (
    TreatmentPlan,
    TreatmentPlanExecution,
    TreatmentPlanTemplate,
    Conversation,
    Message,
    MessageTemplate
)
from .serializers import (
    TreatmentPlanSerializer,
    TreatmentPlanExecutionSerializer,
    TreatmentPlanTemplateSerializer,
    ConversationSerializer,
    ConversationDetailSerializer,
    MessageSerializer,
    MessageTemplateSerializer
)
# AI 推荐服务已移除
from diagnosis.models import CaseRecord, DiagnosisReport
from users.models import UserProfile


def _is_doctor(user):
    """检查用户是否为医生"""
    profile = getattr(user, 'profile', None)
    return getattr(profile, 'role', None) == 'doctor'


# ==================== 治疗方案推荐（已移除） ====================

def _is_admin(user):
    """检查用户是否为管理员"""
    profile = getattr(user, 'profile', None)
    return getattr(profile, 'role', None) == 'admin'
# ==================== 治疗方案管理 ====================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def treatment_plans(request):
    """获取治疗方案列表或创建新方案"""
    if request.method == 'GET':
        case_id = request.GET.get('case_id')
        status_filter = request.GET.get('status')
        patient_id = request.GET.get('patient_id')
        
        # 权限控制
        if _is_admin(request.user) or _is_doctor(request.user):
            queryset = TreatmentPlan.objects.all()
        else:
            # 患者只能查看自己的方案
            queryset = TreatmentPlan.objects.filter(case__patient=request.user)
        
        if case_id:
            queryset = queryset.filter(case_id=case_id)
        if patient_id:
            queryset = queryset.filter(case__patient_id=patient_id)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        queryset = queryset.select_related('case', 'related_report', 'created_by', 'confirmed_by', 'template_used')
        serializer = TreatmentPlanSerializer(queryset, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        # 创建新方案
        if not _is_doctor(request.user):
            return Response({'message': '仅医生可创建治疗方案'}, status=status.HTTP_403_FORBIDDEN)
        
        serializer = TreatmentPlanSerializer(data=request.data)
        if serializer.is_valid():
            plan = serializer.save(created_by=request.user)
            # 生成方案编号
            if not plan.plan_number:
                plan.plan_number = plan.generate_plan_number()
                plan.save()
            
            # 自动创建病历事件
            try:
                case = plan.case
                # 检查是否已存在相同方案的事件
                if case:
                    CaseEvent.objects.create(
                        case=case,
                        event_type='treatment',
                        description=f'创建治疗方案：{plan.title or plan.plan_number}',
                        related_plan=plan,
                        created_by=request.user
                    )
            except Exception as e:
                print(f"自动创建病历事件失败: {e}")
            
            return Response(TreatmentPlanSerializer(plan).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def treatment_plan_detail(request, plan_id):
    """获取、更新或删除治疗方案"""
    plan = get_object_or_404(
        TreatmentPlan.objects.select_related('case', 'related_report'),
        id=plan_id
    )
    
    # 权限检查
    if not (_is_admin(request.user) or _is_doctor(request.user)):
        if plan.case.patient != request.user:
            return Response({'message': '无权访问该治疗方案'}, status=status.HTTP_403_FORBIDDEN)
    
    if request.method == 'GET':
        serializer = TreatmentPlanSerializer(plan)
        return Response(serializer.data)
    
    elif request.method == 'PATCH':
        if not _is_doctor(request.user):
            return Response({'message': '仅医生可修改治疗方案'}, status=status.HTTP_403_FORBIDDEN)
        
        serializer = TreatmentPlanSerializer(plan, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        if not _is_doctor(request.user):
            return Response({'message': '仅医生可删除治疗方案'}, status=status.HTTP_403_FORBIDDEN)
        
        plan.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def confirm_treatment_plan(request, plan_id):
    """确认治疗方案"""
    if not _is_doctor(request.user):
        return Response({'message': '仅医生可确认治疗方案'}, status=status.HTTP_403_FORBIDDEN)
    
    plan = get_object_or_404(TreatmentPlan, id=plan_id)
    plan.status = 'confirmed'
    plan.confirmed_by = request.user
    plan.confirmed_at = timezone.now()
    plan.save()
    
    # 更新会话未读数（通知患者）
    try:
        conversation = Conversation.objects.get(
            patient=plan.case.patient,
            doctor=request.user
        )
        conversation.patient_unread_count += 1
        conversation.last_message = f"治疗方案 {plan.plan_number} 已确认"
        conversation.last_message_at = timezone.now()
        conversation.last_message_by = request.user
        conversation.save()
    except Conversation.DoesNotExist:
        pass
    
    serializer = TreatmentPlanSerializer(plan)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def complete_treatment_plan(request, plan_id):
    """完成治疗方案"""
    if not _is_doctor(request.user):
        return Response({'message': '仅医生可完成治疗方案'}, status=status.HTTP_403_FORBIDDEN)
    
    plan = get_object_or_404(TreatmentPlan, id=plan_id)
    
    # 只能完成执行中的方案
    if plan.status != 'active':
        return Response({'message': '只能完成执行中的方案'}, status=status.HTTP_400_BAD_REQUEST)
    
    plan.status = 'completed'
    plan.save()
    
    # 更新会话未读数（通知患者）
    try:
        conversation = Conversation.objects.get(
            patient=plan.case.patient,
            doctor=request.user
        )
        conversation.patient_unread_count += 1
        conversation.last_message = f"治疗方案 {plan.plan_number} 已完成"
        conversation.last_message_at = timezone.now()
        conversation.last_message_by = request.user
        conversation.save()
    except Conversation.DoesNotExist:
        pass
    
    serializer = TreatmentPlanSerializer(plan)
    return Response(serializer.data)


# ==================== 方案执行记录 ====================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def treatment_plan_executions(request, plan_id):
    """获取或创建方案执行记录"""
    plan = get_object_or_404(TreatmentPlan, id=plan_id)
    
    # 权限检查
    if not (_is_admin(request.user) or _is_doctor(request.user)):
        if plan.case.patient != request.user:
            return Response({'message': '无权访问该执行记录'}, status=status.HTTP_403_FORBIDDEN)
    
    if request.method == 'GET':
        executions = TreatmentPlanExecution.objects.filter(plan=plan).order_by('-execution_date')
        serializer = TreatmentPlanExecutionSerializer(executions, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        serializer = TreatmentPlanExecutionSerializer(data=request.data)
        if serializer.is_valid():
            execution = serializer.save(plan=plan, created_by=request.user)
            
            # 如果方案状态是"已确认"，自动变为"执行中"
            if plan.status == 'confirmed':
                plan.status = 'active'
                plan.save()
            
            return Response(TreatmentPlanExecutionSerializer(execution).data, status=status.HTTP_201_CREATED)
        else:
            # 打印错误以便调试
            try:
                print("TreatmentPlanExecutionSerializer errors:", serializer.errors)
            except Exception:
                pass
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ==================== 治疗方案模板管理 ====================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def treatment_templates(request):
    """获取或创建治疗方案模板"""
    if request.method == 'GET':
        is_public = request.GET.get('public', 'false').lower() == 'true'
        
        if _is_admin(request.user) or _is_doctor(request.user):
            if is_public:
                queryset = TreatmentPlanTemplate.objects.filter(is_public=True, is_active=True)
            else:
                queryset = TreatmentPlanTemplate.objects.filter(is_active=True)
        else:
            return Response({'message': '无权访问模板'}, status=status.HTTP_403_FORBIDDEN)
        
        serializer = TreatmentPlanTemplateSerializer(queryset, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        if not _is_admin(request.user):
            return Response({'message': '仅管理员可创建模板'}, status=status.HTTP_403_FORBIDDEN)
        
        serializer = TreatmentPlanTemplateSerializer(data=request.data)
        if serializer.is_valid():
            template = serializer.save()
            return Response(TreatmentPlanTemplateSerializer(template).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def treatment_template_detail(request, template_id):
    """更新或删除治疗方案模板"""
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可管理模板'}, status=status.HTTP_403_FORBIDDEN)
    
    template = get_object_or_404(TreatmentPlanTemplate, id=template_id)
    
    if request.method == 'PATCH':
        serializer = TreatmentPlanTemplateSerializer(template, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        template.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# ==================== 医患交流 - 会话管理 ====================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def conversations(request):
    """获取会话列表或创建新会话"""
    if request.method == 'GET':
        patient_id = request.GET.get('patient_id')
        doctor_id = request.GET.get('doctor_id')
        
        # 权限控制
        if _is_doctor(request.user):
            queryset = Conversation.objects.filter(doctor=request.user)
            if patient_id:
                queryset = queryset.filter(patient_id=patient_id)
        elif _is_admin(request.user):
            queryset = Conversation.objects.all()
            if patient_id:
                queryset = queryset.filter(patient_id=patient_id)
            if doctor_id:
                queryset = queryset.filter(doctor_id=doctor_id)
        else:
            # 患者只能查看自己的会话
            queryset = Conversation.objects.filter(patient=request.user)
        
        queryset = queryset.select_related('patient', 'doctor', 'related_case', 'last_message_by')
        serializer = ConversationSerializer(queryset, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        # 创建新会话
        patient_id = request.data.get('patient_id')
        doctor_id = request.data.get('doctor_id')
        case_id = request.data.get('case_id')
        
        if _is_doctor(request.user):
            # 医生创建会话，必须指定患者
            if not patient_id:
                return Response({'message': '请指定患者'}, status=status.HTTP_400_BAD_REQUEST)
            doctor = request.user
            patient = get_object_or_404(User, id=patient_id)
        elif _is_admin(request.user):
            # 管理员创建会话，需要指定医生和患者
            if not patient_id or not doctor_id:
                return Response({'message': '请指定医生和患者'}, status=status.HTTP_400_BAD_REQUEST)
            doctor = get_object_or_404(User, id=doctor_id)
            patient = get_object_or_404(User, id=patient_id)
        else:
            # 患者创建会话，需要指定医生
            if not doctor_id:
                return Response({'message': '请指定医生'}, status=status.HTTP_400_BAD_REQUEST)
            patient = request.user
            doctor = get_object_or_404(User, id=doctor_id)
            if not _is_doctor(doctor):
                return Response({'message': '指定的用户不是医生'}, status=status.HTTP_400_BAD_REQUEST)
        
        # 检查是否已存在会话
        conversation, created = Conversation.objects.get_or_create(
            patient=patient,
            doctor=doctor,
            defaults={
                'related_case_id': case_id if case_id else None
            }
        )
        
        if not created and case_id:
            conversation.related_case_id = case_id
            conversation.save()
        
        serializer = ConversationDetailSerializer(conversation)
        return Response(serializer.data, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def conversation_detail(request, conversation_id):
    """获取、更新或删除会话"""
    conversation = get_object_or_404(
        Conversation.objects.select_related('patient', 'doctor', 'related_case'),
        id=conversation_id
    )
    
    # 权限检查
    if not (_is_admin(request.user) or _is_doctor(request.user)):
        if conversation.patient != request.user:
            return Response({'message': '无权访问该会话'}, status=status.HTTP_403_FORBIDDEN)
    
    if request.method == 'GET':
        # 更新未读数
        if conversation.patient == request.user:
            conversation.patient_unread_count = 0
        elif conversation.doctor == request.user:
            conversation.doctor_unread_count = 0
        conversation.save()
        
        serializer = ConversationDetailSerializer(conversation)
        return Response(serializer.data)
    
    elif request.method == 'PATCH':
        # 更新会话（如标记已读、关联病例等）
        if 'related_case_id' in request.data:
            case_id = request.data.get('related_case_id')
            if case_id:
                case = get_object_or_404(CaseRecord, id=case_id)
                conversation.related_case = case
        if 'is_active' in request.data:
            conversation.is_active = request.data.get('is_active')
        conversation.save()
        
        serializer = ConversationDetailSerializer(conversation)
        return Response(serializer.data)
    
    elif request.method == 'DELETE':
        conversation.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def unread_count(request):
    """获取未读消息数"""
    if _is_doctor(request.user):
        total_unread = Conversation.objects.filter(doctor=request.user).aggregate(
            total=Sum('doctor_unread_count')
        )['total'] or 0
    else:
        total_unread = Conversation.objects.filter(patient=request.user).aggregate(
            total=Sum('patient_unread_count')
        )['total'] or 0
    
    return Response({'unread_count': total_unread})


# ==================== 医患交流 - 消息管理 ====================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def messages(request, conversation_id):
    """获取或发送消息"""
    conversation = get_object_or_404(Conversation, id=conversation_id)
    
    # 权限检查
    if not (_is_admin(request.user) or _is_doctor(request.user)):
        if conversation.patient != request.user:
            return Response({'message': '无权访问该会话'}, status=status.HTTP_403_FORBIDDEN)
    
    if request.method == 'GET':
        # 获取消息列表
        messages_list = Message.objects.filter(conversation=conversation).order_by('created_at')
        
        # 标记消息为已读
        if conversation.patient == request.user:
            Message.objects.filter(
                conversation=conversation,
                sender=conversation.doctor,
                is_read=False
            ).update(is_read=True, read_at=timezone.now())
            conversation.patient_unread_count = 0
        elif conversation.doctor == request.user:
            Message.objects.filter(
                conversation=conversation,
                sender=conversation.patient,
                is_read=False
            ).update(is_read=True, read_at=timezone.now())
            conversation.doctor_unread_count = 0
        conversation.save()
        
        serializer = MessageSerializer(messages_list, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        # 发送消息
        serializer = MessageSerializer(data=request.data)
        if serializer.is_valid():
            message = serializer.save(
                conversation=conversation,
                sender=request.user
            )
            
            # 更新会话的最后消息
            conversation.last_message = message.content[:100]
            conversation.last_message_at = message.created_at
            conversation.last_message_by = request.user
            
            # 更新未读数
            if request.user == conversation.patient:
                conversation.doctor_unread_count += 1
            else:
                conversation.patient_unread_count += 1
            
            conversation.save()
            
            return Response(MessageSerializer(message).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def message_detail(request, message_id):
    """获取、更新或删除消息"""
    message = get_object_or_404(Message, id=message_id)
    
    # 权限检查
    if not (_is_admin(request.user) or _is_doctor(request.user)):
        if message.conversation.patient != request.user and message.sender != request.user:
            return Response({'message': '无权访问该消息'}, status=status.HTTP_403_FORBIDDEN)
    
    if request.method == 'GET':
        serializer = MessageSerializer(message)
        return Response(serializer.data)
    
    elif request.method == 'PATCH':
        # 更新消息（如标记重要、已读等）
        if 'is_read' in request.data:
            message.is_read = request.data.get('is_read')
            if message.is_read and not message.read_at:
                message.read_at = timezone.now()
        if 'is_important' in request.data:
            if not _is_doctor(request.user):
                return Response({'message': '仅医生可标记重要消息'}, status=status.HTTP_403_FORBIDDEN)
            message.is_important = request.data.get('is_important')
        message.save()
        
        serializer = MessageSerializer(message)
        return Response(serializer.data)
    
    elif request.method == 'DELETE':
        # 只能删除自己发送的消息
        if message.sender != request.user and not _is_admin(request.user):
            return Response({'message': '只能删除自己发送的消息'}, status=status.HTTP_403_FORBIDDEN)
        message.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def download_message_file(request, message_id):
    """下载消息文件（通过API代理，避免混合内容问题）"""
    message = get_object_or_404(Message, id=message_id)
    
    # 权限检查
    if not (_is_admin(request.user) or _is_doctor(request.user)):
        if message.conversation.patient != request.user and message.sender != request.user:
            return Response({'message': '无权访问该消息'}, status=status.HTTP_403_FORBIDDEN)
    
    # 检查是否有文件
    if not message.file_url:
        return Response({'message': '该消息没有附件'}, status=status.HTTP_404_NOT_FOUND)
    
    # 构建文件路径
    file_path = message.file_url
    if file_path.startswith('/'):
        file_path = file_path[1:]
    
    full_path = os.path.join(settings.MEDIA_ROOT, file_path)
    
    # 检查文件是否存在
    if not os.path.exists(full_path):
        return Response({'message': '文件不存在'}, status=status.HTTP_404_NOT_FOUND)
    
    # 获取文件名
    file_name = message.file_name or os.path.basename(full_path)
    
    # 根据文件类型设置Content-Type
    content_type = 'application/octet-stream'
    if file_name.lower().endswith(('.pdf',)):
        content_type = 'application/pdf'
    elif file_name.lower().endswith(('.doc', '.docx')):
        content_type = 'application/msword'
    elif file_name.lower().endswith(('.xls', '.xlsx')):
        content_type = 'application/vnd.ms-excel'
    elif file_name.lower().endswith(('.jpg', '.jpeg')):
        content_type = 'image/jpeg'
    elif file_name.lower().endswith('.png'):
        content_type = 'image/png'
    elif file_name.lower().endswith('.gif'):
        content_type = 'image/gif'
    elif file_name.lower().endswith('.txt'):
        content_type = 'text/plain'
    
    try:
        with open(full_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type=content_type)
            response['Content-Disposition'] = f'attachment; filename="{file_name}"'
            return response
    except Exception as e:
        return Response({'message': f'读取文件失败: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_message_file(request, conversation_id):
    """上传消息文件（图片/文件）"""
    conversation = get_object_or_404(Conversation, id=conversation_id)
    
    # 权限检查
    if not (_is_admin(request.user) or _is_doctor(request.user)):
        if conversation.patient != request.user:
            return Response({'message': '无权访问该会话'}, status=status.HTTP_403_FORBIDDEN)
    
    if 'file' not in request.FILES:
        return Response({'message': '请上传文件'}, status=status.HTTP_400_BAD_REQUEST)
    
    file = request.FILES['file']
    file_type = request.data.get('file_type', 'file')  # 'image' or 'file'
    
    # 验证文件类型和大小
    max_size = 10 * 1024 * 1024  # 10 MB
    if file.size > max_size:
        return Response({'message': '文件过大，最大支持 10MB'}, status=status.HTTP_400_BAD_REQUEST)

    _, ext = os.path.splitext(file.name)
    ext = ext.lower()
    allowed_image_mimes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp']
    allowed_file_exts = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.zip', '.rar', '.csv']

    if file_type == 'image':
        # 检查 MIME 类型 and extension
        if not (file.content_type in allowed_image_mimes or file.content_type.startswith('image/')):
            return Response({'message': '不支持的图片格式'}, status=status.HTTP_400_BAD_REQUEST)
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            return Response({'message': '不支持的图片后缀'}, status=status.HTTP_400_BAD_REQUEST)
    else:
        # 普通文件
        if ext not in allowed_file_exts:
            return Response({'message': '不支持的文件类型'}, status=status.HTTP_400_BAD_REQUEST)
    
    # 保存文件
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'conversations', str(conversation_id))
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    # 创建消息
    file_url = f"/media/conversations/{conversation_id}/{file.name}"
    message_type = 'image' if file_type == 'image' else 'file'
    
    message = Message.objects.create(
        conversation=conversation,
        sender=request.user,
        message_type=message_type,
        content=f"发送了{'图片' if message_type == 'image' else '文件'}: {file.name}",
        file_url=file_url,
        file_name=file.name
    )
    
    # 更新会话
    conversation.last_message = message.content
    conversation.last_message_at = message.created_at
    conversation.last_message_by = request.user
    
    if request.user == conversation.patient:
        conversation.doctor_unread_count += 1
    else:
        conversation.patient_unread_count += 1
    
    conversation.save()
    
    return Response(MessageSerializer(message).data, status=status.HTTP_201_CREATED)


# ==================== 消息模板管理 ====================

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def message_templates(request):
    """获取或创建消息模板"""
    if request.method == 'GET':
        if _is_doctor(request.user):
            # 医生可以查看自己的模板和公开模板
            queryset = MessageTemplate.objects.filter(
                Q(doctor=request.user) | Q(is_public=True)
            )
        elif _is_admin(request.user):
            # 管理员可以查看所有模板
            queryset = MessageTemplate.objects.all()
        else:
            return Response({'message': '无权访问模板'}, status=status.HTTP_403_FORBIDDEN)
        
        serializer = MessageTemplateSerializer(queryset, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        if not _is_doctor(request.user):
            return Response({'message': '仅医生可创建消息模板'}, status=status.HTTP_403_FORBIDDEN)
        
        serializer = MessageTemplateSerializer(data=request.data)
        if serializer.is_valid():
            template = serializer.save(doctor=request.user)
            return Response(MessageTemplateSerializer(template).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def message_template_detail(request, template_id):
    """更新或删除消息模板"""
    template = get_object_or_404(MessageTemplate, id=template_id)
    
    # 权限检查：只能修改自己的模板，除非是管理员
    if template.doctor != request.user and not _is_admin(request.user):
        return Response({'message': '无权修改该模板'}, status=status.HTTP_403_FORBIDDEN)
    
    if request.method == 'PATCH':
        serializer = MessageTemplateSerializer(template, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        template.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
