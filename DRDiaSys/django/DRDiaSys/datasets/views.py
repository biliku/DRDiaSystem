import os
from django.conf import settings
from django.http import JsonResponse, FileResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
import json
from datetime import datetime
import shutil
import urllib.parse
import mimetypes
from rest_framework.decorators import api_view
from rest_framework.decorators import permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404

from .models import PatientImage
from .serializers import PatientImageSerializer

# 数据集存储路径
DATASET_ROOT = 'F:/DRDiaSys/DRDiaSys/django/DRDiaSys/datasets/dataset'
PATIENT_UPLOAD_FOLDER = 'patient_uploads'

def normalize_path(path):
    """标准化路径"""
    # 解码URL编码的路径
    decoded_path = urllib.parse.unquote(path)
    # 移除开头的斜杠
    if decoded_path.startswith('/'):
        decoded_path = decoded_path[1:]
    # 替换正斜杠为反斜杠（Windows系统）
    normalized = decoded_path.replace('/', '\\')
    return normalized

def get_dataset_info(dataset_path):
    """获取数据集信息"""
    try:
        # 获取图片文件列表
        image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        # 获取创建时间
        created_time = datetime.fromtimestamp(os.path.getctime(dataset_path))
        
        return {
            'id': os.path.basename(dataset_path),
            'name': os.path.basename(dataset_path),
            'image_count': len(image_files),
            'created_at': created_time.isoformat(),
            'status': 'unprocessed',  # 默认状态
            'path': dataset_path
        }
    except Exception as e:
        print(f"Error getting dataset info: {str(e)}")
        return None

@require_http_methods(["GET"])
def list_local_datasets(request):
    """列出本地数据集"""
    try:
        base_path = request.GET.get('base_path', DATASET_ROOT)
        base_path = normalize_path(base_path)
        
        print(f"Listing datasets in: {base_path}")  # 调试信息
        
        if not os.path.exists(base_path):
            print(f"Directory does not exist: {base_path}")  # 调试信息
            return JsonResponse({'error': f'Directory not found: {base_path}'}, status=404)
        
        datasets = []
        
        # 遍历数据集目录
        for dataset_name in os.listdir(base_path):
            dataset_path = os.path.join(base_path, dataset_name)
            if os.path.isdir(dataset_path):
                dataset_info = get_dataset_info(dataset_path)
                if dataset_info:
                    datasets.append(dataset_info)
        
        return JsonResponse({
            'datasets': datasets,
            'total': len(datasets)
        })
    except Exception as e:
        print(f"Error in list_local_datasets: {str(e)}")  # 调试信息
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_statistics(request):
    """获取数据集统计信息"""
    try:
        base_path = request.GET.get('base_path', DATASET_ROOT)
        base_path = normalize_path(base_path)
        
        print(f"Getting statistics for: {base_path}")  # 调试信息
        
        if not os.path.exists(base_path):
            print(f"Directory does not exist: {base_path}")  # 调试信息
            return JsonResponse({'error': f'Directory not found: {base_path}'}, status=404)
        
        stats = {
            'unprocessed': 0,
            'processing': 0,
            'processed': 0
        }
        
        # 遍历数据集目录
        for dataset_name in os.listdir(base_path):
            dataset_path = os.path.join(base_path, dataset_name)
            if os.path.isdir(dataset_path):
                # 这里可以根据实际需求判断数据集状态
                stats['unprocessed'] += 1
        
        return JsonResponse(stats)
    except Exception as e:
        print(f"Error in get_statistics: {str(e)}")  # 调试信息
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["POST"])
def upload_dataset(request):
    """上传数据集"""
    try:
        name = request.POST.get('name')
        base_path = request.POST.get('base_path', DATASET_ROOT)
        base_path = normalize_path(base_path)
        files = request.FILES.getlist('files')
        
        if not name or not files:
            return JsonResponse({'error': 'Missing required fields'}, status=400)
        
        # 创建数据集目录
        dataset_path = os.path.join(base_path, name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # 保存文件
        for file in files:
            file_path = os.path.join(dataset_path, file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
        
        return JsonResponse({'message': 'Dataset uploaded successfully'}, status=201)
    except Exception as e:
        print(f"Error in upload_dataset: {str(e)}")  # 调试信息
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def preview_dataset(request, dataset_id):
    """预览数据集"""
    try:
        base_path = request.GET.get('base_path', DATASET_ROOT)
        base_path = normalize_path(base_path)
        dataset_path = os.path.join(base_path, dataset_id)
        
        print(f"Previewing dataset: {dataset_path}")  # 调试信息
        
        if not os.path.exists(dataset_path):
            print(f"Dataset directory not found: {dataset_path}")  # 调试信息
            return JsonResponse({'error': 'Dataset not found'}, status=404)
        
        # 获取当前目录路径
        current_dir = request.GET.get('dir', '')
        # 规范化目录路径
        current_dir = current_dir.replace('/', os.sep).strip(os.sep)
        current_path = os.path.join(dataset_path, current_dir)
        
        print(f"Current path: {current_path}")  # 调试信息
        
        if not os.path.exists(current_path):
            print(f"Current directory not found: {current_path}")  # 调试信息
            return JsonResponse({'error': 'Directory not found'}, status=404)
        
        # 获取目录结构
        dirs = []
        files = []
        
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            if os.path.isdir(item_path):
                # 构建相对路径
                rel_path = os.path.join(current_dir, item) if current_dir else item
                dirs.append({
                    'name': item,
                    'type': 'directory',
                    'path': rel_path.replace(os.sep, '/')  # 转换为URL格式
                })
            elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # 构建相对路径
                rel_path = os.path.join(current_dir, item) if current_dir else item
                files.append({
                    'name': item,
                    'type': 'file',
                    'path': rel_path.replace(os.sep, '/')  # 转换为URL格式
                })
        
        # 获取分页参数
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        
        # 计算分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_items = files[start_idx:end_idx]
        
        # 构建面包屑导航
        breadcrumbs = []
        if current_dir:
            parts = current_dir.split(os.sep)
            current_path = ''
            for part in parts:
                current_path = os.path.join(current_path, part)
                breadcrumbs.append({
                    'name': part,
                    'path': current_path.replace(os.sep, '/')  # 转换为URL格式
                })
        
        return JsonResponse({
            'dataset_id': dataset_id,
            'current_dir': current_dir.replace(os.sep, '/'),  # 转换为URL格式
            'breadcrumbs': breadcrumbs,
            'directories': dirs,
            'files': paginated_items,
            'total_files': len(files),
            'current_page': page,
            'page_size': page_size,
            'total_pages': (len(files) + page_size - 1) // page_size
        })
    except Exception as e:
        print(f"Error in preview_dataset: {str(e)}")  # 调试信息
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["DELETE"])
def delete_dataset(request, dataset_id):
    """删除数据集"""
    try:
        base_path = request.GET.get('base_path', DATASET_ROOT)
        base_path = normalize_path(base_path)
        dataset_path = os.path.join(base_path, dataset_id)
        
        if not os.path.exists(dataset_path):
            return JsonResponse({'error': 'Dataset not found'}, status=404)
        
        # 删除数据集目录
        shutil.rmtree(dataset_path)
        
        return JsonResponse({}, status=204)
    except Exception as e:
        print(f"Error in delete_dataset: {str(e)}")  # 调试信息
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])
def get_image(request, dataset_id, image_path):
    try:
        # 直接使用文件系统路径
        dataset_path = os.path.join(DATASET_ROOT, dataset_id)
        
        # 规范化图片路径
        image_path = image_path.strip('/')
        full_image_path = os.path.join(dataset_path, image_path)
        
        print(f"Getting image: {full_image_path}")  # 调试信息
        
        # 检查文件是否存在
        if not os.path.exists(full_image_path):
            print(f"Image file not found: {full_image_path}")  # 调试信息
            return Response({'error': 'Image not found'}, status=404)
            
        # 检查文件大小
        file_size = os.path.getsize(full_image_path)
        if file_size > 10 * 1024 * 1024:  # 如果文件大于10MB
            return Response({'error': 'File too large'}, status=413)
            
        # 获取文件扩展名
        _, ext = os.path.splitext(full_image_path)
        ext = ext.lower()
        
        # 设置正确的 Content-Type
        content_type = 'image/jpeg'  # 默认
        if ext == '.png':
            content_type = 'image/png'
        elif ext == '.gif':
            content_type = 'image/gif'
            
        # 使用 StreamingHttpResponse 进行流式传输
        def file_iterator(file_path, chunk_size=8192):
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
                    
        response = StreamingHttpResponse(
            file_iterator(full_image_path),
            content_type=content_type
        )
        
        # 添加响应头
        response['Content-Length'] = file_size
        response['Accept-Ranges'] = 'bytes'
        
        return response
        
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        return Response({'error': str(e)}, status=500) 


def _is_admin(user):
    if user.is_superuser or user.is_staff:
        return True
    profile = getattr(user, 'profile', None)
    return getattr(profile, 'role', None) == 'admin'


def _build_patient_image_path(owner):
    """
    构建患者图像存储路径
    结构: patient_uploads/patient_{patient_id}_{username}/
    每个患者拥有独立的文件夹，便于管理和查找
    """
    # 获取患者用户名，清理特殊字符以确保文件夹名称安全
    username = owner.username
    # 移除或替换可能导致文件系统问题的字符
    safe_username = "".join(c for c in username if c.isalnum() or c in ('_', '-')).strip()
    if not safe_username:
        safe_username = f"user_{owner.id}"
    
    # 创建患者专属文件夹: patient_{id}_{username}
    folder_name = f"patient_{owner.id}_{safe_username}"
    base_dir = os.path.join(DATASET_ROOT, PATIENT_UPLOAD_FOLDER, folder_name)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def patient_images(request):
    """患者影像列表 & 上传"""
    if request.method == 'GET':
        images = PatientImage.objects.filter(owner=request.user).order_by('-created_at')
        serializer = PatientImageSerializer(images, many=True)
        return Response(serializer.data)

    image_file = request.FILES.get('image')
    if not image_file:
        return Response({'message': '请上传影像文件'}, status=400)

    eye_side = request.data.get('eye_side', 'unknown')
    description = request.data.get('description', '')

    # 使用新的文件结构：每个患者拥有独立文件夹
    target_dir = _build_patient_image_path(request.user)
    # 文件名包含：日期时间_眼别_原始文件名，便于在文件夹内识别
    eye_prefix = {'left': 'L', 'right': 'R', 'both': 'B', 'unknown': 'U'}.get(eye_side, 'U')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = f"{timestamp}_{eye_prefix}_{image_file.name}"
    file_path = os.path.join(target_dir, safe_name)

    with open(file_path, 'wb+') as destination:
        for chunk in image_file.chunks():
            destination.write(chunk)

    relative_path = os.path.relpath(file_path, DATASET_ROOT)

    patient_image = PatientImage.objects.create(
        owner=request.user,
        uploaded_by=request.user,
        original_name=image_file.name,
        stored_path=relative_path.replace('\\', '/'),
        eye_side=eye_side,
        description=description,
        file_size=image_file.size
    )

    # 自动创建诊断任务
    try:
        from diagnosis.views import create_diagnosis_task
        from diagnosis.models import DiagnosisTask
        import threading
        
        # 创建诊断任务
        diagnosis_task = DiagnosisTask.objects.create(
            patient_image=patient_image,
            task_type='lesion_segmentation',
            status='pending'
        )
        
        # 异步处理诊断任务
        from diagnosis.views import process_diagnosis_task
        thread = threading.Thread(target=process_diagnosis_task, args=(diagnosis_task.id,))
        thread.daemon = True
        thread.start()
    except Exception as e:
        # 如果诊断任务创建失败，不影响上传成功
        print(f"自动创建诊断任务失败: {e}")

    serializer = PatientImageSerializer(patient_image)
    return Response(serializer.data, status=201)


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def patient_image_detail(request, image_id):
    """患者查看/删除单个影像"""
    image = get_object_or_404(PatientImage, pk=image_id)
    if image.owner != request.user:
        return Response({'message': '无权访问该影像'}, status=403)

    if request.method == 'GET':
        serializer = PatientImageSerializer(image)
        return Response(serializer.data)

    full_path = os.path.join(DATASET_ROOT, image.stored_path)
    if os.path.exists(full_path):
        os.remove(full_path)
    image.delete()
    return Response(status=204)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def download_patient_image(request, image_id):
    """下载患者影像（患者本人或管理员可用）"""
    image = get_object_or_404(PatientImage, pk=image_id)
    if image.owner != request.user and not _is_admin(request.user):
        return Response({'message': '无权访问该影像'}, status=403)

    full_path = os.path.join(DATASET_ROOT, image.stored_path)
    if not os.path.exists(full_path):
        return Response({'message': '文件不存在'}, status=404)

    ext = os.path.splitext(full_path)[1].lower()
    content_type = 'image/jpeg'
    if ext == '.png':
        content_type = 'image/png'
    elif ext == '.gif':
        content_type = 'image/gif'

    response = FileResponse(open(full_path, 'rb'), content_type=content_type)
    mode = request.GET.get('mode', 'download')
    disposition = 'inline' if mode == 'inline' else 'attachment'
    response['Content-Disposition'] = f'{disposition}; filename="{image.original_name}"'
    return response


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def admin_patient_images(request):
    """管理员集中查看患者上传影像 - 按患者分组返回"""
    if not _is_admin(request.user):
        return Response({'message': '仅管理员可访问'}, status=403)

    patient_id = request.GET.get('patient_id')
    queryset = PatientImage.objects.select_related('owner').all().order_by('-created_at')
    if patient_id:
        queryset = queryset.filter(owner_id=patient_id)

    # 按患者分组组织数据
    grouped_data = {}
    for image in queryset:
        owner_id = image.owner.id
        owner_username = image.owner.username
        
        if owner_id not in grouped_data:
            grouped_data[owner_id] = {
                'patient_id': owner_id,
                'patient_name': owner_username,
                'patient_folder': f"patient_{owner_id}_{owner_username}",
                'total_images': 0,
                'total_size': 0,
                'latest_upload': None,
                'images': []
            }
        
        # 序列化影像数据
        serializer = PatientImageSerializer(image)
        image_data = serializer.data
        
        # 更新统计信息
        grouped_data[owner_id]['total_images'] += 1
        grouped_data[owner_id]['total_size'] += image.file_size or 0
        if not grouped_data[owner_id]['latest_upload'] or image.created_at > grouped_data[owner_id]['latest_upload']:
            grouped_data[owner_id]['latest_upload'] = image.created_at
        
        grouped_data[owner_id]['images'].append(image_data)
    
    # 转换为列表格式，按最新上传时间排序
    result = list(grouped_data.values())
    # 按最新上传时间排序，没有上传时间的排到最后
    def get_sort_key(x):
        if x['latest_upload']:
            return x['latest_upload']
        # 返回一个很早的时间作为默认值
        return datetime(1970, 1, 1)
    result.sort(key=get_sort_key, reverse=True)
    
    return Response(result)