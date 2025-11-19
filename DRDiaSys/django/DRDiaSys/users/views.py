from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User, Group
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.decorators import api_view, permission_classes
from django.contrib.auth.hashers import make_password
from django.db.models import Count
from rest_framework.pagination import PageNumberPagination
from django.db import models

def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }

def ensure_groups_exist():
    """確保系統基本用戶組存在"""
    required_groups = ['admin', 'doctor', 'patient']
    for group_name in required_groups:
        Group.objects.get_or_create(name=group_name)

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    username = request.data.get('username')
    password = request.data.get('password')
    email = request.data.get('email')

    if User.objects.filter(username=username).exists():
        return Response({'message': '用戶名已存在'}, status=status.HTTP_400_BAD_REQUEST)
    
    if User.objects.filter(email=email).exists():
        return Response({'message': '郵箱已存在'}, status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.create(
        username=username,
        email=email,
        password=make_password(password)
        
    )

    # 確保必要的角色組存在
    ensure_groups_exist()

    # 默認將新用戶加入 patient 組
    patient_group = Group.objects.get(name='patient')
    user.groups.add(patient_group)
    
    # 確保UserProfile的role字段與用戶組一致
    if hasattr(user, 'profile'):
        user.profile.role = 'patient'
        user.profile.save()
        print(f"註冊用戶 {username} 的UserProfile角色已設置為: patient")

    return Response({'message': '註冊成功'}, status=status.HTTP_201_CREATED)

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    username = request.data.get('username')
    password = request.data.get('password')

    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({'message': '用戶不存在'}, status=status.HTTP_400_BAD_REQUEST)

    if not user.check_password(password):
        return Response({'message': '密碼錯誤'}, status=status.HTTP_400_BAD_REQUEST)

    if not user.is_active:
        return Response({'message': '用戶已被禁用'}, status=status.HTTP_400_BAD_REQUEST)

    tokens = get_tokens_for_user(user)
    
    # 獲取用戶的第一個組作為角色
    user_role = user.groups.first().name if user.groups.exists() else 'patient'

    return Response({
        'token': tokens['access'],
        'refresh': tokens['refresh'],
        'username': user.username,
        'userRole': user_role
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_list(request):
    """获取用户列表，支持分页、搜索和角色过滤"""
    page_size = request.query_params.get('page_size', 10)
    page = request.query_params.get('page', 1)
    search_query = request.query_params.get('search', '').strip()
    role_filter = request.query_params.get('role', '').strip()
    
    try:
        page = int(page)
        page_size = int(page_size)
    except ValueError:
        page = 1
        page_size = 10
    
    # 创建基础查询
    users = User.objects.all().order_by('-date_joined')  # 默认按注册时间降序排列
    
    # 如果有角色过滤条件，筛选用户
    if role_filter:
        print(f"按角色筛选用户: '{role_filter}'")
        # 确保角色组存在
        ensure_groups_exist()
        try:
            role_group = Group.objects.get(name=role_filter)
            users = users.filter(groups=role_group)
            print(f"角色 '{role_filter}' 筛选结果: {users.count()} 个用户")
        except Group.DoesNotExist:
            print(f"警告: 角色 '{role_filter}' 不存在")
    
    # 如果有搜索条件，筛选用户
    if search_query:
        print(f"搜索用户，关键词: '{search_query}'")
        users = users.filter(
            models.Q(username__icontains=search_query) | 
            models.Q(first_name__icontains=search_query) |
            models.Q(email__icontains=search_query)
        )
        print(f"搜索结果数量: {users.count()} 个用户")
    
    # 获取总数
    total_users = users.count()
    
    # 手动分页
    start = (page - 1) * page_size
    end = start + page_size
    paginated_users = users[start:end]
    
    filter_info = []
    if role_filter:
        filter_info.append(f"角色={role_filter}")
    if search_query:
        filter_info.append(f"搜索={search_query}")
    
    filter_str = "，".join(filter_info)
    print(f"分页: 第 {page} 页，每页 {page_size} 条，返回 {len(paginated_users)} 条记录{filter_str and '，过滤条件：' + filter_str}")
    
    user_data = []
    for user in paginated_users:
        groups = [group.name for group in user.groups.all()]
        user_data.append({
            'id': user.id,
            'username': user.username,
            'first_name': user.first_name,
            'email': user.email,
            'date_joined': user.date_joined,
            'is_active': user.is_active,
            'groups': groups,
            'is_staff': user.is_staff
        })
    
    return Response({
        'results': user_data,
        'total': total_users,
        'page': page,
        'page_size': page_size,
        'total_pages': (total_users + page_size - 1) // page_size,  # 向上取整
        'search_query': search_query,
        'role_filter': role_filter,
        'has_search': bool(search_query),
        'has_filter': bool(role_filter)
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def create_user(request):
    # 確保必要的角色組存在
    ensure_groups_exist()
    
    username = request.data.get('username')
    password = request.data.get('password')
    first_name = request.data.get('first_name')
    groups = request.data.get('groups', [])

    if User.objects.filter(username=username).exists():
        return Response({'message': '用戶名已存在'}, status=status.HTTP_400_BAD_REQUEST)

    # 判斷是否為管理員
    is_admin = 'admin' in groups
    
    user = User.objects.create(
        username=username,
        first_name=first_name,
        password=make_password(password),
        is_staff=is_admin  # 如果是admin組，設置is_staff為True
    )

    # 添加用戶到指定組
    role = 'patient'  # 默認角色
    for group_name in groups:
        group = Group.objects.get(name=group_name)
        user.groups.add(group)
        # 使用第一個組名作為用戶的角色
        if role == 'patient':  # 只有在還沒設置其他角色時才設置
            role = group_name
    
    # 同步更新UserProfile表中的role字段
    if hasattr(user, 'profile'):
        user.profile.role = role
        user.profile.save()
        print(f"用戶 {username} 的UserProfile角色已設置為: {role}")

    return Response({'message': '用戶創建成功'}, status=status.HTTP_201_CREATED)

@api_view(['PUT'])
@permission_classes([IsAuthenticated, IsAdminUser])
def update_user(request, user_id):
    """更新用戶信息，包括基本信息、狀態和角色"""
    # 確保必要的角色組存在
    ensure_groups_exist()
    
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return Response({'message': '用戶不存在'}, status=status.HTTP_404_NOT_FOUND)

    print(f"正在更新用戶 {user.username} (ID: {user_id}) 的信息")
    print(f"請求數據: {request.data}")
    
    # 更新基本信息
    if 'first_name' in request.data:
        user.first_name = request.data['first_name']
        print(f"更新用戶名: {user.first_name}")
    
    # 處理用戶狀態（激活/禁用）
    if 'is_active' in request.data:
        old_status = user.is_active
        new_status = request.data['is_active']
        # 將字符串'true'/'false'轉換為布爾值
        if isinstance(new_status, str):
            new_status = new_status.lower() == 'true'
        
        user.is_active = new_status
        print(f"更新用戶狀態: 從 {old_status} 變為 {new_status}")
        
        # 同步更新UserProfile的狀態
        if hasattr(user, 'profile'):
            user.profile.is_active = new_status
            user.profile.save()
            print(f"同步更新 UserProfile 狀態為: {new_status}")

    # 更新用戶組
    if 'groups' in request.data:
        old_groups = [g.name for g in user.groups.all()]
        user.groups.clear()
        role = None
        is_admin = False
        
        print(f"原有用戶組: {old_groups}")
        print(f"新用戶組: {request.data['groups']}")
        
        for group_name in request.data['groups']:
            group = Group.objects.get(name=group_name)
            user.groups.add(group)
            # 檢查是否為管理員組
            if group_name == 'admin':
                is_admin = True
            # 使用第一個組名作為用戶的角色
            if role is None:
                role = group_name
        
        # 更新is_staff狀態
        user.is_staff = is_admin
        print(f"更新管理員狀態 is_staff: {is_admin}")
        
        # 同步更新UserProfile表中的role字段
        if role and hasattr(user, 'profile'):
            user.profile.role = role
            user.profile.save()
            print(f"同步更新 UserProfile 角色為: {role}")

    user.save()
    print(f"用戶 {user.username} 更新成功")
    
    # 返回完整的用戶信息
    response_data = {
        'message': '用戶更新成功',
        'user': {
            'id': user.id,
            'username': user.username,
            'first_name': user.first_name,
            'email': user.email,
            'is_active': user.is_active,
            'groups': [group.name for group in user.groups.all()],
            'is_staff': user.is_staff
        }
    }
    
    return Response(response_data)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def reset_password(request, user_id):
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return Response({'message': '用戶不存在'}, status=status.HTTP_404_NOT_FOUND)

    # 生成隨機密碼
    import random
    import string
    new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    user.set_password(new_password)
    user.save()

    return Response({
        'message': '密碼重置成功',
        'new_password': new_password
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def role_statistics(request):
    """獲取用戶角色統計數據"""
    # 確保所有基本角色組存在
    ensure_groups_exist()
    
    # 打印當前所有用戶組及其關聯的用戶數量，用於調試
    all_groups = Group.objects.all()
    print(f"系統中所有用戶組: {[g.name for g in all_groups]}")
    
    # 獲取所有用戶組並計算每個用戶組的用戶數量
    role_counts = Group.objects.annotate(user_count=Count('user')).values('name', 'user_count')
    
    # 輸出原始角色計數用於調試
    print("角色統計原始數據:")
    for role in role_counts:
        print(f"角色: {role['name']}, 數量: {role['user_count']}")
    
    # 確保所有必需的角色都在返回數據中，即使是0個用戶
    role_distribution = []
    required_roles = {'admin', 'doctor', 'patient'}
    existing_roles = {role['name']: role['user_count'] for role in role_counts}
    
    for role_name in required_roles:
        role_distribution.append({
            'name': role_name,
            'user_count': existing_roles.get(role_name, 0)
        })
    
    # 添加其他非必需角色，如果有的話
    for role_name, count in existing_roles.items():
        if role_name not in required_roles:
            role_distribution.append({
                'name': role_name,
                'user_count': count
            })
    
    # 計算總用戶數和活躍/非活躍用戶數
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    inactive_users = User.objects.filter(is_active=False).count()
    
    # 組織返回的數據
    statistics = {
        'total_users': total_users,
        'role_distribution': role_distribution,
        'active_users': active_users,
        'inactive_users': inactive_users
    }
    
    # 輸出最終返回的統計數據
    print("最終返回的角色統計數據:", statistics)
    
    return Response(statistics)

@api_view(['POST'])
@permission_classes([AllowAny])
def fix_admin_permissions(request):
    """臨時接口：修復所有管理員用戶的權限設置"""
    admin_group = Group.objects.get(name='admin')
    admin_users = User.objects.filter(groups=admin_group)
    
    fixed_count = 0
    for user in admin_users:
        if not user.is_staff:
            user.is_staff = True
            user.save()
            fixed_count += 1
            print(f"修復用戶 {user.username} 的管理員權限")
    
    return Response({
        'message': f'成功修復 {fixed_count} 個管理員用戶的權限',
        'fixed_users': [user.username for user in admin_users if user.is_staff]
    })