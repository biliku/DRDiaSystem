from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('users/', views.user_list, name='user_list'),
    path('users/create/', views.create_user, name='create_user'),
    path('users/<int:user_id>/', views.update_user, name='update_user'),
    path('users/<int:user_id>/reset_password/', views.reset_password, name='reset_password'),
    path('users/role_statistics/', views.role_statistics, name='role_statistics'),
    path('fix_admin_permissions/', views.fix_admin_permissions, name='fix_admin_permissions'),
    # 患者信息录入相关API
    path('patient/info/', views.patient_info, name='patient_info'),
    path('patient/conditions/', views.condition_info_list, name='condition_info_list'),
    path('patient/conditions/<int:condition_id>/', views.condition_info_detail, name='condition_info_detail'),
]
