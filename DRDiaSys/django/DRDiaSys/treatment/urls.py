from django.urls import path
from . import views

urlpatterns = [
    # 治疗方案管理
    path('plans/', views.treatment_plans, name='treatment_plans'),
    path('plans/<int:plan_id>/', views.treatment_plan_detail, name='treatment_plan_detail'),
    path('plans/<int:plan_id>/confirm/', views.confirm_treatment_plan, name='confirm_treatment_plan'),
    path('plans/<int:plan_id>/complete/', views.complete_treatment_plan, name='complete_treatment_plan'),
    path('plans/<int:plan_id>/executions/', views.treatment_plan_executions, name='treatment_plan_executions'),

    # 治疗方案模板管理
    path('templates/', views.treatment_templates, name='treatment_templates'),
    path('templates/<int:template_id>/', views.treatment_template_detail, name='treatment_template_detail'),

    # 医患交流 - 会话管理
    path('conversations/', views.conversations, name='conversations'),
    path('conversations/<int:conversation_id>/', views.conversation_detail, name='conversation_detail'),
    path('conversations/unread-count/', views.unread_count, name='unread_count'),

    # 医患交流 - 消息管理
    path('conversations/<int:conversation_id>/messages/', views.messages, name='messages'),
    path('conversations/<int:conversation_id>/upload/', views.upload_message_file, name='upload_message_file'),

    # 消息文件下载（通过API代理，避免混合内容问题）- 必须放在 message_id/ 之前
    path('messages/<int:message_id>/download/', views.download_message_file, name='download_message_file'),

    path('messages/<int:message_id>/', views.message_detail, name='message_detail'),

    # 消息模板管理
    path('message-templates/', views.message_templates, name='message_templates'),
    path('message-templates/<int:template_id>/', views.message_template_detail, name='message_template_detail'),

    # 统计
    path('statistics/', views.treatment_statistics, name='treatment_statistics'),
]

