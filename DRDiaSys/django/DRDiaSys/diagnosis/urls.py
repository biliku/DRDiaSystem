from django.urls import path
from . import views

urlpatterns = [
    # 诊断任务
    path('tasks/', views.diagnosis_task_list, name='diagnosis_task_list'),
    path('tasks/create/', views.create_diagnosis_task, name='create_diagnosis_task'),
    path('tasks/<int:task_id>/', views.diagnosis_task_detail, name='diagnosis_task_detail'),
    path('tasks/<int:task_id>/result-image/', views.get_result_image, name='get_result_image'),
    
    # 诊断报告
    path('reports/', views.diagnosis_report_list, name='diagnosis_report_list'),
    path('reports/<int:report_id>/', views.diagnosis_report_detail, name='diagnosis_report_detail'),
    path('reports/<int:report_id>/delete/', views.delete_diagnosis_report, name='delete_diagnosis_report'),
    path('reports/<int:report_id>/review/', views.review_diagnosis_report, name='review_diagnosis_report'),
    path('reports/<int:report_id>/download/', views.download_diagnosis_report, name='download_diagnosis_report'),

    # 病例管理（医生端）
    path('cases/', views.case_records, name='case_records'),
    path('cases/<int:case_id>/', views.case_record_detail, name='case_record_detail'),
    path('cases/<int:case_id>/events/', views.add_case_event, name='add_case_event'),
]

