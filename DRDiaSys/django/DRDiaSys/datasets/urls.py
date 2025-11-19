from django.urls import path
from . import views
# from . import process_views

urlpatterns = [
    path('list_local/', views.list_local_datasets, name='list_local_datasets'),
    path('statistics/', views.get_statistics, name='get_statistics'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('<str:dataset_id>/preview/', views.preview_dataset, name='preview_dataset'),
    path('<str:dataset_id>/image/<path:image_path>', views.get_image, name='get_image'),
    path('<str:dataset_id>/', views.delete_dataset, name='delete_dataset'),
    
    # path('<str:dataset_id>/submit_process/', process_views.submit_process, name='submit_process'),
    # path('process/start/', process_views.start_process, name='start_process'),
    # path('process/<str:task_id>/progress/', process_views.get_process_progress, name='get_process_progress'),
] 