from django.apps import AppConfig


class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'

    def ready(self):
        """應用程序就緒時執行"""
        # 導入信號處理器
        import users.signals
        
        # 確保系統用戶組存在
        from django.contrib.auth.models import Group
        required_groups = ['admin', 'doctor', 'patient']
        for group_name in required_groups:
            Group.objects.get_or_create(name=group_name)
