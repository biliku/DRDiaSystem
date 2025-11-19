from django.db.models.signals import post_migrate
from django.dispatch import receiver
from django.contrib.auth.models import Group

@receiver(post_migrate)
def create_default_groups(sender, **kwargs):
    """在數據庫遷移後創建默認用戶組"""
    # 創建基本角色
    required_groups = ['admin', 'doctor', 'patient']
    for group_name in required_groups:
        Group.objects.get_or_create(name=group_name)
        print(f"創建或確認用戶組: {group_name}") 