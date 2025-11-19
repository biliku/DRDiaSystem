from rest_framework import serializers
from .models import Dataset

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = [
            'id',
            'name',
            'type',
            'description',
            'image_count',
            'status',
            'path',
            'version',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ('id', 'image_count', 'created_at', 'updated_at', 'path', 'status', 'version') # 某些字段由后端管理 