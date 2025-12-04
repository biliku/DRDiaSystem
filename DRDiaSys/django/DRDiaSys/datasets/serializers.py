from rest_framework import serializers
from .models import Dataset, PatientImage

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


class PatientImageSerializer(serializers.ModelSerializer):
    owner_name = serializers.CharField(source='owner.username', read_only=True)
    uploader_name = serializers.CharField(source='uploaded_by.username', read_only=True)

    class Meta:
        model = PatientImage
        fields = [
            'id',
            'owner',
            'owner_name',
            'uploaded_by',
            'uploader_name',
            'original_name',
            'stored_path',
            'dataset_folder',
            'eye_side',
            'description',
            'file_size',
            'created_at',
            'updated_at',
        ]
        read_only_fields = (
            'id',
            'stored_path',
            'dataset_folder',
            'file_size',
            'created_at',
            'updated_at',
            'owner_name',
            'uploader_name',
        )