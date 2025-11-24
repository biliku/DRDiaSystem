from django.contrib import admin
from .models import UserProfile, PatientInfo, ConditionInfo


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'is_active', 'created_at', 'updated_at')
    list_filter = ('role', 'is_active', 'created_at')
    search_fields = ('user__username', 'user__email')


@admin.register(PatientInfo)
class PatientInfoAdmin(admin.ModelAdmin):
    list_display = ('user', 'real_name', 'gender', 'age', 'phone', 'created_at', 'updated_at')
    list_filter = ('gender', 'blood_type', 'created_at')
    search_fields = ('user__username', 'real_name', 'phone', 'id_card')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(ConditionInfo)
class ConditionInfoAdmin(admin.ModelAdmin):
    list_display = ('user', 'has_diabetes', 'diabetes_type', 'created_at', 'updated_at')
    list_filter = ('has_diabetes', 'diabetes_type', 'created_at')
    search_fields = ('user__username', 'symptom_description', 'medical_history')
    readonly_fields = ('created_at', 'updated_at')
