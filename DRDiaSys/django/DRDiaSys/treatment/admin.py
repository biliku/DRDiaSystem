from django.contrib import admin
from .models import (
    TreatmentPlan,
    TreatmentPlanExecution,
    TreatmentPlanTemplate,
    Conversation,
    Message,
    MessageTemplate
)


@admin.register(TreatmentPlanTemplate)
class TreatmentPlanTemplateAdmin(admin.ModelAdmin):
    list_display = ['id', 'dr_grade', 'diabetes_type', 'priority', 'is_active', 'created_at']
    list_filter = ['dr_grade', 'diabetes_type', 'is_active']
    search_fields = ['lifestyle_advice', 'precautions']
    ordering = ['-priority', '-created_at']


@admin.register(TreatmentPlan)
class TreatmentPlanAdmin(admin.ModelAdmin):
    list_display = [
        'plan_number',
        'title',
        'case',
        'status',
        'is_ai_recommended',
        'created_by',
        'confirmed_by',
        'created_at'
    ]
    list_filter = ['status', 'is_ai_recommended', 'created_at']
    search_fields = ['plan_number', 'title', 'case__title']
    readonly_fields = ['plan_number', 'created_at', 'updated_at', 'confirmed_at']
    ordering = ['-created_at']


@admin.register(TreatmentPlanExecution)
class TreatmentPlanExecutionAdmin(admin.ModelAdmin):
    list_display = ['plan', 'execution_date', 'follow_up_completed', 'created_by', 'created_at']
    list_filter = ['follow_up_completed', 'execution_date']
    search_fields = ['plan__plan_number', 'patient_feedback']
    ordering = ['-execution_date']


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'patient',
        'doctor',
        'is_active',
        'patient_unread_count',
        'doctor_unread_count',
        'last_message_at'
    ]
    list_filter = ['is_active', 'last_message_at']
    search_fields = ['patient__username', 'doctor__username']
    ordering = ['-last_message_at']


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'conversation',
        'sender',
        'message_type',
        'is_read',
        'is_important',
        'created_at'
    ]
    list_filter = ['message_type', 'is_read', 'is_important', 'created_at']
    search_fields = ['content', 'sender__username']
    ordering = ['-created_at']


@admin.register(MessageTemplate)
class MessageTemplateAdmin(admin.ModelAdmin):
    list_display = ['title', 'doctor', 'category', 'is_public', 'usage_count', 'created_at']
    list_filter = ['is_public', 'category', 'created_at']
    search_fields = ['title', 'content', 'doctor__username']
    ordering = ['-usage_count', '-created_at']
