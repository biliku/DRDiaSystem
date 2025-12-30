# 治疗方案推荐与医患交流模块

## 模块概述

本模块包含两个核心功能：
1. **治疗方案推荐模块** - 基于AI诊断结果推荐治疗方案
2. **医患交流模块** - 医生与患者实时沟通平台

## 安装与配置

### 1. 数据库迁移

```bash
# 创建迁移文件
python manage.py makemigrations treatment

# 执行迁移
python manage.py migrate treatment
```

### 2. 初始化治疗方案模板

在Django Admin中创建治疗方案模板，或通过API创建：

```python
# 示例：创建DR0级治疗方案模板
POST /api/treatment/templates/
{
    "dr_grade": 0,
    "lesion_types": [],
    "medications": [
        {
            "name": "定期复查",
            "dosage": "每6-12个月复查一次",
            "duration": "长期",
            "notes": "保持血糖控制"
        }
    ],
    "follow_up_plan": {
        "next_date": "2024-07-01",
        "check_items": ["眼底检查", "血糖检测"],
        "interval_days": 180
    },
    "lifestyle_advice": "控制血糖，定期复查，注意饮食和运动",
    "precautions": "无明显病变，保持现状即可",
    "priority": 100,
    "is_active": true
}
```

## API接口文档

### 治疗方案推荐

#### 1. 获取治疗方案推荐
```
GET /api/treatment/recommend/?case_id=1&report_id=1
```

#### 2. 基于模板创建方案
```
POST /api/treatment/recommend/
{
    "case_id": 1,
    "template_id": 1,
    "report_id": 1,
    "title": "DR1级治疗方案"
}
```

### 治疗方案管理

#### 1. 获取方案列表
```
GET /api/treatment/plans/?case_id=1&status=active
```

#### 2. 创建治疗方案
```
POST /api/treatment/plans/
{
    "case": 1,
    "related_report": 1,
    "title": "个性化治疗方案",
    "medications": [...],
    "follow_up_plan": {...},
    "lifestyle_advice": "...",
    "precautions": "..."
}
```

#### 3. 获取方案详情
```
GET /api/treatment/plans/{plan_id}/
```

#### 4. 更新方案
```
PATCH /api/treatment/plans/{plan_id}/
{
    "status": "confirmed",
    "medications": [...]
}
```

#### 5. 确认方案
```
POST /api/treatment/plans/{plan_id}/confirm/
```

#### 6. 获取执行记录
```
GET /api/treatment/plans/{plan_id}/executions/
```

#### 7. 创建执行记录
```
POST /api/treatment/plans/{plan_id}/executions/
{
    "execution_date": "2024-01-15",
    "medication_taken": {...},
    "follow_up_completed": true,
    "patient_feedback": "用药后感觉良好"
}
```

### 治疗方案模板管理

#### 1. 获取模板列表
```
GET /api/treatment/templates/?public=true
```

#### 2. 创建模板（管理员）
```
POST /api/treatment/templates/
{
    "dr_grade": 1,
    "medications": [...],
    ...
}
```

### 医患交流 - 会话管理

#### 1. 获取会话列表
```
GET /api/treatment/conversations/
```

#### 2. 创建会话
```
POST /api/treatment/conversations/
{
    "patient_id": 1,  # 医生创建时指定
    "doctor_id": 2,   # 患者创建时指定
    "case_id": 1      # 可选，关联病例
}
```

#### 3. 获取会话详情（包含消息）
```
GET /api/treatment/conversations/{conversation_id}/
```

#### 4. 获取未读消息数
```
GET /api/treatment/conversations/unread-count/
```

### 医患交流 - 消息管理

#### 1. 获取消息列表
```
GET /api/treatment/conversations/{conversation_id}/messages/
```

#### 2. 发送消息
```
POST /api/treatment/conversations/{conversation_id}/messages/
{
    "message_type": "text",
    "content": "您好，请问我的检查结果如何？",
    "related_report": 1,        # 可选
    "related_treatment_plan": 1  # 可选
}
```

#### 3. 上传文件（图片/文件）
```
POST /api/treatment/conversations/{conversation_id}/upload/
Content-Type: multipart/form-data
{
    "file": <file>,
    "file_type": "image"  # 或 "file"
}
```

#### 4. 更新消息（标记已读/重要）
```
PATCH /api/treatment/messages/{message_id}/
{
    "is_read": true,
    "is_important": true
}
```

### 消息模板管理

#### 1. 获取模板列表
```
GET /api/treatment/message-templates/
```

#### 2. 创建模板（医生）
```
POST /api/treatment/message-templates/
{
    "title": "复查提醒",
    "content": "您好，请按时复查眼底检查",
    "category": "复查提醒",
    "is_public": false
}
```

## 权限说明

- **患者**：
  - 查看自己的治疗方案
  - 查看与医生的会话
  - 发送消息给医生
  - 记录方案执行情况

- **医生**：
  - 查看所有相关患者的方案和会话
  - 创建、编辑、确认治疗方案
  - 获取AI推荐方案
  - 创建消息模板

- **管理员**：
  - 所有权限
  - 管理治疗方案模板

## 使用流程示例

### 治疗方案推荐流程

1. 医生查看病例详情
2. 调用 `GET /api/treatment/recommend/?case_id=1&report_id=1` 获取推荐
3. 选择推荐方案或自定义方案
4. 创建治疗方案 `POST /api/treatment/plans/`
5. 确认方案 `POST /api/treatment/plans/{id}/confirm/`
6. 患者查看方案并记录执行情况

### 医患交流流程

1. 患者或医生创建会话 `POST /api/treatment/conversations/`
2. 发送消息 `POST /api/treatment/conversations/{id}/messages/`
3. 对方收到通知并查看消息
4. 持续交流，可关联报告或治疗方案

## 注意事项

1. 所有API都需要JWT认证
2. 文件上传路径：`media/conversations/{conversation_id}/`
3. 治疗方案编号自动生成：`TP{timestamp}{case_id:04d}`
4. AI推荐算法基于规则匹配，可根据实际需求优化

