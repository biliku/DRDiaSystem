<template>
  <div class="template-list">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>治疗方案模板</span>
          <el-tag type="info">DR {{ selectedGrade || '全部' }}级</el-tag>
        </div>
      </template>

      <!-- DR分级选择 -->
      <div class="grade-filter">
        <el-radio-group v-model="selectedGrade" @change="fetchTemplates">
          <el-radio-button :label="null">全部</el-radio-button>
          <el-radio-button :label="0">无DR</el-radio-button>
          <el-radio-button :label="1">轻度</el-radio-button>
          <el-radio-button :label="2">中度</el-radio-button>
          <el-radio-button :label="3">重度</el-radio-button>
          <el-radio-button :label="4">PDR</el-radio-button>
        </el-radio-group>
      </div>

      <el-table :data="templates" v-loading="loading" empty-text="暂无模板">
        <el-table-column prop="dr_grade" label="DR分级" width="100">
          <template #default="scope">
            <el-tag :type="getGradeType(scope.row.dr_grade)">
              {{ getGradeLabel(scope.row.dr_grade) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="title" label="模板名称" min-width="200" />
        <el-table-column prop="priority" label="优先级" width="80" align="center" />
        <el-table-column label="操作" width="120" fixed="right">
          <template #default="scope">
            <el-button type="primary" link size="small" @click="viewDetail(scope.row)">
              查看详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 模板详情对话框 -->
    <el-dialog
      v-model="detailVisible"
      :title="selectedTemplate?.title"
      width="900px"
      top="5vh"
    >
      <div v-if="selectedTemplate" class="template-detail">
        <el-descriptions :column="3" border>
          <el-descriptions-item label="DR分级">
            <el-tag :type="getGradeType(selectedTemplate.dr_grade)">
              {{ getGradeLabel(selectedTemplate.dr_grade) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="优先级">{{ selectedTemplate.priority }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="selectedTemplate.is_active ? 'success' : 'info'">
              {{ selectedTemplate.is_active ? '启用' : '停用' }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <el-collapse v-model="activeNames" class="detail-section">
          <!-- 基础管理目标 -->
          <el-collapse-item title="一、基础管理目标" name="basic">
            <div class="info-grid">
              <div class="info-item">
                <label>血糖控制目标</label>
                <div class="info-content">
                  <p><strong>空腹血糖：</strong>{{ selectedTemplate.blood_sugar_target?.fasting || '未设置' }}</p>
                  <p><strong>餐后血糖：</strong>{{ selectedTemplate.blood_sugar_target?.postprandial || '未设置' }}</p>
                  <p><strong>HbA1c：</strong>{{ selectedTemplate.blood_sugar_target?.hba1c || '未设置' }}</p>
                </div>
              </div>
              <div class="info-item">
                <label>血压控制目标</label>
                <div class="info-content">
                  <p><strong>收缩压：</strong>{{ selectedTemplate.blood_pressure_target?.systolic || '未设置' }}</p>
                  <p><strong>舒张压：</strong>{{ selectedTemplate.blood_pressure_target?.diastolic || '未设置' }}</p>
                </div>
              </div>
              <div class="info-item full-width">
                <label>血脂管理</label>
                <div class="info-content">
                  <p>{{ selectedTemplate.lipid_management || '未设置' }}</p>
                </div>
              </div>
            </div>
          </el-collapse-item>

          <!-- 眼科治疗 -->
          <el-collapse-item title="二、眼科治疗" name="ophthalmic">
            <div class="info-grid">
              <div class="info-item">
                <label>抗VEGF治疗</label>
                <div class="info-content" v-if="selectedTemplate.anti_vegf_treatment?.drug">
                  <p><strong>药物：</strong>{{ selectedTemplate.anti_vegf_treatment.drug }}</p>
                  <p><strong>适应症：</strong>{{ selectedTemplate.anti_vegf_treatment.indication }}</p>
                  <p><strong>频率：</strong>{{ selectedTemplate.anti_vegf_treatment.frequency }}</p>
                  <p><strong>疗程：</strong>{{ selectedTemplate.anti_vegf_treatment.course }}</p>
                </div>
                <p v-else class="no-data">当前无需抗VEGF治疗</p>
              </div>
              <div class="info-item">
                <label>激光治疗</label>
                <div class="info-content" v-if="selectedTemplate.laser_treatment?.type">
                  <p><strong>类型：</strong>{{ selectedTemplate.laser_treatment.type }}</p>
                  <p><strong>次数：</strong>{{ selectedTemplate.laser_treatment.sessions }}</p>
                  <p><strong>间隔：</strong>{{ selectedTemplate.laser_treatment.interval }}</p>
                </div>
                <p v-else class="no-data">当前无需激光治疗</p>
              </div>
              <div class="info-item full-width">
                <label>手术治疗指征</label>
                <div class="info-content">
                  <p>{{ selectedTemplate.surgical_treatment || '当前无手术指征' }}</p>
                </div>
              </div>
            </div>
          </el-collapse-item>

          <!-- 药物治疗 -->
          <el-collapse-item title="三、药物治疗" name="medications">
            <el-table :data="selectedTemplate.medications" border stripe>
              <el-table-column prop="category" label="类别" width="100">
                <template #default="scope">
                  <el-tag size="small">{{ scope.row.category }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="name" label="药物名称" />
              <el-table-column prop="dosage" label="用法用量" />
              <el-table-column prop="duration" label="疗程" width="100" />
              <el-table-column prop="notes" label="备注" />
            </el-table>
          </el-collapse-item>

          <!-- 生活方式 -->
          <el-collapse-item title="四、生活方式干预" name="lifestyle">
            <div v-if="selectedTemplate.diet_guidance" class="lifestyle-item">
              <h5>饮食指导</h5>
              <div class="lifestyle-content">{{ selectedTemplate.diet_guidance }}</div>
            </div>
            <div v-if="selectedTemplate.exercise_guidance" class="lifestyle-item">
              <h5>运动指导</h5>
              <div class="lifestyle-content">{{ selectedTemplate.exercise_guidance }}</div>
            </div>
            <div v-if="selectedTemplate.lifestyle_advice" class="lifestyle-item">
              <h5>综合建议</h5>
              <div class="lifestyle-content">{{ selectedTemplate.lifestyle_advice }}</div>
            </div>
          </el-collapse-item>

          <!-- 随访监测 -->
          <el-collapse-item title="五、随访监测" name="monitoring">
            <div class="info-grid">
              <div class="info-item">
                <label>复查计划</label>
                <div class="info-content">
                  <p><strong>间隔：</strong>{{ selectedTemplate.follow_up_plan?.interval_days || '未设置' }} 天</p>
                  <p><strong>项目：</strong>{{ selectedTemplate.follow_up_plan?.check_items?.join('、') || '未设置' }}</p>
                </div>
              </div>
              <div class="info-item">
                <label>预警症状</label>
                <div class="info-content warning">
                  <p>{{ selectedTemplate.warning_symptoms || '无' }}</p>
                </div>
              </div>
            </div>
          </el-collapse-item>

          <!-- 注意事项 -->
          <el-collapse-item title="六、注意事项" name="precautions">
            <div class="precautions-content">
              {{ selectedTemplate.precautions || '无' }}
            </div>
          </el-collapse-item>
        </el-collapse>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage } from 'element-plus'

export default {
  name: 'TreatmentTemplateList',
  data() {
    return {
      loading: false,
      templates: [],
      selectedGrade: null,
      detailVisible: false,
      selectedTemplate: null,
      activeNames: ['basic', 'ophthalmic', 'medications', 'lifestyle', 'monitoring']
    }
  },
  created() {
    this.fetchTemplates()
  },
  methods: {
    async fetchTemplates() {
      this.loading = true
      try {
        const params = {}
        if (this.selectedGrade !== null) {
          params.dr_grade = this.selectedGrade
        }
        const res = await api.get('/api/treatment/templates/', { params })
        this.templates = res.data
      } catch (error) {
        ElMessage.error('获取模板列表失败')
      } finally {
        this.loading = false
      }
    },
    viewDetail(row) {
      this.selectedTemplate = row
      this.detailVisible = true
    },
    getGradeType(grade) {
      const map = { 0: 'success', 1: 'info', 2: 'warning', 3: 'danger', 4: 'danger' }
      return map[grade] || 'info'
    },
    getGradeLabel(grade) {
      const map = { 0: '无DR', 1: '轻度', 2: '中度', 3: '重度', 4: 'PDR' }
      return map[grade] || grade
    }
  }
}
</script>

<style scoped>
.template-list {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.grade-filter {
  margin-bottom: 20px;
  text-align: center;
}

.detail-section {
  margin-top: 15px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
}

.info-item {
  background: #f8f9fa;
  padding: 12px;
  border-radius: 4px;
}

.info-item.full-width {
  grid-column: span 2;
}

.info-item label {
  font-weight: bold;
  color: #303133;
  display: block;
  margin-bottom: 8px;
}

.info-content p {
  margin: 4px 0;
  font-size: 14px;
}

.info-content .no-data {
  color: #909399;
  font-style: italic;
}

.info-content .warning {
  background: #fdf6ec;
  padding: 10px;
  border-radius: 4px;
  color: #e6a23c;
}

.lifestyle-item {
  background: #f0f9eb;
  padding: 12px;
  border-radius: 4px;
  margin-bottom: 12px;
}

.lifestyle-item h5 {
  margin: 0 0 8px 0;
  color: #67c23a;
}

.lifestyle-content {
  white-space: pre-wrap;
  font-size: 14px;
  line-height: 1.8;
}

.precautions-content {
  background: #fef0f0;
  padding: 15px;
  border-radius: 4px;
  color: #f56c6c;
  white-space: pre-wrap;
}
</style>

