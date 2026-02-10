<template>
  <div class="plan-list">
    <el-table
      :data="plans"
      v-loading="loading"
      empty-text="暂无治疗方案"
      @row-click="handleRowClick"
    >
      <el-table-column prop="plan_number" label="方案编号" width="180" />
      <el-table-column prop="title" label="方案标题" min-width="150" />
      <el-table-column label="状态" width="100">
        <template #default="scope">
          <el-tag :type="getStatusType(scope.row.status)">
            {{ getStatusLabel(scope.row.status) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="创建时间" width="180">
        <template #default="scope">
          {{ formatTime(scope.row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" :width="actionWidth" fixed="right">
        <template #default="scope">
          <!-- 医生端操作按钮 -->
          <template v-if="isDoctor">
            <el-button
              v-if="scope.row.status === 'draft'"
              type="primary"
              link
              size="small"
              @click.stop="$emit('edit', scope.row)"
            >
              编辑
            </el-button>
            <el-button
              v-if="scope.row.status === 'draft'"
              type="success"
              link
              size="small"
              @click.stop="$emit('confirm', scope.row)"
            >
              确认
            </el-button>
            <el-button
              v-if="scope.row.status === 'draft'"
              type="danger"
              link
              size="small"
              @click.stop="$emit('delete', scope.row)"
            >
              删除
            </el-button>
            <el-button
              v-if="scope.row.status === 'active'"
              type="success"
              link
              size="small"
              @click.stop="$emit('complete', scope.row)"
            >
              完成
            </el-button>
          </template>

          <!-- 患者端操作按钮 -->
          <template v-else>
            <el-button
              type="primary"
              link
              size="small"
              @click.stop="$emit('view-detail', scope.row)"
            >
              查看详情
            </el-button>
            <el-button
              v-if="scope.row.status === 'active' || scope.row.status === 'confirmed'"
              type="success"
              link
              size="small"
              @click.stop="$emit('add-execution', scope.row)"
            >
              记录执行
            </el-button>
          </template>

          <!-- 医生端操作按钮：执行记录 -->
          <el-button
            v-if="isDoctor"
            type="info"
            link
            size="small"
            @click.stop="$emit('view-executions', scope.row)"
          >
            执行记录
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 方案详情对话框 -->
    <el-dialog
      v-model="detailDialogVisible"
      title="治疗方案详情"
      width="900px"
      top="5vh"
    >
      <div v-if="selectedPlan" class="plan-detail">
        <!-- 基本信息 -->
        <el-descriptions :column="3" border class="basic-info">
          <el-descriptions-item label="方案编号">{{ selectedPlan.plan_number }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(selectedPlan.status)">
              {{ getStatusLabel(selectedPlan.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="方案标题" :span="2">{{ selectedPlan.title }}</el-descriptions-item>
        </el-descriptions>

          <!-- 基础管理目标 -->
          <el-collapse v-if="isDoctor" v-model="activeNames" class="detail-section">
            <el-collapse-item title="一、基础管理目标" name="basic">
              <div class="info-grid">
                <div class="info-item">
                  <label>血糖控制目标</label>
                  <div class="info-content">
                    <p v-if="formatRange(selectedPlan.blood_sugar_target, 'fasting')">
                      <strong>空腹血糖：</strong>{{ formatRange(selectedPlan.blood_sugar_target, 'fasting') }}
                    </p>
                    <p v-if="formatRange(selectedPlan.blood_sugar_target, 'postprandial')">
                      <strong>餐后血糖：</strong>{{ formatRange(selectedPlan.blood_sugar_target, 'postprandial') }}
                    </p>
                    <p v-if="formatRange(selectedPlan.blood_sugar_target, 'hba1c')">
                      <strong>糖化血红蛋白：</strong>{{ formatRange(selectedPlan.blood_sugar_target, 'hba1c') }}
                    </p>
                  </div>
                </div>
                <div class="info-item">
                  <label>血压控制目标</label>
                  <div class="info-content">
                    <p v-if="formatRange(selectedPlan.blood_pressure_target, 'systolic')">
                      <strong>收缩压：</strong>{{ formatRange(selectedPlan.blood_pressure_target, 'systolic') }}
                    </p>
                    <p v-if="formatRange(selectedPlan.blood_pressure_target, 'diastolic')">
                      <strong>舒张压：</strong>{{ formatRange(selectedPlan.blood_pressure_target, 'diastolic') }}
                    </p>
                  </div>
                </div>
                <div class="info-item full-width">
                  <label>血脂管理</label>
                  <div class="info-content">
                    <p>{{ selectedPlan.lipid_management || '无特殊要求' }}</p>
                  </div>
                </div>
              </div>
            </el-collapse-item>

          <!-- 治疗方案 -->
          <el-collapse-item title="二、治疗方案" name="treatments">
            <el-table :data="selectedPlan.treatments" border stripe v-if="selectedPlan.treatments?.length > 0">
              <el-table-column prop="category" label="治疗类别" width="100">
                <template #default="scope">
                  <el-tag size="small">{{ getTreatmentCategoryLabel(scope.row.category) }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="item" label="具体项目" />
              <el-table-column label="频率" width="120">
                <template #default="scope">
                  {{ scope.row.frequency }}{{ scope.row.frequency_unit }}
                </template>
              </el-table-column>
              <el-table-column label="疗程/间隔" width="120">
                <template #default="scope">
                  {{ scope.row.course }}{{ scope.row.course_unit }}
                </template>
              </el-table-column>
              <el-table-column prop="notes" label="备注/适应症" />
            </el-table>
            <div v-else class="no-data">
              <p>当前无特殊治疗方案</p>
            </div>
          </el-collapse-item>

          <!-- 药物治疗 -->
          <el-collapse-item title="三、药物治疗" name="medications">
            <el-table :data="selectedPlan.medications" border stripe>
              <el-table-column prop="category" label="类别" width="120">
                <template #default="scope">
                  <el-tag size="small">{{ getMedCategoryLabel(scope.row.category) }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="name" label="药物名称" width="150" />
              <el-table-column prop="dosage" label="用法用量" />
              <el-table-column prop="duration" label="疗程" width="100" />
              <el-table-column prop="notes" label="备注" />
            </el-table>
          </el-collapse-item>

          <!-- 生活方式干预 -->
          <el-collapse-item title="四、生活方式干预" name="lifestyle">
            <div class="lifestyle-section">
              <div v-if="selectedPlan.diet_guidance" class="lifestyle-item">
                <h5><el-icon><Food /></el-icon> 饮食指导</h5>
                <div class="lifestyle-content" v-html="formatLifestyle(selectedPlan.diet_guidance)"></div>
              </div>
              <div v-if="selectedPlan.exercise_guidance" class="lifestyle-item">
                <h5><el-icon><Grid /></el-icon> 运动指导</h5>
                <div class="lifestyle-content" v-html="formatLifestyle(selectedPlan.exercise_guidance)"></div>
              </div>
              <div v-if="selectedPlan.lifestyle_advice" class="lifestyle-item">
                <h5><el-icon><InfoFilled /></el-icon> 综合建议</h5>
                <div class="lifestyle-content" v-html="formatLifestyle(selectedPlan.lifestyle_advice)"></div>
              </div>
            </div>
          </el-collapse-item>

          <!-- 随访监测 -->
          <el-collapse-item title="五、随访监测计划" name="monitoring">
            <div class="info-grid">
              <div class="info-item">
                <label>复查计划</label>
                <div class="info-content">
                  <p v-if="selectedPlan.follow_up_plan?.interval_days">
                    <strong>复查间隔：</strong>{{ selectedPlan.follow_up_plan.interval_days }} 天
                  </p>
                  <p v-if="selectedPlan.follow_up_plan?.check_items?.length">
                    <strong>检查项目：</strong>{{ selectedPlan.follow_up_plan.check_items.join('、') }}
                  </p>
                  <p v-if="selectedPlan.follow_up_plan?.next_date">
                    <strong>下次复查：</strong>{{ selectedPlan.follow_up_plan.next_date }}
                  </p>
                  <p v-if="selectedPlan.follow_up_plan?.notes">
                    <strong>备注：</strong>{{ selectedPlan.follow_up_plan.notes }}
                  </p>
                </div>
              </div>
              <div class="info-item">
                <label>日常监测项目</label>
                <div class="info-content">
                  <el-table v-if="selectedPlan.monitoring_plan?.length" :data="selectedPlan.monitoring_plan" size="small" border>
                    <el-table-column prop="item" label="监测项目" />
                    <el-table-column prop="frequency" label="频率" width="120" />
                    <el-table-column prop="target" label="目标值" />
                  </el-table>
                  <p v-else class="no-data">无特定监测计划</p>
                </div>
              </div>
              <div class="info-item full-width">
                <label>预警症状</label>
                <div class="info-content warning">
                  <p class="warning-text">{{ selectedPlan.warning_symptoms || '无特殊预警症状' }}</p>
                </div>
              </div>
            </div>
          </el-collapse-item>

          <!-- 注意事项 -->
          <el-collapse-item v-if="isDoctor" title="六、注意事项" name="precautions">
            <div class="precautions-content">
              <p>{{ selectedPlan.precautions || '无特殊注意事项' }}</p>
            </div>
          </el-collapse-item>
        </el-collapse>

        <!-- 患者端简化视图 -->
        <div v-else class="patient-view">
          <!-- 药物治疗 -->
          <div class="patient-section">
            <h4><el-icon><FirstAidKit /></el-icon> 药物治疗</h4>
            <el-table :data="selectedPlan.medications" border stripe size="small">
              <el-table-column prop="name" label="药物名称" width="120" />
              <el-table-column prop="dosage" label="用法用量" />
              <el-table-column prop="duration" label="疗程" width="100" />
              <el-table-column prop="notes" label="备注" />
            </el-table>
          </div>

          <!-- 生活方式 -->
          <div class="patient-section" v-if="selectedPlan.diet_guidance || selectedPlan.exercise_guidance">
            <h4><el-icon><Food /></el-icon> 生活指导</h4>
            <div v-if="selectedPlan.diet_guidance" class="patient-subsection">
              <h5>饮食建议</h5>
              <div v-html="formatLifestyle(selectedPlan.diet_guidance)"></div>
            </div>
            <div v-if="selectedPlan.exercise_guidance" class="patient-subsection">
              <h5>运动建议</h5>
              <div v-html="formatLifestyle(selectedPlan.exercise_guidance)"></div>
            </div>
          </div>

          <!-- 复查计划 -->
          <div class="patient-section">
            <h4><el-icon><Calendar /></el-icon> 复查计划</h4>
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item label="复查间隔">
                {{ selectedPlan.follow_up_plan?.interval_days || '-' }} 天
              </el-descriptions-item>
              <el-descriptions-item label="检查项目">
                {{ selectedPlan.follow_up_plan?.check_items?.join('、') || '待定' }}
              </el-descriptions-item>
              <el-descriptions-item label="下次复查" :span="2">
                {{ selectedPlan.follow_up_plan?.next_date || '待定' }}
              </el-descriptions-item>
            </el-descriptions>
          </div>

          <!-- 预警症状 -->
          <div v-if="selectedPlan.warning_symptoms" class="patient-section warning-section">
            <h4><el-icon><WarningFilled /></el-icon> 预警症状</h4>
            <div class="warning-content" v-html="formatLifestyle(selectedPlan.warning_symptoms)"></div>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage } from 'element-plus'
import { 
  Food, Grid, InfoFilled, FirstAidKit, 
  Calendar, WarningFilled 
} from '@element-plus/icons-vue'

export default {
  name: 'PlanList',
  components: {
    Food, Grid, InfoFilled, FirstAidKit, 
    Calendar, WarningFilled
  },
  props: {
    caseId: {
      type: Number,
      required: false
    },
    status: {
      type: String,
      default: ''
    },
    isDoctor: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      loading: false,
      plans: [],
      detailDialogVisible: false,
      selectedPlan: null,
      activeNames: ['basic', 'medications', 'lifestyle', 'monitoring']
    }
  },
  computed: {
    actionWidth() {
      return this.isDoctor ? 280 : 200
    }
  },
  created() {
    this.fetchPlans()
  },
  watch: {
    caseId() {
      this.fetchPlans()
    },
    status() {
      this.fetchPlans()
    }
  },
  methods: {
    async fetchPlans() {
      this.loading = true
      try {
        let url = '/api/treatment/plans/'
        const params = {}

        if (this.isDoctor && this.caseId) {
          params.case_id = this.caseId
        }

        if (this.status && this.status !== 'all') {
          params.status = this.status
        }

        const res = await api.get(url, { params })
        this.plans = res.data
      } catch (error) {
        ElMessage.error('获取方案列表失败')
      } finally {
        this.loading = false
      }
    },
    handleRowClick(row) {
      this.selectedPlan = row
      this.detailDialogVisible = true
    },
    getStatusType(status) {
      const map = {
        draft: 'info',
        confirmed: 'warning',
        active: 'success',
        completed: '',
        cancelled: 'danger'
      }
      return map[status] || ''
    },
    getStatusLabel(status) {
      const map = {
        draft: '草稿',
        confirmed: '已确认',
        active: '执行中',
        completed: '已完成',
        cancelled: '已取消'
      }
      return map[status] || status
    },
    formatTime(time) {
      if (!time) return '-'
      return new Date(time).toLocaleString('zh-CN')
    },
    getMedCategoryLabel(category) {
      const map = {
        'ophthalmic': '眼科用药',
        'systemic': '全身用药',
        'injection': '注射用药',
        'nutrient': '营养补充'
      }
      return map[category] || category || '其他'
    },
    getTreatmentCategoryLabel(category) {
      const map = {
        'anti_vegf': '抗VEGF',
        'laser': '激光治疗',
        'surgical': '手术治疗',
        'other': '其他治疗'
      }
      return map[category] || category || '未知'
    },
    formatLifestyle(text) {
      if (!text) return ''
      // 将换行和特殊字符转换为HTML格式
      return text
        .replace(/\n/g, '<br>')
        .replace(/•\s*/g, '<span class="bullet">•</span>')
    },
    formatRange(target, field) {
      if (!target) return ''
      const min = target[`${field}_min`]
      const max = target[`${field}_max`]
      const unit = target[`${field}_unit`] || ''
      if (min && max) {
        return `${min} — ${max} ${unit}`
      } else if (min) {
        return `${min} ${unit}`
      } else if (max) {
        return `${max} ${unit}`
      }
      return ''
    }
  }
}
</script>

<style scoped>
.plan-list {
  margin-top: 20px;
}

.plan-detail {
  max-height: 70vh;
  overflow-y: auto;
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
  margin-bottom: 8px;
  display: block;
}

.info-content p {
  margin: 4px 0;
  font-size: 14px;
  color: #606266;
}

.info-content .no-data {
  color: #909399;
  font-style: italic;
}

.info-content .warning {
  background: #fdf6ec;
  padding: 10px;
  border-radius: 4px;
  border-left: 3px solid #e6a23c;
}

.warning-text {
  color: #e6a23c;
  font-weight: 500;
}

.lifestyle-section {
  display: grid;
  gap: 15px;
}

.lifestyle-item {
  background: #f0f9eb;
  padding: 12px;
  border-radius: 4px;
}

.lifestyle-item h5 {
  margin: 0 0 8px 0;
  color: #67c23a;
  display: flex;
  align-items: center;
  gap: 6px;
}

.lifestyle-content {
  font-size: 14px;
  line-height: 1.8;
  color: #606266;
}

.lifestyle-content :deep(.bullet) {
  margin-right: 5px;
}

.precautions-content {
  background: #fef0f0;
  padding: 15px;
  border-radius: 4px;
  color: #f56c6c;
  line-height: 1.6;
}

/* 患者端样式 */
.patient-view {
  max-height: 65vh;
  overflow-y: auto;
}

.patient-section {
  margin-bottom: 20px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
}

.patient-section h4 {
  margin: 0 0 12px 0;
  color: #303133;
  display: flex;
  align-items: center;
  gap: 8px;
}

.patient-subsection {
  margin-bottom: 12px;
}

.patient-subsection h5 {
  margin: 0 0 8px 0;
  color: #606266;
}

.warning-section {
  background: #fdf6ec;
  border: 1px solid #e6a23c;
}

.warning-content {
  color: #e6a23c;
  line-height: 1.8;
}

.basic-info {
  margin-bottom: 15px;
}
</style>
