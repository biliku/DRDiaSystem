<template>
  <div class="patient-treatment-plan-page">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <h3>我的治疗方案</h3>
          <el-button type="primary" @click="fetchPlans">
            <el-icon><RefreshRight /></el-icon>
            刷新
          </el-button>
        </div>
      </template>

      <el-tabs v-model="activeTab">
        <el-tab-pane label="执行中" name="active">
          <PlanList :plans="activePlans" :loading="loading" @view-detail="handleViewDetail" @add-execution="handleAddExecution" />
        </el-tab-pane>
        <el-tab-pane label="已完成" name="completed">
          <PlanList :plans="completedPlans" :loading="loading" @view-detail="handleViewDetail" />
        </el-tab-pane>
        <el-tab-pane label="全部" name="all">
          <PlanList :plans="allPlans" :loading="loading" @view-detail="handleViewDetail" @add-execution="handleAddExecution" />
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 方案详情对话框 -->
    <el-dialog
      v-model="detailDialogVisible"
      title="治疗方案详情"
      width="800px"
    >
      <div v-if="selectedPlan">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="方案编号">{{ selectedPlan.plan_number }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(selectedPlan.status)">
              {{ getStatusLabel(selectedPlan.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="方案标题" :span="2">{{ selectedPlan.title }}</el-descriptions-item>
        </el-descriptions>

        <h4 style="margin-top: 20px">药物治疗</h4>
        <el-table :data="selectedPlan.medications" border>
          <el-table-column prop="name" label="药物名称" />
          <el-table-column prop="dosage" label="用法用量" />
          <el-table-column prop="duration" label="疗程" />
        </el-table>

        <h4 style="margin-top: 20px">复查计划</h4>
        <el-descriptions :column="1" border>
          <el-descriptions-item label="下次复查日期">
            {{ selectedPlan.follow_up_plan?.next_date || '待定' }}
          </el-descriptions-item>
          <el-descriptions-item label="检查项目">
            {{ selectedPlan.follow_up_plan?.check_items?.join('、') || '待定' }}
          </el-descriptions-item>
        </el-descriptions>

        <h4 style="margin-top: 20px">生活方式建议</h4>
        <p>{{ selectedPlan.lifestyle_advice || '无' }}</p>

        <h4 style="margin-top: 20px">注意事项</h4>
        <p>{{ selectedPlan.precautions || '无' }}</p>
      </div>
    </el-dialog>

    <!-- 添加执行记录对话框 -->
    <el-dialog
      v-model="executionDialogVisible"
      title="记录执行情况"
      width="600px"
    >
      <el-form :model="executionForm" label-width="120px">
        <el-form-item label="执行日期" required>
          <el-date-picker
            v-model="executionForm.execution_date"
            type="date"
            placeholder="选择日期"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="用药情况">
          <div v-for="(med, index) in selectedPlan?.medications || []" :key="index" class="medication-item">
            <span style="width: 150px; display: inline-block">{{ med.name }}</span>
            <el-input
              v-model="executionForm.medication_taken[med.name]"
              placeholder="是否按时用药"
              style="width: 300px"
            />
          </div>
        </el-form-item>
        <el-form-item label="复查完成">
          <el-switch v-model="executionForm.follow_up_completed" />
        </el-form-item>
        <el-form-item label="我的反馈">
          <el-input
            v-model="executionForm.patient_feedback"
            type="textarea"
            :rows="4"
            placeholder="请记录用药后的感受、症状变化等"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="executionDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveExecution" :loading="saving">提交</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage } from 'element-plus'
import { RefreshRight } from '@element-plus/icons-vue'
import PlanList from './components/PatientPlanList.vue'

export default {
  name: 'PatientTreatmentPlan',
  components: {
    RefreshRight,
    PlanList
  },
  data() {
    return {
      loading: false,
      plans: [],
      activeTab: 'active',
      detailDialogVisible: false,
      selectedPlan: null,
      executionDialogVisible: false,
      executionForm: {
        execution_date: new Date(),
        medication_taken: {},
        follow_up_completed: false,
        patient_feedback: ''
      },
      saving: false
    }
  },
  computed: {
    activePlans() {
      return this.plans.filter(p => p.status === 'active' || p.status === 'confirmed')
    },
    completedPlans() {
      return this.plans.filter(p => p.status === 'completed')
    },
    allPlans() {
      return this.plans
    }
  },
  created() {
    this.fetchPlans()
  },
  methods: {
    async fetchPlans() {
      this.loading = true
      try {
        const res = await api.get('/api/treatment/plans/')
        this.plans = res.data
      } catch (error) {
        ElMessage.error('获取治疗方案失败')
      } finally {
        this.loading = false
      }
    },
    handleViewDetail(plan) {
      this.selectedPlan = plan
      this.detailDialogVisible = true
    },
    handleAddExecution(plan) {
      this.selectedPlan = plan
      this.executionForm = {
        execution_date: this.formatDate(new Date()),
        medication_taken: {},
        follow_up_completed: false,
        patient_feedback: ''
      }
      // 初始化用药记录
      const meds = {}
      plan.medications?.forEach(med => {
        if (med?.name) {
          meds[med.name] = ''
        }
      })
      this.executionForm.medication_taken = meds
      this.executionDialogVisible = true
    },
    async saveExecution() {
      if (!this.executionForm.execution_date) {
        ElMessage.warning('请选择执行日期')
        return
      }
      this.saving = true
      try {
        const payload = {
          ...this.executionForm,
          execution_date: this.formatDate(this.executionForm.execution_date)
        }
        await api.post(`/api/treatment/plans/${this.selectedPlan.id}/executions/`, payload)
        ElMessage.success('执行记录提交成功')
        this.executionDialogVisible = false
        this.fetchPlans()
      } catch (error) {
        ElMessage.error('提交执行记录失败')
      } finally {
        this.saving = false
      }
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
    formatDate(val) {
      if (!val) return ''
      // 如果是字符串（已是日期），直接返回
      if (typeof val === 'string') {
        return val.split('T')[0]
      }
      // Date对象转 yyyy-mm-dd
      const d = new Date(val)
      if (isNaN(d.getTime())) return ''
      const y = d.getFullYear()
      const m = `${d.getMonth() + 1}`.padStart(2, '0')
      const day = `${d.getDate()}`.padStart(2, '0')
      return `${y}-${m}-${day}`
    }
  }
}
</script>

<style scoped>
.patient-treatment-plan-page {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h3 {
  margin: 0;
}

.medication-item {
  margin-bottom: 10px;
}
</style>

