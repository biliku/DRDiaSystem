<template>
  <div class="patient-plan-page">
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
          <PlanList
            :is-doctor="false"
            status="active"
            @view-detail="handleViewDetail"
            @add-execution="handleAddExecution"
          />
        </el-tab-pane>
        <el-tab-pane label="已完成" name="completed">
          <PlanList
            :is-doctor="false"
            status="completed"
            @view-detail="handleViewDetail"
          />
        </el-tab-pane>
        <el-tab-pane label="全部" name="all">
          <PlanList
            :is-doctor="false"
            @view-detail="handleViewDetail"
            @add-execution="handleAddExecution"
          />
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 方案详情对话框 -->
    <el-dialog
      v-model="detailDialogVisible"
      title="治疗方案详情"
      width="850px"
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
        <div class="detail-section">
          <div class="section-title">
            <el-icon><Aim /></el-icon>
            <h4>一、基础管理目标</h4>
          </div>
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="空腹血糖">
              {{ formatRange(selectedPlan.blood_sugar_target, 'fasting') }}
            </el-descriptions-item>
            <el-descriptions-item label="餐后血糖">
              {{ formatRange(selectedPlan.blood_sugar_target, 'postprandial') }}
            </el-descriptions-item>
            <el-descriptions-item label="HbA1c">
              {{ formatRange(selectedPlan.blood_sugar_target, 'hba1c') }}
            </el-descriptions-item>
            <el-descriptions-item label="血压目标">
              {{ formatRange(selectedPlan.blood_pressure_target, 'systolic') }} / 
              {{ formatRange(selectedPlan.blood_pressure_target, 'diastolic') }}
            </el-descriptions-item>
            <el-descriptions-item label="血脂管理" :span="2">
              {{ selectedPlan.lipid_management || '未设置' }}
            </el-descriptions-item>
          </el-descriptions>
        </div>

        <!-- 药物治疗 -->
        <div class="detail-section">
          <div class="section-title">
            <el-icon><FirstAidKit /></el-icon>
            <h4>二、药物治疗</h4>
          </div>
          <el-table :data="selectedPlan.medications" border size="small" stripe v-if="selectedPlan.medications?.length > 0">
            <el-table-column prop="category" label="类别" width="100">
              <template #default="scope">
                <el-tag size="small">{{ getMedCategoryLabel(scope.row.category) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="name" label="药物名称" width="120" />
            <el-table-column prop="dosage" label="用法用量" />
            <el-table-column prop="duration" label="疗程" width="100" />
            <el-table-column prop="notes" label="备注" />
          </el-table>
          <el-empty v-else description="无药物治疗" :image-size="60" />
        </div>

        <!-- 治疗方案 -->
        <div class="detail-section" v-if="selectedPlan.treatments?.length > 0">
          <div class="section-title">
            <el-icon><Operation /></el-icon>
            <h4>三、治疗方案</h4>
          </div>
          <el-table :data="selectedPlan.treatments" border size="small" stripe>
            <el-table-column prop="category" label="治疗类别" width="110">
              <template #default="scope">
                <el-tag size="small">{{ getTreatmentCategoryLabel(scope.row.category) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="item" label="具体项目" />
            <el-table-column prop="frequency" label="频率/次数" width="120" />
            <el-table-column prop="course" label="疗程/间隔" width="110" />
            <el-table-column prop="notes" label="备注/适应症" />
          </el-table>
        </div>

        <!-- 生活方式 -->
        <div class="detail-section" v-if="selectedPlan.diet_guidance || selectedPlan.exercise_guidance">
          <div class="section-title">
            <el-icon><Food /></el-icon>
            <h4>三、生活指导</h4>
          </div>
          <div class="lifestyle-content">
            <div v-if="selectedPlan.diet_guidance" class="lifestyle-item">
              <h5>饮食建议</h5>
              <div class="content-text">{{ selectedPlan.diet_guidance }}</div>
            </div>
            <div v-if="selectedPlan.exercise_guidance" class="lifestyle-item">
              <h5>运动建议</h5>
              <div class="content-text">{{ selectedPlan.exercise_guidance }}</div>
            </div>
          </div>
        </div>

        <!-- 复查计划 -->
        <div class="detail-section">
          <div class="section-title">
            <el-icon><Calendar /></el-icon>
            <h4>四、复查计划</h4>
          </div>
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="复查间隔">
              {{ selectedPlan.follow_up_plan?.interval_days || '未设置' }} 天
            </el-descriptions-item>
            <el-descriptions-item label="检查项目">
              {{ selectedPlan.follow_up_plan?.check_items?.join('、') || '待定' }}
            </el-descriptions-item>
            <el-descriptions-item label="下次复查">
              {{ selectedPlan.follow_up_plan?.next_date || '待定' }}
            </el-descriptions-item>
          </el-descriptions>
        </div>

        <!-- 预警症状 -->
        <div v-if="selectedPlan.warning_symptoms" class="detail-section warning-section">
          <div class="section-title">
            <el-icon><WarningFilled /></el-icon>
            <h4>五、预警症状</h4>
          </div>
          <div class="warning-content">
            {{ selectedPlan.warning_symptoms }}
          </div>
        </div>

        <!-- 执行记录日历 -->
        <div v-if="executions.length > 0 || planStartDate" class="detail-section">
          <!-- 补录提示 -->
          <div v-if="planStartDate && missedDays > 0" class="missed-tip">
            <el-icon><InfoFilled /></el-icon>
            <span>该方案自 {{ planStartDate }} 起，有 <strong>{{ missedDays }}</strong> 天未记录，可点击日期补录</span>
          </div>
          
          <div class="section-header">
            <div class="section-title">
              <el-icon><List /></el-icon>
              <h4>我的执行记录</h4>
            </div>
            <div class="exec-stats">
              <span class="exec-count">共 {{ executions.length }} 条记录</span>
              <span v-if="planStartDate" class="missed-count">漏记 {{ missedDays }} 天</span>
            </div>
          </div>
          <el-calendar v-model="currentMonth" class="small-calendar">
            <template #date-cell="{ data }">
              <div
                class="calendar-date"
                :class="{
                  'is-executed': isExecutedDate(data.day),
                  'is-selected': selectedExecutionDate === data.day,
                  'is-missed': isMissedDate(data.day)
                }"
                @click="viewExecutionDetail(data.day)"
              >
                {{ data.day.split('-')[2] }}
                <span v-if="isExecutedDate(data.day)" class="executed-dot"></span>
                <span v-if="isMissedDate(data.day)" class="missed-dot">✕</span>
              </div>
            </template>
          </el-calendar>
          <!-- 图例说明 -->
          <div class="calendar-legend">
            <div class="legend-item"><span class="dot executed"></span> 已记录</div>
            <div class="legend-item"><span class="dot missed"></span> 未记录</div>
          </div>
        </div>
        <div v-else class="detail-section no-executions">
          <el-empty description="暂无执行记录，点击下方按钮开始记录" :image-size="60" />
        </div>
      </div>
    </el-dialog>

    <!-- 执行详情对话框 -->
    <el-dialog
      v-model="executionDetailDialogVisible"
      title="执行详情"
      width="650px"
    >
      <div v-if="selectedExecution">
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="执行日期">{{ selectedExecution.execution_date }}</el-descriptions-item>
          <el-descriptions-item label="提交时间">
            {{ formatDateTime(selectedExecution.created_at) }}
          </el-descriptions-item>
        </el-descriptions>

        <!-- 用药情况 -->
        <el-divider content-position="left">用药情况</el-divider>
        <el-table :data="medicationList" border size="small">
          <el-table-column prop="name" label="药物名称" width="120" />
          <el-table-column label="是否按时用药" width="150">
            <template #default="{ row }">
              <el-tag :type="selectedExecution.medication_taken?.[row.name] === 'yes' ? 'success' : 'danger'" size="small">
                {{ selectedExecution.medication_taken?.[row.name] === 'yes' ? '是' : selectedExecution.medication_taken?.[row.name] === 'no' ? '否' : '未记录' }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
        <p v-if="selectedExecution.medication_notes" class="detail-notes">
          <strong>用药备注：</strong>{{ selectedExecution.medication_notes }}
        </p>

        <!-- 血糖记录 -->
        <el-divider content-position="left">血糖记录</el-divider>
        <el-descriptions :column="3" border size="small">
          <el-descriptions-item label="空腹血糖">
            {{ selectedExecution.blood_sugar_fasting || '-' }} <span class="unit">mmol/L</span>
          </el-descriptions-item>
          <el-descriptions-item label="餐后血糖">
            {{ selectedExecution.blood_sugar_postprandial || '-' }} <span class="unit">mmol/L</span>
          </el-descriptions-item>
          <el-descriptions-item label="糖化血红蛋白">
            {{ selectedExecution.blood_sugar_hba1c || '-' }} <span class="unit">%</span>
          </el-descriptions-item>
        </el-descriptions>

        <!-- 血压记录 -->
        <el-divider content-position="left">血压记录</el-divider>
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="收缩压">
            {{ selectedExecution.blood_pressure_systolic || '-' }} <span class="unit">mmHg</span>
          </el-descriptions-item>
          <el-descriptions-item label="舒张压">
            {{ selectedExecution.blood_pressure_diastolic || '-' }} <span class="unit">mmHg</span>
          </el-descriptions-item>
        </el-descriptions>

        <!-- 生活方式 -->
        <el-divider content-position="left">生活方式</el-divider>
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="饮食计划">
            <el-tag :type="selectedExecution.diet_completed ? 'success' : 'info'" size="small">
              {{ selectedExecution.diet_completed ? '按计划执行' : '未按计划' }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="运动计划">
            <el-tag :type="selectedExecution.exercise_completed ? 'success' : 'info'" size="small">
              {{ selectedExecution.exercise_completed ? '按计划执行' : '未按计划' }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>
        <p v-if="selectedExecution.diet_notes" class="detail-notes">
          <strong>饮食备注：</strong>{{ selectedExecution.diet_notes }}
        </p>
        <p v-if="selectedExecution.exercise_notes" class="detail-notes">
          <strong>运动备注：</strong>{{ selectedExecution.exercise_notes }}
        </p>

        <!-- 复查情况 -->
        <el-divider content-position="left">复查情况</el-divider>
        <div>
          <span style="font-weight: 500;">复查完成：</span>
          <el-tag :type="selectedExecution.follow_up_completed ? 'success' : 'info'" size="small">
            {{ selectedExecution.follow_up_completed ? '已完成' : '未完成' }}
          </el-tag>
        </div>

        <!-- 患者反馈 -->
        <el-divider content-position="left">患者反馈</el-divider>
        <p class="feedback-text">{{ selectedExecution.patient_feedback || '无' }}</p>
      </div>
    </el-dialog>

    <!-- 添加执行记录对话框 -->
    <el-dialog
      v-model="executionDialogVisible"
      title="记录执行情况"
      width="650px"
    >
      <el-form :model="executionForm" label-width="100px">
        <el-form-item label="执行日期" required>
          <el-date-picker
            v-model="executionForm.execution_date"
            type="date"
            placeholder="选择日期"
            style="width: 100%"
          />
        </el-form-item>

        <!-- 用药情况 -->
        <el-divider content-position="left">用药情况</el-divider>
        <div v-for="(med, index) in selectedPlan?.medications || []" :key="index" class="medication-item">
          <span style="width: 120px; display: inline-block; font-weight: 500;">{{ med.name }}</span>
          <el-radio-group v-model="executionForm.medication_taken[med.name]" size="small">
            <el-radio-button label="yes">是</el-radio-button>
            <el-radio-button label="no">否</el-radio-button>
          </el-radio-group>
        </div>
        <el-form-item label="用药备注" style="margin-top: 10px">
          <el-input
            v-model="executionForm.medication_notes"
            type="textarea"
            :rows="2"
            placeholder="请输入用药特殊情况说明"
          />
        </el-form-item>

        <!-- 血糖记录 -->
        <el-divider content-position="left">血糖记录 (mmol/L)</el-divider>
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="空腹血糖">
              <el-input v-model="executionForm.blood_sugar_fasting" placeholder="如：5.6" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="餐后血糖">
              <el-input v-model="executionForm.blood_sugar_postprandial" placeholder="如：7.8" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="糖化血红蛋白">
              <el-input v-model="executionForm.blood_sugar_hba1c" placeholder="如：6.5" />
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 血压记录 -->
        <el-divider content-position="left">血压记录 (mmHg)</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="收缩压">
              <el-input v-model="executionForm.blood_pressure_systolic" placeholder="如：120" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="舒张压">
              <el-input v-model="executionForm.blood_pressure_diastolic" placeholder="如：80" />
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 饮食与运动 -->
        <el-divider content-position="left">生活方式</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="饮食计划">
              <el-switch v-model="executionForm.diet_completed" active-text="按计划执行" />
              <el-input
                v-model="executionForm.diet_notes"
                type="textarea"
                :rows="2"
                placeholder="饮食特殊情况说明"
                style="margin-top: 5px"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="运动计划">
              <el-switch v-model="executionForm.exercise_completed" active-text="按计划执行" />
              <el-input
                v-model="executionForm.exercise_notes"
                type="textarea"
                :rows="2"
                placeholder="运动特殊情况说明"
                style="margin-top: 5px"
              />
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 复查完成 -->
        <el-divider content-position="left">复查情况</el-divider>
        <el-form-item label="复查完成">
          <el-switch v-model="executionForm.follow_up_completed" active-text="已完成复查" />
        </el-form-item>

        <!-- 患者反馈 -->
        <el-divider content-position="left">反馈信息</el-divider>
        <el-form-item label="我的反馈">
          <el-input
            v-model="executionForm.patient_feedback"
            type="textarea"
            :rows="3"
            placeholder="请记录用药后的感受、症状变化、身体状况等"
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
import { RefreshRight, Aim, FirstAidKit, Food, Calendar, WarningFilled, List, Operation, InfoFilled } from '@element-plus/icons-vue'
import PlanList from './components/PlanList.vue'

export default {
  name: 'PatientPlan',
  components: {
    RefreshRight,
    Aim,
    FirstAidKit,
    Food,
    Calendar,
    WarningFilled,
    List,
    Operation,
    InfoFilled,
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
      executionDetailDialogVisible: false,
      executionForm: {
        execution_date: new Date(),
        medication_taken: {},
        medication_notes: '',
        follow_up_completed: false,
        patient_feedback: '',
        // 血糖记录
        blood_sugar_fasting: '',
        blood_sugar_postprandial: '',
        blood_sugar_hba1c: '',
        // 血压记录
        blood_pressure_systolic: '',
        blood_pressure_diastolic: '',
        // 饮食与运动
        diet_completed: false,
        diet_notes: '',
        exercise_completed: false,
        exercise_notes: ''
      },
      saving: false,
      executions: [],
      currentMonth: new Date(),
      selectedExecutionDate: null,
      selectedExecution: null
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
    },
    medicationList() {
      return this.selectedPlan?.medications || []
    },
    planStartDate() {
      return this.selectedPlan?.created_at ? new Date(this.selectedPlan.created_at).toISOString().split('T')[0] : null
    },
    missedDays() {
      if (!this.planStartDate) return 0
      
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      const startDate = new Date(this.planStartDate)
      startDate.setHours(0, 0, 0, 0)
      
      const totalDays = Math.floor((today - startDate) / (1000 * 60 * 60 * 24)) + 1
      if (totalDays <= 0) return 0
      
      // 统计已有记录的天数
      const recordedDates = new Set(this.executions.map(e => e.execution_date))
      
      // 计算漏记天数
      let missed = 0
      for (let i = 0; i < totalDays; i++) {
        const currentDate = new Date(startDate)
        currentDate.setDate(startDate.getDate() + i)
        const dateStr = this.formatDate(currentDate)
        if (!recordedDates.has(dateStr)) {
          missed++
        }
      }
      
      return missed
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
    async fetchExecutions(planId) {
      try {
        const res = await api.get(`/api/treatment/plans/${planId}/executions/`)
        this.executions = res.data
      } catch (error) {
        console.error('获取执行记录失败:', error)
        this.executions = []
      }
    },
    isExecutedDate(dateStr) {
      return this.executions.some(exec => exec.execution_date === dateStr)
    },
    isMissedDate(dateStr) {
      // 简化：只标记有执行记录中的遗漏
      if (!this.planStartDate) return false
      
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      const checkDate = new Date(dateStr)
      checkDate.setHours(0, 0, 0, 0)
      
      // 未来日期不标记
      if (checkDate > today) return false
      
      // 已有记录不标记
      if (this.isExecutedDate(dateStr)) return false
      
      // 在方案开始日期之后
      const startDate = new Date(this.planStartDate)
      startDate.setHours(0, 0, 0, 0)
      
      return checkDate >= startDate
    },
    viewExecutionDetail(dateStr) {
      console.log('viewExecutionDetail 被调用, dateStr:', dateStr)
      
      // 检查是否有执行记录
      const execution = this.executions.find(exec => exec.execution_date === dateStr)
      console.log('execution:', execution)
      
      if (execution) {
        // 有记录，显示详情
        this.selectedExecution = execution
        this.selectedExecutionDate = dateStr
        this.executionDetailDialogVisible = true
        this.executionDialogVisible = false
      } else {
        // 没有记录，直接打开补录对话框
        this.selectedExecution = null
        this.selectedExecutionDate = null
        this.executionDetailDialogVisible = false
        this.executionForm.execution_date = dateStr
        this.executionDialogVisible = true
      }
    },
    async handleViewDetail(plan) {
      this.selectedPlan = plan
      this.selectedExecutionDate = null
      this.selectedExecution = null
      this.currentMonth = new Date()
      await this.fetchExecutions(plan.id)
      this.detailDialogVisible = true
    },
    handleAddExecution(plan) {
      this.selectedPlan = plan
      this.executionForm = {
        execution_date: this.formatDate(new Date()),
        medication_taken: {},
        medication_notes: '',
        follow_up_completed: false,
        patient_feedback: '',
        // 血糖记录
        blood_sugar_fasting: '',
        blood_sugar_postprandial: '',
        blood_sugar_hba1c: '',
        // 血压记录
        blood_pressure_systolic: '',
        blood_pressure_diastolic: '',
        // 饮食与运动
        diet_completed: false,
        diet_notes: '',
        exercise_completed: false,
        exercise_notes: ''
      }
      // 初始化用药记录
      const meds = {}
      plan.medications?.forEach(med => {
        if (med?.name) {
          meds[med.name] = 'yes'  // 默认按时服药
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
        await this.fetchExecutions(this.selectedPlan.id)
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
    formatDate(val) {
      if (!val) return ''
      if (typeof val === 'string') {
        return val.split('T')[0]
      }
      const d = new Date(val)
      if (isNaN(d.getTime())) return ''
      const y = d.getFullYear()
      const m = `${d.getMonth() + 1}`.padStart(2, '0')
      const day = `${d.getDate()}`.padStart(2, '0')
      return `${y}-${m}-${day}`
    },
    formatDateTime(val) {
      if (!val) return ''
      return val.replace('T', ' ').split('.')[0]
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
.patient-plan-page {
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

.plan-detail {
  max-height: 70vh;
  overflow-y: auto;
}

.basic-info {
  margin-bottom: 15px;
}

.detail-section {
  margin-top: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.section-title h4 {
  margin: 0;
  color: #303133;
  font-size: 15px;
}

.section-title .el-icon {
  color: #409eff;
  font-size: 18px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.exec-count {
  font-size: 12px;
  color: #909399;
}

.lifestyle-content {
  display: grid;
  gap: 12px;
}

.lifestyle-item {
  background: #fff;
  padding: 12px;
  border-radius: 4px;
  border-left: 3px solid #67c23a;
}

.lifestyle-item h5 {
  margin: 0 0 8px 0;
  color: #67c23a;
  font-size: 14px;
}

.content-text {
  white-space: pre-wrap;
  font-size: 14px;
  line-height: 1.8;
  color: #606266;
}

.warning-section {
  background: #fdf6ec;
  border-left: 3px solid #e6a23c;
}

.warning-content {
  white-space: pre-wrap;
  font-size: 14px;
  line-height: 1.8;
  color: #e6a23c;
}

.medication-item {
  margin-bottom: 10px;
  padding: 8px;
  background: #f5f7fa;
  border-radius: 4px;
}

.small-calendar {
  width: 100%;
  max-width: 500px;
}

.small-calendar :deep(.el-calendar__body) {
  font-size: 12px;
}

.small-calendar :deep(.el-calendar-day) {
  height: 40px;
  padding: 2px;
}

.small-calendar :deep(.el-calendar-table .el-calendar-day) {
  height: 40px;
}

.feedback-text {
  color: #606266;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
}

.detail-notes {
  color: #606266;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  margin-top: 8px;
  padding: 8px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.unit {
  color: #909399;
  font-size: 12px;
  margin-left: 4px;
}

.calendar-date {
  height: 100%;
  padding: 2px;
  cursor: pointer;
  position: relative;
  border-radius: 4px;
  transition: background-color 0.2s;
  font-size: 12px;
}

.calendar-date:hover {
  background-color: #f5f5f5;
}

.calendar-date.is-executed {
  background-color: #e8f5e9;
}

.calendar-date.is-selected {
  background-color: #c8e6c9;
  border: 2px solid #4caf50;
}

.executed-dot {
  position: absolute;
  bottom: 1px;
  left: 50%;
  transform: translateX(-50%);
  width: 5px;
  height: 5px;
  background-color: #4caf50;
  border-radius: 50%;
}

.no-executions {
  text-align: center;
  padding: 20px;
}

/* 补录提示 */
.missed-tip {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: #fff7e6;
  border: 1px solid #ffcc80;
  border-radius: 6px;
  color: #e65100;
  font-size: 13px;
  margin-bottom: 12px;
}

.missed-tip strong {
  color: #ff9800;
  font-size: 15px;
}

/* 漏记统计 */
.exec-stats {
  display: flex;
  gap: 12px;
  align-items: center;
}

.missed-count {
  font-size: 12px;
  color: #ff9800;
  background: #fff7e6;
  padding: 2px 8px;
  border-radius: 10px;
}

/* 未记录日期样式 */
.calendar-date.is-missed {
  background-color: #fff7e6 !important;
  opacity: 0.8;
}

.calendar-date.is-missed:hover {
  background-color: #ffe0b2 !important;
}

.missed-dot {
  position: absolute;
  bottom: 1px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 10px;
  color: #ff9800;
}

/* 图例说明 */
.calendar-legend {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-top: 8px;
  padding: 6px;
  background: #fafafa;
  border-radius: 4px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: #606266;
}

.legend-item .dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.legend-item .dot.executed {
  background: #4caf50;
}

.legend-item .dot.missed {
  background: #ff9800;
}
</style>
