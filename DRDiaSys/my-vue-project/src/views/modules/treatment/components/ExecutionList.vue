<template>
  <div class="execution-list doctor-execution">
    <!-- 统计汇总区域 -->
    <div class="stats-summary" v-if="executions.length > 0 || planStartDate">
      <div class="stat-item">
        <div class="stat-value">{{ monthAdherence }}%</div>
        <div class="stat-label">本月依从性</div>
      </div>
      <div class="stat-item">
        <div class="stat-value abnormal">{{ abnormalDays }}</div>
        <div class="stat-label">异常天数</div>
      </div>
      <div class="stat-item">
        <div class="stat-value warning">{{ missedCount }}</div>
        <div class="stat-label">漏服/漏做</div>
      </div>
      <div class="stat-item">
        <div class="stat-value missed">{{ missedDays }}</div>
        <div class="stat-label">漏记天数</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">{{ totalExecutions }}</div>
        <div class="stat-label">总记录数</div>
      </div>
    </div>

    <!-- 补录提示 -->
    <div v-if="planStartDate && missedDays > 0" class="missed-tip">
      <el-icon><InfoFilled /></el-icon>
      <span>该方案自 {{ planStartDate }} 起，有 <strong>{{ missedDays }}</strong> 天未记录执行情况，可补录历史记录</span>
    </div>

    <!-- 日历视图 -->
    <div class="calendar-container" v-if="executions.length > 0">
      <el-calendar v-model="currentMonth">
        <template #date-cell="{ data }">
          <div
            class="calendar-date"
            :class="getDateClass(data.day)"
            @click="selectDate(data.day)"
          >
            <span class="date-number">{{ data.day.split('-')[2] }}</span>
            <span v-if="getExecutionForDate(data.day)" class="exec-status-dot"></span>
            <span v-if="hasAbnormal(data.day)" class="abnormal-icon">!</span>
            <span v-if="isMissedDate(data.day)" class="missed-dot">✕</span>
          </div>
        </template>
      </el-calendar>
      <!-- 图例说明 -->
      <div class="calendar-legend">
        <div class="legend-item"><span class="dot completed"></span> 全部完成</div>
        <div class="legend-item"><span class="dot abnormal"></span> 有遗漏</div>
        <div class="legend-item"><span class="dot missed"></span> 未记录</div>
      </div>
    </div>

    <!-- 无记录状态 -->
    <div v-else class="no-records">
      <el-empty description="暂无执行记录" :image-size="80" />
    </div>

    <!-- 未记录日期提示 -->
    <div v-if="!selectedExecution && selectedExecutionDate && isMissedDate(selectedExecutionDate)" class="detail-panel missed-info">
      <div class="panel-header">
        <h4>{{ selectedExecutionDate }}</h4>
        <el-tag type="warning" size="small">未记录</el-tag>
      </div>
      <div class="missed-tip">
        <el-icon><Warning /></el-icon>
        <span>该日期暂无执行记录，患者尚未填写</span>
      </div>
    </div>

    <!-- 详情面板 -->
    <div v-if="selectedExecution" class="detail-panel">
      <div class="panel-header">
        <h4>{{ selectedExecutionDate }} 执行详情</h4>
        <el-tag :type="getExecutionStatusType(selectedExecution)" size="small">
          {{ getExecutionStatusLabel(selectedExecution) }}
        </el-tag>
      </div>

      <!-- 执行汇总 -->
      <div class="summary-section">
        <div class="summary-item">
          <span class="label">提交时间：</span>
          <span class="value">{{ formatDateTime(selectedExecution.created_at) }}</span>
        </div>
        <div class="summary-item">
          <span class="label">记录人：</span>
          <span class="value">{{ selectedExecution.created_by_name || '患者' }}</span>
        </div>
      </div>

      <!-- 用药情况 -->
      <div class="task-section">
        <div class="section-title">
          <el-icon><FirstAidKit /></el-icon>
          <span>用药情况</span>
        </div>
        <div class="task-list">
          <div v-for="med in medicationList" :key="med.name" class="task-item" :class="{ missed: selectedExecution.medication_taken?.[med.name] === 'no' }">
            <el-icon v-if="selectedExecution.medication_taken?.[med.name] === 'yes'" class="task-icon success"><CircleCheck /></el-icon>
            <el-icon v-else-if="selectedExecution.medication_taken?.[med.name] === 'no'" class="task-icon danger"><CircleClose /></el-icon>
            <el-icon v-else class="task-icon info"><QuestionFilled /></el-icon>
            <span class="task-name">{{ med.name }}</span>
            <span class="task-status">
              {{ selectedExecution.medication_taken?.[med.name] === 'yes' ? '已服用' : selectedExecution.medication_taken?.[med.name] === 'no' ? '未服用' : '未记录' }}
            </span>
          </div>
        </div>
        <p v-if="selectedExecution.medication_notes" class="task-notes">
          <el-icon><Edit /></el-icon> {{ selectedExecution.medication_notes }}
        </p>
      </div>

      <!-- 生理指标 -->
      <div class="task-section">
        <div class="section-title">
          <el-icon><Aim /></el-icon>
          <span>生理指标</span>
        </div>
        <div class="vital-grid">
          <div class="vital-item" :class="{ abnormal: isVitalAbnormal('fasting', selectedExecution.blood_sugar_fasting) }">
            <span class="vital-label">空腹血糖</span>
            <span class="vital-value">{{ selectedExecution.blood_sugar_fasting || '-' }}</span>
            <span class="vital-unit">mmol/L</span>
          </div>
          <div class="vital-item" :class="{ abnormal: isVitalAbnormal('postprandial', selectedExecution.blood_sugar_postprandial) }">
            <span class="vital-label">餐后血糖</span>
            <span class="vital-value">{{ selectedExecution.blood_sugar_postprandial || '-' }}</span>
            <span class="vital-unit">mmol/L</span>
          </div>
          <div class="vital-item">
            <span class="vital-label">HbA1c</span>
            <span class="vital-value">{{ selectedExecution.blood_sugar_hba1c || '-' }}</span>
            <span class="vital-unit">%</span>
          </div>
          <div class="vital-item" :class="{ abnormal: isVitalAbnormal('systolic', selectedExecution.blood_pressure_systolic) }">
            <span class="vital-label">血压</span>
            <span class="vital-value">
              {{ selectedExecution.blood_pressure_systolic || '-' }}/{{ selectedExecution.blood_pressure_diastolic || '-' }}
            </span>
            <span class="vital-unit">mmHg</span>
          </div>
        </div>
      </div>

      <!-- 生活方式 -->
      <div class="task-section">
        <div class="section-title">
          <el-icon><Food /></el-icon>
          <span>生活方式</span>
        </div>
        <div class="lifestyle-items">
          <div class="lifestyle-item" :class="{ missed: !selectedExecution.diet_completed }">
            <el-icon v-if="selectedExecution.diet_completed" class="task-icon success"><CircleCheck /></el-icon>
            <el-icon v-else class="task-icon warning"><Warning /></el-icon>
            <span>饮食计划</span>
            <el-tag :type="selectedExecution.diet_completed ? 'success' : 'info'" size="small">
              {{ selectedExecution.diet_completed ? '按计划执行' : '未按计划' }}
            </el-tag>
          </div>
          <div class="lifestyle-item" :class="{ missed: !selectedExecution.exercise_completed }">
            <el-icon v-if="selectedExecution.exercise_completed" class="task-icon success"><CircleCheck /></el-icon>
            <el-icon v-else class="task-icon warning"><Warning /></el-icon>
            <span>运动计划</span>
            <el-tag :type="selectedExecution.exercise_completed ? 'success' : 'info'" size="small">
              {{ selectedExecution.exercise_completed ? '按计划执行' : '未按计划' }}
            </el-tag>
          </div>
        </div>
        <p v-if="selectedExecution.diet_notes || selectedExecution.exercise_notes" class="task-notes">
          <el-icon><Edit /></el-icon>
          饮食备注: {{ selectedExecution.diet_notes || '无' }} | 运动备注: {{ selectedExecution.exercise_notes || '无' }}
        </p>
      </div>

      <!-- 复查情况 -->
      <div class="task-section">
        <div class="section-title">
          <el-icon><Calendar /></el-icon>
          <span>复查情况</span>
        </div>
        <div class="follow-up-status">
          <el-tag :type="selectedExecution.follow_up_completed ? 'success' : 'info'" size="large">
            {{ selectedExecution.follow_up_completed ? '已完成复查' : '未复查' }}
          </el-tag>
        </div>
      </div>

      <!-- 患者反馈 -->
      <div class="task-section" v-if="selectedExecution.patient_feedback">
        <div class="section-title">
          <el-icon><ChatDotRound /></el-icon>
          <span>患者反馈</span>
        </div>
        <div class="feedback-box">
          {{ selectedExecution.patient_feedback }}
        </div>
      </div>

      <!-- 异常警示 -->
      <div v-if="hasAbnormal(selectedExecutionDate)" class="abnormal-warning">
        <el-icon><WarningFilled /></el-icon>
        <span>存在异常情况，请关注</span>
      </div>
    </div>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage } from 'element-plus'
import { CircleCheck, CircleClose, Warning, QuestionFilled, Edit, FirstAidKit, Aim, Food, Calendar, ChatDotRound, WarningFilled, InfoFilled } from '@element-plus/icons-vue'

export default {
  name: 'ExecutionList',
  components: {
    CircleCheck,
    CircleClose,
    Warning,
    QuestionFilled,
    Edit,
    FirstAidKit,
    Aim,
    Food,
    Calendar,
    ChatDotRound,
    WarningFilled,
    InfoFilled
  },
  props: {
    planId: {
      type: Number,
      required: true
    },
    medications: {
      type: Array,
      default: () => []
    },
    planStartDate: {
      type: String,
      default: null
    }
  },
  data() {
    return {
      loading: false,
      executions: [],
      currentMonth: new Date(),
      selectedExecutionDate: null,
      selectedExecution: null
    }
  },
  computed: {
    medicationList() {
      return this.medications || []
    },
    totalExecutions() {
      return this.executions.length
    },
    abnormalDays() {
      return this.executions.filter(e => this.isExecutionAbnormal(e)).length
    },
    missedCount() {
      let count = 0
      this.executions.forEach(e => {
        if (e.medication_taken) {
          Object.values(e.medication_taken).forEach(v => {
            if (v === 'no') count++
          })
        }
        if (!e.diet_completed) count++
        if (!e.exercise_completed) count++
      })
      return count
    },
    monthAdherence() {
      if (!this.planStartDate) {
        // 如果没有方案开始日期，使用原有逻辑
        if (this.executions.length === 0) return 0
        const normalCount = this.executions.filter(e => !this.isExecutionAbnormal(e)).length
        return Math.round((normalCount / this.executions.length) * 100)
      }
      
      // 计算从方案开始到今天的天数
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      const startDate = new Date(this.planStartDate)
      startDate.setHours(0, 0, 0, 0)
      
      const totalDays = Math.floor((today - startDate) / (1000 * 60 * 60 * 24)) + 1
      const totalRecorded = this.executions.length
      
      if (totalDays <= 0) return 100
      return Math.round((totalRecorded / totalDays) * 100)
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
      
      // 计算漏记天数（从开始日期到今天，减去已有记录的天数）
      let missed = 0
      for (let i = 0; i < totalDays; i++) {
        const currentDate = new Date(startDate)
        currentDate.setDate(startDate.getDate() + i)
        const dateStr = this.formatDateString(currentDate)
        if (!recordedDates.has(dateStr)) {
          missed++
        }
      }
      
      return missed
    }
  },
  created() {
    this.fetchExecutions()
  },
  watch: {
    planId() {
      this.fetchExecutions()
    }
  },
  methods: {
    async fetchExecutions() {
      this.loading = true
      try {
        const res = await api.get(`/api/treatment/plans/${this.planId}/executions/`)
        this.executions = res.data
        // 默认选择最近一次执行记录
        if (this.executions.length > 0) {
          const latest = this.executions[this.executions.length - 1]
          this.selectDate(latest.execution_date)
        }
      } catch (error) {
        console.error('获取执行记录失败:', error)
        ElMessage.error('获取执行记录失败')
      } finally {
        this.loading = false
      }
    },
    getExecutionForDate(dateStr) {
      return this.executions.find(e => e.execution_date === dateStr)
    },
    isMissedDate(dateStr) {
      if (!this.planStartDate) return false
      
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      const checkDate = new Date(dateStr)
      checkDate.setHours(0, 0, 0, 0)
      
      // 如果是未来日期，不标记为漏记
      if (checkDate > today) return false
      
      // 如果已有记录，不标记为漏记
      if (this.getExecutionForDate(dateStr)) return false
      
      // 如果在方案开始日期之后，标记为漏记
      const startDate = new Date(this.planStartDate)
      startDate.setHours(0, 0, 0, 0)
      
      return checkDate >= startDate
    },
    formatDateString(date) {
      const year = date.getFullYear()
      const month = String(date.getMonth() + 1).padStart(2, '0')
      const day = String(date.getDate()).padStart(2, '0')
      return `${year}-${month}-${day}`
    },
    selectDate(dateStr) {
      console.log('selectDate 被调用, dateStr:', dateStr)
      const execution = this.getExecutionForDate(dateStr)
      if (execution) {
        // 有执行记录，显示详情
        this.selectedExecutionDate = dateStr
        this.selectedExecution = execution
      } else if (this.isMissedDate(dateStr)) {
        // 没有执行记录但需要标记
        this.selectedExecutionDate = dateStr
        this.selectedExecution = null
      } else {
        // 其他日期
        this.selectedExecutionDate = null
        this.selectedExecution = null
      }
    },
    hasAbnormal(dateStr) {
      const execution = this.getExecutionForDate(dateStr)
      return execution && this.isExecutionAbnormal(execution)
    },
    isExecutionAbnormal(execution) {
      // 有漏服药物或未完成生活方式计划视为异常
      if (execution.medication_taken) {
        const values = Object.values(execution.medication_taken)
        if (values.some(v => v === 'no')) return true
      }
      if (!execution.diet_completed || !execution.exercise_completed) return true
      return false
    },
    getDateClass(dateStr) {
      // 如果有执行记录
      const execution = this.getExecutionForDate(dateStr)
      if (execution) {
        if (this.selectedExecutionDate === dateStr) {
          return 'is-selected'
        }
        if (this.isExecutionAbnormal(execution)) {
          return 'is-abnormal'
        }
        return 'is-completed'
      }
      
      // 如果是未记录的日期
      if (this.isMissedDate(dateStr)) {
        if (this.selectedExecutionDate === dateStr) {
          return 'is-selected is-missed'
        }
        return 'is-missed'
      }
      
      return ''
    },
    getExecutionStatusType(execution) {
      if (this.isExecutionAbnormal(execution)) return 'danger'
      return 'success'
    },
    getExecutionStatusLabel(execution) {
      if (this.isExecutionAbnormal(execution)) return '有遗漏'
      return '全部完成'
    },
    isVitalAbnormal(type, value) {
      if (!value) return false
      const num = parseFloat(value)
      if (isNaN(num)) return false
      
      // 血糖异常阈值
      if (type === 'fasting' && (num < 3.9 || num > 7.8)) return true
      if (type === 'postprandial' && (num < 4.4 || num > 10.0)) return true
      
      // 血压异常阈值
      if (type === 'systolic' && num > 140) return true
      
      return false
    },
    formatDateTime(time) {
      if (!time) return '-'
      return new Date(time).toLocaleString('zh-CN')
    }
  }
}
</script>

<style scoped>
.doctor-execution {
  padding: 16px;
}

.stats-summary {
  display: flex;
  justify-content: space-around;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.stat-item {
  text-align: center;
  color: white;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
}

.stat-value.abnormal,
.stat-value.warning {
  background: rgba(255, 255, 255, 0.2);
  padding: 0 15px;
  border-radius: 20px;
}

.stat-label {
  font-size: 12px;
  opacity: 0.9;
  margin-top: 4px;
}

.calendar-container {
  margin-bottom: 20px;
}

.calendar-date {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  cursor: pointer;
  border-radius: 8px;
  transition: all 0.3s;
}

.date-number {
  font-size: 14px;
  font-weight: 500;
}

.exec-status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-top: 4px;
}

.is-completed .exec-status-dot {
  background: #67c23a;
}

.is-abnormal .exec-status-dot {
  background: #f56c6c;
}

.is-selected {
  background-color: #ecf5ff;
}

.is-completed {
  background-color: #f0f9eb;
}

.is-abnormal {
  background-color: #fef0f0;
}

.abnormal-icon {
  position: absolute;
  top: 2px;
  right: 4px;
  color: #f56c6c;
  font-weight: bold;
  font-size: 12px;
}

.no-records {
  padding: 40px;
}

.missed-info {
  background: #fff7e6;
  border-color: #ffd591;
}

.missed-info .panel-header {
  border-bottom-color: #ffe7ba;
}

.missed-info .missed-tip {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px 0;
  color: #d46b08;
  font-size: 14px;
}

.detail-panel {
  background: #fafafa;
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #ebeef5;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #ebeef5;
}

.panel-header h4 {
  margin: 0;
  color: #303133;
}

.summary-section {
  display: flex;
  gap: 30px;
  margin-bottom: 16px;
  padding: 12px;
  background: white;
  border-radius: 8px;
}

.summary-item {
  display: flex;
  align-items: center;
}

.summary-item .label {
  color: #909399;
  margin-right: 8px;
}

.summary-item .value {
  color: #303133;
  font-weight: 500;
}

.task-section {
  background: white;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 12px;
}

.section-title .el-icon {
  color: #409eff;
}

.task-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.task-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #f5f7fa;
  border-radius: 6px;
  flex: 1;
  min-width: 200px;
}

.task-item.missed {
  background: #fef0f0;
}

.task-icon {
  font-size: 16px;
}

.task-icon.success {
  color: #67c23a;
}

.task-icon.danger {
  color: #f56c6c;
}

.task-icon.warning {
  color: #e6a23c;
}

.task-icon.info {
  color: #909399;
}

.task-name {
  flex: 1;
  font-size: 13px;
}

.task-status {
  font-size: 12px;
  color: #606266;
}

.task-notes {
  margin-top: 8px;
  padding: 8px 12px;
  background: #fdf6ec;
  border-radius: 4px;
  font-size: 13px;
  color: #e6a23c;
  display: flex;
  align-items: center;
  gap: 6px;
}

.vital-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.vital-item {
  padding: 12px;
  background: #f5f7fa;
  border-radius: 8px;
  text-align: center;
}

.vital-item.abnormal {
  background: #fef0f0;
  border: 1px solid #fbc4c4;
}

.vital-label {
  display: block;
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}

.vital-value {
  display: block;
  font-size: 18px;
  font-weight: bold;
  color: #303133;
}

.vital-item.abnormal .vital-value {
  color: #f56c6c;
}

.vital-unit {
  font-size: 12px;
  color: #909399;
}

.lifestyle-items {
  display: flex;
  gap: 16px;
}

.lifestyle-item {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: #f5f7fa;
  border-radius: 6px;
}

.lifestyle-item.missed {
  background: #fef0f0;
}

.follow-up-status {
  text-align: center;
  padding: 8px;
}

.feedback-box {
  padding: 12px;
  background: #ecf5ff;
  border-radius: 8px;
  border-left: 4px solid #409eff;
  font-size: 14px;
  color: #303133;
  line-height: 1.6;
}

.abnormal-warning {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: #fef0f0;
  border: 1px solid #fbc4c4;
  border-radius: 8px;
  color: #f56c6c;
  font-weight: 600;
  margin-top: 16px;
}

:deep(.el-calendar-table .el-calendar-day) {
  height: 60px;
}

:deep(.el-calendar-table current-month-day) {
  height: 60px;
}

/* 漏记天数样式 */
.stat-value.missed {
  background: rgba(255, 255, 255, 0.2);
  padding: 0 15px;
  border-radius: 20px;
  color: #ff9800;
}

/* 未记录日期样式 */
.is-missed {
  background-color: #fff7e6 !important;
  opacity: 0.7;
}

.is-missed .date-number {
  color: #909399;
}

.missed-dot {
  position: absolute;
  bottom: 4px;
  color: #ff9800;
  font-size: 12px;
}

/* 补录提示 */
.missed-tip {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: #fff7e6;
  border: 1px solid #ffcc80;
  border-radius: 8px;
  color: #e65100;
  font-size: 14px;
  margin-bottom: 16px;
}

.missed-tip strong {
  color: #ff9800;
  font-size: 16px;
}

/* 图例说明 */
.calendar-legend {
  display: flex;
  justify-content: center;
  gap: 24px;
  margin-top: 12px;
  padding: 8px;
  background: #fafafa;
  border-radius: 8px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #606266;
}

.legend-item .dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.legend-item .dot.completed {
  background: #67c23a;
}

.legend-item .dot.abnormal {
  background: #f56c6c;
}

.legend-item .dot.missed {
  background: #ff9800;
}
</style>
