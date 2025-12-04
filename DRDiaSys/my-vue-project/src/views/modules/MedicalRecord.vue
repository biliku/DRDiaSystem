<template>
  <div class="medical-record-page" v-if="isDoctor">
    <el-row :gutter="20">
      <el-col :span="7">
        <el-card shadow="never" class="sidebar-card">
          <div class="sidebar-header">
            <div>
              <h3>病例列表</h3>
              <p class="subtitle">仅展示已复核患者的病例</p>
            </div>
            <el-button type="primary" size="small" @click="openCreateDialog">
              <el-icon><Plus /></el-icon>
              新建病例
            </el-button>
          </div>
          <div class="filter-bar">
            <el-input
              v-model="filters.keyword"
              placeholder="搜索患者/标题"
              clearable
              @input="applyFilter"
            >
              <template #prefix>
                <el-icon><Search /></el-icon>
              </template>
            </el-input>
            <el-select v-model="filters.status" placeholder="状态" @change="applyFilter">
              <el-option label="全部" value="all" />
              <el-option label="进行中" value="active" />
              <el-option label="已结束" value="closed" />
              <el-option label="已归档" value="archived" />
            </el-select>
          </div>
          <el-table
            :data="filteredCases"
            height="calc(100vh - 260px)"
            v-loading="caseLoading"
            @row-click="handleCaseSelect"
            :row-class-name="rowClassName"
          >
            <el-table-column prop="title" label="病例" min-width="120" />
            <el-table-column prop="patient_name" label="患者" width="100" />
            <el-table-column label="状态" width="90">
              <template #default="scope">
                <el-tag :type="statusType(scope.row.status)" size="small">
                  {{ statusLabel(scope.row.status) }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>

      <el-col :span="17">
        <el-card shadow="never" class="detail-card" v-if="selectedCase">
          <div class="detail-header">
            <div>
              <h2>{{ selectedCase.title }}</h2>
              <p class="patient-info">
                患者：{{ selectedCase.patient_name }} ｜ 首份报告：{{ selectedCase.primary_report_info?.report_number }}
              </p>
            </div>
            <div class="detail-actions">
              <el-select v-model="selectedCase.status" size="small" @change="updateCaseStatus">
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
              <el-button type="primary" size="small" @click="openEventDialog">
                <el-icon><Edit /></el-icon>
                新增记录
              </el-button>
            </div>
          </div>

          <el-descriptions :column="2" border class="case-summary">
            <el-descriptions-item label="病例摘要">
              {{ selectedCase.summary || '暂无描述' }}
            </el-descriptions-item>
            <el-descriptions-item label="首份AI诊断">
              <div class="report-info">
                <p>报告编号：{{ selectedCase.primary_report_info?.report_number }}</p>
                <p>AI结论：{{ selectedCase.primary_report_info?.ai_summary || '—' }}</p>
                <p>医生诊断：{{ selectedCase.primary_report_info?.doctor_conclusion || '—' }}</p>
                <el-button
                  v-if="selectedCase.primary_report_info?.pdf_path"
                  type="primary"
                  link
                  @click="previewPrimaryReport"
                >
                  查看报告
                </el-button>
              </div>
            </el-descriptions-item>
          </el-descriptions>

          <h3 class="section-title">随访与事件记录</h3>
          <el-empty v-if="!selectedCase.events?.length" description="暂无记录" />
          <el-timeline v-else class="timeline">
            <el-timeline-item
              v-for="event in selectedCase.events"
              :key="event.id"
              :timestamp="formatDateTime(event.created_at)"
              :type="timelineType(event.event_type)"
            >
              <div class="event-item">
                <strong>{{ eventTypeLabel(event.event_type) }}</strong>
                <p class="event-desc">{{ event.description }}</p>
                <p v-if="event.related_report" class="event-report">
                  关联报告：{{ event.related_report.report_number }}
                  <el-button
                    v-if="event.related_report.pdf_path"
                    type="primary"
                    link
                    @click="previewReport(event.related_report)"
                  >
                    查看PDF
                  </el-button>
                </p>
                <p v-if="event.next_followup_date" class="event-follow">
                  下次随访：{{ event.next_followup_date }}
                </p>
              </div>
            </el-timeline-item>
          </el-timeline>
        </el-card>
        <el-card v-else class="detail-card" shadow="never">
          <el-empty description="请选择或创建一个病例" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 创建病例 -->
    <el-dialog v-model="createDialogVisible" title="新建病例" width="520px" :close-on-click-modal="false">
      <el-form :model="createForm" label-width="90px">
        <el-form-item label="病例标题">
          <el-input v-model="createForm.title" placeholder="例如：张三-DR随访" />
        </el-form-item>
        <el-form-item label="首份报告">
          <el-select
            v-model="createForm.primary_report_id"
            filterable
            placeholder="请选择已确认且未建档的报告"
          >
            <el-option
              v-for="report in availableReports"
              :key="report.id"
              :label="`${report.report_number} ｜ ${report.patient_name} ｜ ${formatDateTime(report.created_at)}`"
              :value="report.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="病例摘要">
          <el-input v-model="createForm.summary" type="textarea" :rows="3" />
        </el-form-item>
        <el-form-item label="病例状态">
          <el-select v-model="createForm.status">
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="初始记录">
          <el-input
            v-model="createForm.initial_event_desc"
            type="textarea"
            :rows="2"
            placeholder="可描述建立病例的原因或初步计划"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="createSubmitting" @click="submitCreateCase">创建</el-button>
      </template>
    </el-dialog>

    <!-- 新增事件 -->
    <el-dialog
      v-model="eventDialogVisible"
      title="新增病例记录"
      width="520px"
      :close-on-click-modal="false"
    >
      <el-form :model="eventForm" label-width="90px">
        <el-form-item label="事件类型">
          <el-select v-box v-model="eventForm.event_type">
            <el-option v-for="item in eventTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="eventForm.description" type="textarea" :rows="3" placeholder="填写本次记录内容" />
        </el-form-item>
        <el-form-item label="关联报告">
          <el-select
            v-model="eventForm.related_report_id"
            clearable
            filterable
            placeholder="可选择其他已确认的报告"
          >
            <el-option
              v-for="report in selectableReports"
              :key="report.id"
              :label="`${report.report_number} ｜ ${formatDateTime(report.created_at)}`"
              :value="report.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="下次随访">
          <el-date-picker
            v-model="eventForm.next_followup_date"
            type="date"
            placeholder="选择日期"
            value-format="YYYY-MM-DD"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="eventDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="eventSubmitting" @click="submitEvent">保存</el-button>
      </template>
    </el-dialog>
  </div>

  <el-result
    v-else
    icon="warning"
    title="仅限医生访问"
    sub-title="当前账号无权访问病例管理模块"
  />
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Search, Plus, Edit } from '@element-plus/icons-vue'
import api from '../../api'

const userRole = ref(localStorage.getItem('userRole') || 'doctor')
const isDoctor = computed(() => userRole.value === 'doctor')

const caseLoading = ref(false)
const cases = ref([])
const selectedCase = ref(null)
const filters = ref({ keyword: '', status: 'all' })

const createDialogVisible = ref(false)
const createSubmitting = ref(false)
const createForm = ref({
  primary_report_id: '',
  title: '',
  summary: '',
  status: 'active',
  initial_event_desc: ''
})
const availableReports = ref([])

const eventDialogVisible = ref(false)
const eventSubmitting = ref(false)
const eventForm = ref({
  event_type: 'doctor_note',
  description: '',
  related_report_id: null,
  next_followup_date: ''
})

const eventTypeOptions = [
  { label: 'AI诊断报告', value: 'ai_report' },
  { label: '医生记录', value: 'doctor_note' },
  { label: '随访计划', value: 'follow_up' },
  { label: '治疗方案', value: 'treatment' }
]

const statusOptions = [
  { label: '进行中', value: 'active' },
  { label: '已结束', value: 'closed' },
  { label: '已归档', value: 'archived' }
]

const filteredCases = computed(() => {
  if (!cases.value) return []
  const keyword = filters.value.keyword.trim().toLowerCase()
  return cases.value.filter(item => {
    const matchKeyword =
      !keyword ||
      item.title.toLowerCase().includes(keyword) ||
      (item.patient_name || '').toLowerCase().includes(keyword)
    const matchStatus = filters.value.status === 'all' || item.status === filters.value.status
    return matchKeyword && matchStatus
  })
})

const selectableReports = computed(() => {
  if (!selectedCase.value) return []
  const seen = new Set()
  const result = []
  ;(selectedCase.value.events || []).forEach(e => {
    const r = e.related_report
    if (r && !seen.has(r.id)) {
      seen.add(r.id)
      result.push(r)
    }
  })
  return result
})

const fetchCases = async () => {
  if (!isDoctor.value) return
  try {
    caseLoading.value = true
    const params = {}
    if (filters.value.status !== 'all') {
      params.status = filters.value.status
    }
    const { data } = await api.get('/api/diagnosis/cases/', { params })
    cases.value = data || []
    if (cases.value.length) {
      await fetchCaseDetail(cases.value[0].id)
    } else {
      selectedCase.value = null
    }
  } catch (error) {
    console.error('获取病例失败', error)
    ElMessage.error(error.response?.data?.message || '获取病例失败')
  } finally {
    caseLoading.value = false
  }
}

const fetchCaseDetail = async (caseId) => {
  try {
    const { data } = await api.get(`/api/diagnosis/cases/${caseId}/`)
    selectedCase.value = data
  } catch (error) {
    console.error('获取病例详情失败', error)
    ElMessage.error(error.response?.data?.message || '获取病例详情失败')
  }
}

const handleCaseSelect = (row) => {
  fetchCaseDetail(row.id)
}

const rowClassName = ({ row }) => {
  return selectedCase.value && row.id === selectedCase.value.id ? 'is-selected' : ''
}

const applyFilter = () => {
  fetchCases()
}

const openCreateDialog = async () => {
  createDialogVisible.value = true
  createForm.value = {
    primary_report_id: '',
    title: '',
    summary: '',
    status: 'active',
    initial_event_desc: 'AI诊断报告已确认，建立病例'
  }
  await fetchAvailableReports()
}

const fetchAvailableReports = async () => {
  try {
    const { data } = await api.get('/api/diagnosis/reports/', {
      params: { status: 'finalized', unassigned: 'true' }
    })
    availableReports.value = data || []
  } catch (error) {
    console.error('获取可用报告失败', error)
    ElMessage.error(error.response?.data?.message || '获取可用报告失败')
  }
}

const submitCreateCase = async () => {
  if (!createForm.value.primary_report_id || !createForm.value.title) {
    ElMessage.warning('请完善病例标题和首份报告')
    return
  }
  try {
    createSubmitting.value = true
    await api.post('/api/diagnosis/cases/', createForm.value)
    ElMessage.success('病例创建成功')
    createDialogVisible.value = false
    await fetchCases()
  } catch (error) {
    console.error('创建病例失败', error)
    ElMessage.error(error.response?.data?.message || '创建病例失败')
  } finally {
    createSubmitting.value = false
  }
}

const openEventDialog = () => {
  if (!selectedCase.value) {
    ElMessage.warning('请先选择病例')
    return
  }
  eventForm.value = {
    event_type: 'doctor_note',
    description: '',
    related_report_id: null,
    next_followup_date: ''
  }
  eventDialogVisible.value = true
}

const submitEvent = async () => {
  if (!selectedCase.value) return
  if (!eventForm.value.description) {
    ElMessage.warning('请填写事件描述')
    return
  }
  try {
    eventSubmitting.value = true
    await api.post(`/api/diagnosis/cases/${selectedCase.value.id}/events/`, eventForm.value)
    ElMessage.success('记录已添加')
    eventDialogVisible.value = false
    await fetchCaseDetail(selectedCase.value.id)
  } catch (error) {
    console.error('新增记录失败', error)
    ElMessage.error(error.response?.data?.message || '新增记录失败')
  } finally {
    eventSubmitting.value = false
  }
}

const updateCaseStatus = async () => {
  if (!selectedCase.value) return
  try {
    await api.patch(`/api/diagnosis/cases/${selectedCase.value.id}/`, {
      status: selectedCase.value.status
    })
    ElMessage.success('病例状态已更新')
    await fetchCases()
  } catch (error) {
    console.error('更新病例失败', error)
    ElMessage.error(error.response?.data?.message || '更新病例失败')
  }
}

const previewReport = async (report) => {
  try {
    const response = await api.get(`/api/diagnosis/reports/${report.id}/download/`, {
      responseType: 'blob'
    })
    const file = new Blob([response.data], { type: 'application/pdf' })
    const url = window.URL.createObjectURL(file)
    window.open(url)
  } catch (error) {
    console.error('预览报告失败', error)
    ElMessage.error(error.response?.data?.message || '预览报告失败')
  }
}

const previewPrimaryReport = () => {
  if (selectedCase.value?.primary_report_info) {
    previewReport(selectedCase.value.primary_report_info)
  }
}

const statusLabel = (status) => {
  const map = {
    active: '进行中',
    closed: '已结束',
    archived: '已归档'
  }
  return map[status] || status
}

const statusType = (status) => {
  const map = {
    active: 'success',
    closed: 'info',
    archived: 'warning'
  }
  return map[status] || 'info'
}

const timelineType = (type) => {
  const map = {
    ai_report: 'primary',
    doctor_note: 'info',
    follow_up: 'warning',
    treatment: 'success'
  }
  return map[type] || 'info'
}

const eventTypeLabel = (type) => {
  const option = eventTypeOptions.find(item => item.value === type)
  return option ? option.label : type
}

const formatDateTime = (val) => {
  if (!val) return '-'
  return new Date(val).toLocaleString('zh-CN')
}

onMounted(() => {
  if (isDoctor.value) {
    fetchCases()
  }
})
</script>

<style scoped>
.medical-record-page {
  padding: 20px;
}

.sidebar-card,
.detail-card {
  height: calc(100vh - 120px);
}

.sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 10px;
}

.sidebar-header h3 {
  margin: 0;
}

.subtitle {
  margin: 4px 0 0;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}

.filter-bar {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.filter-bar .el-select {
  width: 120px;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
}

.detail-header h2 {
  margin: 0;
}

.patient-info {
  margin-top: 4px;
  color: var(--el-text-color-secondary);
}

.detail-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.case-summary {
  margin-bottom: 24px;
}

.section-title {
  margin: 20px 0 12px;
  font-size: 16px;
  font-weight: 600;
}

.timeline {
  max-height: 420px;
  overflow-y: auto;
  padding-right: 8px;
}

.event-item p {
  margin: 4px 0;
  color: var(--el-text-color-regular);
}

.event-desc {
  font-size: 14px;
}

.medical-record-page :deep(.el-table__row.is-selected) {
  background-color: #ecf5ff;
}
</style>

