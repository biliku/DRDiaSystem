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
              <el-button type="success" size="small" @click="openPatientInfoDialog">
                <el-icon><User /></el-icon>
                患者信息
              </el-button>
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
                <p v-if="event.related_plan" class="event-plan">
                  方案编号：{{ event.related_plan.plan_number }}
                  <el-button
                    type="success"
                    link
                    @click="viewTreatmentPlan(event.related_plan.id)"
                  >
                    查看方案
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
          <el-select v-model="eventForm.event_type">
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

    <!-- 患者信息对话框 -->
    <el-dialog
      v-model="patientInfoDialogVisible"
      :title="`患者信息 - ${selectedCase?.patient_name || ''}`"
      width="800px"
      top="5vh"
    >
      <div v-loading="patientInfoLoading">
        <el-tabs v-model="patientInfoActiveTab">
          <!-- 基本信息 -->
          <el-tab-pane label="基本信息" name="basic">
            <el-descriptions :column="2" border v-if="patientInfoData?.patient_info">
              <el-descriptions-item label="姓名">{{ patientInfoData.patient_info.real_name || '-' }}</el-descriptions-item>
              <el-descriptions-item label="性别">{{ getGenderLabel(patientInfoData.patient_info.gender) }}</el-descriptions-item>
              <el-descriptions-item label="出生日期">{{ patientInfoData.patient_info.birth_date || '-' }}</el-descriptions-item>
              <el-descriptions-item label="年龄">{{ patientInfoData.patient_info.age || '-' }}</el-descriptions-item>
              <el-descriptions-item label="联系电话">{{ patientInfoData.patient_info.phone || '-' }}</el-descriptions-item>
              <el-descriptions-item label="邮箱">{{ patientInfoData.patient_info.email || '-' }}</el-descriptions-item>
              <el-descriptions-item label="血型">{{ getBloodTypeLabel(patientInfoData.patient_info.blood_type) }}</el-descriptions-item>
              <el-descriptions-item label="紧急联系人">{{ patientInfoData.patient_info.emergency_contact || '-' }}</el-descriptions-item>
              <el-descriptions-item label="紧急电话">{{ patientInfoData.patient_info.emergency_phone || '-' }}</el-descriptions-item>
              <el-descriptions-item label="地址" :span="2">
                {{ formatAddress(patientInfoData.patient_info) }}
              </el-descriptions-item>
            </el-descriptions>
            <el-empty v-else description="暂无个人信息" :image-size="80" />
          </el-tab-pane>

          <!-- 病情信息 -->
          <el-tab-pane label="病情信息" name="condition">
            <div v-if="patientInfoData?.condition_info">
              <el-descriptions :column="2" border>
                <el-descriptions-item label="是否患有糖尿病">
                  <el-tag :type="patientInfoData.condition_info.has_diabetes ? 'danger' : 'success'" size="small">
                    {{ patientInfoData.condition_info.has_diabetes ? '是' : '否' }}
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="糖尿病类型">
                  {{ getDiabetesTypeLabel(patientInfoData.condition_info.diabetes_type) }}
                </el-descriptions-item>
                <el-descriptions-item label="糖尿病病程">
                  {{ patientInfoData.condition_info.diabetes_duration ? `${patientInfoData.condition_info.diabetes_duration} 年` : '-' }}
                </el-descriptions-item>
                <el-descriptions-item label="血糖水平">
                  {{ patientInfoData.condition_info.blood_sugar_level ? `${patientInfoData.condition_info.blood_sugar_level} mmol/L` : '-' }}
                </el-descriptions-item>
                <el-descriptions-item label="糖化血红蛋白">
                  {{ patientInfoData.condition_info.hba1c ? `${patientInfoData.condition_info.hba1c}%` : '-' }}
                </el-descriptions-item>
              </el-descriptions>

              <el-divider content-position="left">症状信息</el-divider>
              <el-descriptions :column="1" border size="small">
                <el-descriptions-item label="症状列表">
                  <el-tag
                    v-for="symptom in patientInfoData.condition_info.symptoms"
                    :key="symptom"
                    type="warning"
                    size="small"
                    style="margin-right: 5px; margin-bottom: 5px;"
                  >
                    {{ getSymptomLabel(symptom) }}
                  </el-tag>
                  <span v-if="!patientInfoData.condition_info.symptoms?.length">无</span>
                </el-descriptions-item>
                <el-descriptions-item label="症状描述">{{ patientInfoData.condition_info.symptom_description || '无' }}</el-descriptions-item>
                <el-descriptions-item label="症状持续时间">{{ patientInfoData.condition_info.symptom_duration || '未知' }}</el-descriptions-item>
              </el-descriptions>

              <el-divider content-position="left">病史信息</el-divider>
              <el-descriptions :column="1" border size="small">
                <el-descriptions-item label="既往病史">{{ patientInfoData.condition_info.medical_history || '无' }}</el-descriptions-item>
                <el-descriptions-item label="家族病史">{{ patientInfoData.condition_info.family_history || '无' }}</el-descriptions-item>
                <el-descriptions-item label="用药史">{{ patientInfoData.condition_info.medication_history || '无' }}</el-descriptions-item>
                <el-descriptions-item label="过敏史">
                  <el-tag v-if="patientInfoData.condition_info.allergy_history" type="danger" size="small">
                    {{ patientInfoData.condition_info.allergy_history }}
                  </el-tag>
                  <span v-else>无</span>
                </el-descriptions-item>
                <el-descriptions-item label="其他疾病">{{ patientInfoData.condition_info.other_conditions || '无' }}</el-descriptions-item>
                <el-descriptions-item label="备注">{{ patientInfoData.condition_info.notes || '无' }}</el-descriptions-item>
              </el-descriptions>
            </div>
            <el-empty v-else description="暂无病情信息" :image-size="80" />
          </el-tab-pane>
        </el-tabs>
      </div>
      <template #footer>
        <el-button @click="patientInfoDialogVisible = false">关闭</el-button>
      </template>
    </el-dialog>

    <!-- 治疗方案查看对话框 -->
    <el-dialog
      v-model="treatmentPlanDialogVisible"
      :title="`治疗方案详情 - ${treatmentPlanData?.plan_number || ''}`"
      width="900px"
      top="5vh"
      destroy-on-close
    >
      <div v-loading="treatmentPlanLoading">
        <div v-if="treatmentPlanData">
          <!-- 基本信息 -->
          <el-descriptions :column="2" border>
            <el-descriptions-item label="方案编号">{{ treatmentPlanData.plan_number }}</el-descriptions-item>
            <el-descriptions-item label="状态">
              <el-tag :type="treatmentPlanData.status === 'completed' ? 'success' : treatmentPlanData.status === 'active' ? 'primary' : 'info'">
                {{ treatmentPlanData.status === 'completed' ? '已完成' : treatmentPlanData.status === 'active' ? '执行中' : treatmentPlanData.status === 'confirmed' ? '已确认' : '草稿' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="创建时间">{{ treatmentPlanData.created_at }}</el-descriptions-item>
            <el-descriptions-item label="创建医生">{{ treatmentPlanData.created_by_name }}</el-descriptions-item>
          </el-descriptions>

          <!-- 血糖血压目标 -->
          <div v-if="hasTargetData" class="plan-section">
            <h4>血糖血压目标</h4>
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item label="空腹血糖">
                {{ formatBsTarget(treatmentPlanData.blood_sugar_target, 'fasting') }}
              </el-descriptions-item>
              <el-descriptions-item label="餐后血糖">
                {{ formatBsTarget(treatmentPlanData.blood_sugar_target, 'postprandial') }}
              </el-descriptions-item>
              <el-descriptions-item label="糖化血红蛋白">
                {{ formatBsTarget(treatmentPlanData.blood_sugar_target, 'hba1c') }}
              </el-descriptions-item>
              <el-descriptions-item label="血压">
                {{ formatBpTarget(treatmentPlanData.blood_pressure_target) }}
              </el-descriptions-item>
            </el-descriptions>
          </div>

          <!-- 药物治疗 -->
          <div v-if="treatmentPlanData.medications?.length" class="plan-section">
            <h4>药物治疗</h4>
            <el-table :data="treatmentPlanData.medications" size="small" border>
              <el-table-column prop="category" label="类别" width="100">
                <template #default="scope">
                  <el-tag size="small">{{ getMedCategoryLabel(scope.row.category) }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="name" label="药物名称" width="120" />
              <el-table-column label="给药方案" min-width="200">
                <template #default="scope">
                  {{ scope.row.route || '' }} {{ scope.row.dose || '' }} {{ scope.row.frequency || '' }}
                </template>
              </el-table-column>
              <el-table-column prop="duration" label="疗程" width="100" />
            </el-table>
          </div>

          <!-- 眼科治疗 -->
          <div v-if="treatmentPlanData.treatments?.length" class="plan-section">
            <h4>眼科治疗</h4>
            <el-table :data="treatmentPlanData.treatments" size="small" border>
              <el-table-column label="类别" width="120">
                <template #default="scope">
                  <el-tag size="small">{{ getTreatmentCategoryLabel(scope.row.category) }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="item" label="具体项目" />
              <el-table-column label="频次" width="100">
                <template #default="scope">
                  {{ scope.row.frequency }}{{ scope.row.frequency_unit }}
                </template>
              </el-table-column>
              <el-table-column label="疗程" width="100">
                <template #default="scope">
                  {{ scope.row.course }}{{ scope.row.course_unit }}
                </template>
              </el-table-column>
            </el-table>
          </div>

          <!-- 生活方式指导 -->
          <div v-if="treatmentPlanData.diet_guidance || treatmentPlanData.exercise_guidance" class="plan-section">
            <h4>生活方式指导</h4>
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item v-if="treatmentPlanData.diet_guidance" label="饮食指导">{{ treatmentPlanData.diet_guidance }}</el-descriptions-item>
              <el-descriptions-item v-if="treatmentPlanData.exercise_guidance" label="运动指导">{{ treatmentPlanData.exercise_guidance }}</el-descriptions-item>
            </el-descriptions>
          </div>

          <!-- 监测计划 -->
          <div v-if="monitoringPlanData.length" class="plan-section">
            <h4>监测计划</h4>
            <el-table :data="monitoringPlanData" size="small" border>
              <el-table-column prop="item" label="监测项目" />
              <el-table-column prop="frequency" label="监测频率" width="150" />
              <el-table-column prop="target" label="目标值" width="150" />
            </el-table>
          </div>

          <!-- 警示症状 -->
          <div v-if="treatmentPlanData.warning_symptoms" class="plan-section">
            <h4>警示症状</h4>
            <p>{{ treatmentPlanData.warning_symptoms }}</p>
          </div>
        </div>
        <el-empty v-else description="暂无方案信息" :image-size="80" />
      </div>
      <template #footer>
        <el-button @click="treatmentPlanDialogVisible = false">关闭</el-button>
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
import { Search, Plus, Edit, User } from '@element-plus/icons-vue'
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

// 患者信息相关
const patientInfoDialogVisible = ref(false)
const patientInfoLoading = ref(false)
const patientInfoActiveTab = ref('basic')
const patientInfoData = ref(null)

// 治疗方案查看相关
const treatmentPlanDialogVisible = ref(false)
const treatmentPlanLoading = ref(false)
const treatmentPlanData = ref(null)

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

// 判断是否有血糖血压目标数据
const hasTargetData = computed(() => {
  if (!treatmentPlanData.value) return false
  const bs = treatmentPlanData.value.blood_sugar_target
  const bp = treatmentPlanData.value.blood_pressure_target
  
  // 检查血糖目标 - 兼容 min_max 格式和字符串格式
  const hasBloodSugar = bs && (
    // 新格式：{fasting_min, fasting_max, ...}
    (bs.fasting_min !== undefined || bs.fasting_max !== undefined) ||
    (bs.postprandial_min !== undefined || bs.postprandial_max !== undefined) ||
    (bs.hba1c_min !== undefined || bs.hba1c_max !== undefined) ||
    // 旧格式：{fasting: 'xxx', ...}
    (typeof bs === 'object' && (
      (bs.fasting && typeof bs.fasting === 'string' && bs.fasting.trim()) ||
      (bs.postprandial && typeof bs.postprandial === 'string' && bs.postprandial.trim()) ||
      (bs.hba1c && typeof bs.hba1c === 'string' && bs.hba1c.trim())
    ))
  )
  
  // 检查血压目标
  const hasBloodPressure = bp && (
    // 新格式：{systolic_min, systolic_max, ...}
    (bp.systolic_min !== undefined || bp.systolic_max !== undefined) ||
    (bp.diastolic_min !== undefined || bp.diastolic_max !== undefined) ||
    // 旧格式：{systolic: 'xxx', ...}
    (typeof bp === 'object' && (
      (bp.systolic && typeof bp.systolic === 'string' && bp.systolic.trim()) ||
      (bp.diastolic && typeof bp.diastolic === 'string' && bp.diastolic.trim())
    ))
  )
  
  return hasBloodSugar || hasBloodPressure
})

// 处理监测计划数据
const monitoringPlanData = computed(() => {
  if (!treatmentPlanData.value) return []
  const mp = treatmentPlanData.value.monitoring_plan
  if (!mp) return []
  if (Array.isArray(mp)) return mp.filter(item => item && item.item)
  return []
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

// 查看治疗方案详情
const viewTreatmentPlan = async (planId) => {
  treatmentPlanDialogVisible.value = true
  treatmentPlanLoading.value = true
  treatmentPlanData.value = null

  try {
    const { data } = await api.get(`/api/treatment/plans/${planId}/`)
    treatmentPlanData.value = data
  } catch (error) {
    console.error('获取治疗方案详情失败', error)
    ElMessage.error(error.response?.data?.message || '获取治疗方案详情失败')
    treatmentPlanDialogVisible.value = false
  } finally {
    treatmentPlanLoading.value = false
  }
}

// 获取患者信息
const openPatientInfoDialog = async () => {
  if (!selectedCase.value?.patient) {
    ElMessage.warning('无法获取患者ID')
    return
  }
  patientInfoDialogVisible.value = true
  patientInfoActiveTab.value = 'basic'
  patientInfoData.value = null
  patientInfoLoading.value = true

  try {
    const { data } = await api.get(`/api/patient/${selectedCase.value.patient}/medical-info/`)
    patientInfoData.value = data
  } catch (error) {
    console.error('获取患者信息失败', error)
    ElMessage.error(error.response?.data?.message || '获取患者信息失败')
  } finally {
    patientInfoLoading.value = false
  }
}

const getGenderLabel = (gender) => {
  const map = { 'M': '男', 'F': '女', 'O': '其他' }
  return map[gender] || '-'
}

const getBloodTypeLabel = (bloodType) => {
  const map = {
    'A': 'A型',
    'B': 'B型',
    'AB': 'AB型',
    'O': 'O型',
    'UNKNOWN': '未知'
  }
  return map[bloodType] || '-'
}

const getMedCategoryLabel = (category) => {
  const map = {
    'ophthalmic': '眼科用药',
    'systemic': '全身用药',
    'hypoglycemic': '降糖药',
    'antihypertensive': '降压药',
    'lipid_lowering': '降脂药',
    'nutrient': '营养神经'
  }
  return map[category] || category || '-'
}

// 翻译治疗类别
const getTreatmentCategoryLabel = (category) => {
  const map = {
    'anti_vegf': '抗VEGF',
    'laser': '激光治疗',
    'surgical': '手术治疗',
    'other': '其他治疗'
  }
  return map[category] || category || '未知'
}

// 格式化血糖目标（兼容 min_max 格式和字符串格式）
const formatBsTarget = (bs, field) => {
  if (!bs) return '-'
  // 新格式：{fasting_min, fasting_max, fasting_unit}
  if (bs[`${field}_min`] !== undefined || bs[`${field}_max`] !== undefined) {
    const min = bs[`${field}_min`]
    const max = bs[`${field}_max`]
    const unit = bs[`${field}_unit`] || ''
    if (min && max) {
      return `${min}-${max} ${unit}`
    } else if (min) {
      return `${min} ${unit}`
    } else if (max) {
      return `${max} ${unit}`
    }
    return '-'
  }
  // 旧格式：{fasting: '4.4-7.0 mmol/L'}
  if (bs[field]) return bs[field]
  return '-'
}

// 格式化血压目标
const formatBpTarget = (bp) => {
  if (!bp) return '-'
  // 新格式：{systolic_min, systolic_max, systolic_unit, diastolic_min, diastolic_max, diastolic_unit}
  if (bp.systolic_min !== undefined || bp.systolic_max !== undefined) {
    const systolicUnit = bp.systolic_unit || 'mmHg'
    const diastolicUnit = bp.diastolic_unit || 'mmHg'
    
    let result = ''
    if (bp.systolic_min && bp.systolic_max) {
      result += `${bp.systolic_min}-${bp.systolic_max} ${systolicUnit}`
    } else if (bp.systolic_min) {
      result += `${bp.systolic_min} ${systolicUnit}`
    } else if (bp.systolic_max) {
      result += `${bp.systolic_max} ${systolicUnit}`
    }
    
    if (bp.diastolic_min !== undefined || bp.diastolic_max !== undefined) {
      result += ' / '
      if (bp.diastolic_min && bp.diastolic_max) {
        result += `${bp.diastolic_min}-${bp.diastolic_max} ${diastolicUnit}`
      } else if (bp.diastolic_min) {
        result += `${bp.diastolic_min} ${diastolicUnit}`
      } else if (bp.diastolic_max) {
        result += `${bp.diastolic_max} ${diastolicUnit}`
      }
    }
    return result
  }
  // 旧格式：{systolic: 'xxx', diastolic: 'xxx'}
  if (typeof bp === 'object') {
    if (bp.systolic && bp.diastolic) {
      return `${bp.systolic} / ${bp.diastolic}`
    }
    return bp.systolic || bp.diastolic || '-'
  }
  return String(bp)
}

const getDiabetesTypeLabel = (type) => {
  const map = {
    'TYPE1': '1型糖尿病',
    'TYPE2': '2型糖尿病',
    'GESTATIONAL': '妊娠期糖尿病',
    'OTHER': '其他类型',
    'NONE': '无糖尿病'
  }
  return map[type] || '-'
}

const getSymptomLabel = (symptom) => {
  const map = {
    'BLURRED_VISION': '视力模糊',
    'FLOATERS': '飞蚊症',
    'DARK_SPOTS': '暗点',
    'POOR_NIGHT_VISION': '夜视能力差',
    'COLOR_VISION_LOSS': '色觉减退',
    'NONE': '无明显症状'
  }
  return map[symptom] || symptom
}

const formatAddress = (info) => {
  if (!info) return '-'
  const parts = []
  if (info.province) parts.push(info.province)
  if (info.city) parts.push(info.city)
  if (info.district) parts.push(info.district)
  if (info.address) parts.push(info.address)
  return parts.length ? parts.join(' ') : '-'
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
  if (type === 'treatment') return '治疗方案'
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

.detail-actions .el-select,
.detail-actions .el-button {
  margin: 0;
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

.event-plan {
  color: var(--el-color-success);
}

.plan-section {
  margin-top: 16px;
}

.plan-section h4 {
  margin: 12px 0 8px;
  font-size: 15px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.medical-record-page :deep(.el-table__row.is-selected) {
  background-color: #ecf5ff;
}

.patient-info-tabs {
  margin-top: 10px;
}
</style>

