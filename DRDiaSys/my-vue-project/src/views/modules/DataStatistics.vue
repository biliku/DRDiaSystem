<template>
  <div class="data-statistics-container">
    <!-- 错误提示 -->
    <el-alert
      v-if="errorMessage"
      :title="errorMessage"
      type="error"
      show-icon
      closable
      @close="errorMessage = ''"
      style="margin-bottom: 20px;"
    />

    <!-- 顶部统计卡片 -->
    <div class="stats-cards">
      <!-- 用户统计 -->
      <div class="stat-card">
        <div class="stat-icon users-icon">
          <el-icon><User /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">总用户数</p>
          <h3 class="stat-value">{{ statistics.totalUsers }}</h3>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon doctors-icon">
          <el-icon><UserFilled /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">医生数量</p>
          <h3 class="stat-value">{{ statistics.doctorCount }}</h3>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon patients-icon">
          <el-icon><Avatar /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">患者数量</p>
          <h3 class="stat-value">{{ statistics.patientCount }}</h3>
        </div>
      </div>

      <!-- 诊断统计 -->
      <div class="stat-card">
        <div class="stat-icon diagnosis-icon">
          <el-icon><Monitor /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">诊断任务</p>
          <h3 class="stat-value">{{ diagnosisStats.tasks?.total || 0 }}</h3>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon reports-icon">
          <el-icon><Document /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">诊断报告</p>
          <h3 class="stat-value">{{ diagnosisStats.reports?.total || 0 }}</h3>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon cases-icon">
          <el-icon><Folder /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">病例记录</p>
          <h3 class="stat-value">{{ diagnosisStats.cases?.total || 0 }}</h3>
        </div>
      </div>

      <!-- 治疗统计 -->
      <div class="stat-card">
        <div class="stat-icon treatment-icon">
          <el-icon><FirstAidKit /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">治疗方案</p>
          <h3 class="stat-value">{{ treatmentStats.plans?.total || 0 }}</h3>
        </div>
      </div>
      <div class="stat-card">
        <div icon="chat-icon">
          <el-icon><ChatDotRound /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">医患会话</p>
          <h3 class="stat-value">{{ treatmentStats.conversations?.total || 0 }}</h3>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon messages-icon">
          <el-icon><ChatLineRound /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">消息总数</p>
          <h3 class="stat-value">{{ treatmentStats.messages?.total || 0 }}</h3>
        </div>
      </div>

      <!-- 数据集统计 -->
      <div class="stat-card">
        <div class="stat-icon images-icon">
          <el-icon><Picture /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">数据集影像</p>
          <h3 class="stat-value">{{ statistics.datasetImageCount }}</h3>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon upload-icon">
          <el-icon><Upload /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">患者上传影像</p>
          <h3 class="stat-value">{{ statistics.patientImageCount }}</h3>
        </div>
      </div>
    </div>

    <!-- 图表区域 -->
    <div class="charts-grid">
      <!-- 诊断任务状态分布 -->
      <el-card class="chart-card" shadow="never">
        <template #header>
          <div class="chart-header">
            <span>诊断任务状态分布</span>
          </div>
        </template>
        <div ref="taskStatusChartRef" class="chart-container"></div>
      </el-card>

      <!-- 诊断报告状态分布 -->
      <el-card class="chart-card" shadow="never">
        <template #header>
          <div class="chart-header">
            <span>诊断报告状态分布</span>
          </div>
        </template>
        <div ref="reportStatusChartRef" class="chart-container"></div>
      </el-card>

      <!-- 治疗方案状态分布 -->
      <el-card class="chart-card" shadow="never">
        <template #header>
          <div class="chart-header">
            <span>治疗方案状态分布</span>
          </div>
        </template>
        <div ref="planStatusChartRef" class="chart-container"></div>
      </el-card>

      <!-- 病灶类型分布 -->
      <el-card class="chart-card" shadow="never">
        <template #header>
          <div class="chart-header">
            <span>病灶类型分布</span>
          </div>
        </template>
        <div ref="lesionChartRef" class="chart-container"></div>
      </el-card>

      <!-- 7天诊断趋势 -->
      <el-card class="chart-card" shadow="never">
        <template #header>
          <div class="chart-header">
            <span>7天诊断趋势</span>
          </div>
        </template>
        <div ref="diagnosisTrendChartRef" class="chart-container"></div>
      </el-card>

      <!-- 7天治疗方案趋势 -->
      <el-card class="chart-card" shadow="never">
        <template #header>
          <div class="chart-header">
            <span>7天治疗方案趋势</span>
          </div>
        </template>
        <div ref="treatmentTrendChartRef" class="chart-container"></div>
      </el-card>

      <!-- 用户角色分布 -->
      <el-card class="chart-card" shadow="never">
        <template #header>
          <div class="chart-header">
            <span>用户角色分布</span>
          </div>
        </template>
        <div ref="roleChartRef" class="chart-container"></div>
      </el-card>

      <!-- 医患沟通统计 -->
      <el-card class="chart-card" shadow="never">
        <template #header>
          <div class="chart-header">
            <span>医患沟通统计</span>
          </div>
        </template>
        <div ref="chatChartRef" class="chart-container"></div>
      </el-card>
    </div>

    <!-- 刷新按钮 -->
    <div class="refresh-btn-container">
      <el-button type="primary" :loading="loading" @click="fetchAllStatistics">
        <el-icon><RefreshRight /></el-icon>
        刷新数据
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { User, UserFilled, Avatar, Picture, Upload, Document, RefreshRight, Monitor, Folder, FirstAidKit, ChatDotRound, ChatLineRound } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import api from '../../api'

// 状态变量
const loading = ref(false)
const errorMessage = ref('')
const taskStatusChartRef = ref(null)
const reportStatusChartRef = ref(null)
const planStatusChartRef = ref(null)
const lesionChartRef = ref(null)
const diagnosisTrendChartRef = ref(null)
const treatmentTrendChartRef = ref(null)
const roleChartRef = ref(null)
const chatChartRef = ref(null)

let taskStatusChart = null
let reportStatusChart = null
let planStatusChart = null
let lesionChart = null
let diagnosisTrendChart = null
let treatmentTrendChart = null
let roleChart = null
let chatChart = null

// 统计数据
const statistics = ref({
  totalUsers: 0,
  doctorCount: 0,
  patientCount: 0,
  adminCount: 0,
  datasetImageCount: 0,
  patientImageCount: 0,
  reportCount: 0
})

const diagnosisStats = ref({
  tasks: { total: 0, pending: 0, processing: 0, completed: 0, failed: 0 },
  reports: { total: 0, draft: 0, confirmed: 0, archived: 0 },
  cases: { total: 0, active: 0, closed: 0 },
  lesion_types: {},
  daily_tasks: [],
  daily_reports: []
})

const treatmentStats = ref({
  plans: { total: 0, pending: 0, in_progress: 0, completed: 0, cancelled: 0 },
  ai_recommended: 0,
  doctor_created: 0,
  executions: { total: 0, follow_ups_completed: 0 },
  conversations: { total: 0, active: 0 },
  messages: { total: 0, unread_patient: 0, unread_doctor: 0 },
  templates: { total: 0, public: 0 },
  daily_plans: [],
  daily_messages: []
})

// 获取所有统计数据
const fetchAllStatistics = async () => {
  loading.value = true
  errorMessage.value = ''
  try {
    await Promise.all([
      fetchRoleStatistics(),
      fetchDatasetStatistics(),
      fetchPatientImageStatistics(),
      fetchDiagnosisStatistics(),
      fetchTreatmentStatistics()
    ])
  } catch (error) {
    console.error('获取统计数据失败', error)
    errorMessage.value = '获取统计数据失败，请确保已登录且有权限访问'
  } finally {
    loading.value = false
  }
}

// 获取用户角色统计
const fetchRoleStatistics = async () => {
  try {
    const { data } = await api.get('/api/users/role_statistics/')
    statistics.value.totalUsers = data.total_users || 0
    statistics.value.doctorCount = data.roles?.find(r => r.name === 'doctor')?.user_count || 0
    statistics.value.patientCount = data.roles?.find(r => r.name === 'patient')?.user_count || 0
    statistics.value.adminCount = data.roles?.find(r => r.name === 'admin')?.user_count || 0
    updateRoleChart(data.roles || [])
  } catch (error) {
    console.error('获取角色统计失败', error)
    statistics.value.totalUsers = '-'
    statistics.value.doctorCount = '-'
    statistics.value.patientCount = '-'
  }
}

// 获取数据集统计
const fetchDatasetStatistics = async () => {
  try {
    const { data } = await api.get('/api/datasets/statistics/')
    statistics.value.datasetImageCount = data.total_images || 0
  } catch (error) {
    console.error('获取数据集统计失败', error)
    statistics.value.datasetImageCount = '-'
  }
}

// 获取患者上传影像统计
const fetchPatientImageStatistics = async () => {
  try {
    const { data } = await api.get('/api/datasets/admin/patient-images/')
    const patients = Array.isArray(data) ? data : []
    let totalImages = 0
    patients.forEach(patient => {
      totalImages += patient.total_images || 0
    })
    statistics.value.patientImageCount = totalImages
  } catch (error) {
    console.error('获取患者影像统计失败', error)
    statistics.value.patientImageCount = '-'
  }
}

// 获取诊断统计
const fetchDiagnosisStatistics = async () => {
  try {
    const { data } = await api.get('/api/diagnosis/statistics/')
    diagnosisStats.value = data || diagnosisStats.value
    updateTaskStatusChart(data?.tasks || {})
    updateReportStatusChart(data?.reports || {})
    updateLesionChart(data?.lesion_types || {})
    updateDiagnosisTrendChart(data?.daily_tasks || [], data?.daily_reports || [])
  } catch (error) {
    console.error('获取诊断统计失败', error)
  }
}

// 获取治疗统计
const fetchTreatmentStatistics = async () => {
  try {
    const { data } = await api.get('/api/treatment/statistics/')
    treatmentStats.value = data || treatmentStats.value
    updatePlanStatusChart(data?.plans || {})
    updateTreatmentTrendChart(data?.daily_plans || [], data?.daily_messages || [])
    updateChatChart(data || {})
  } catch (error) {
    console.error('获取治疗统计失败', error)
  }
}

// 诊断任务状态饼图
const updateTaskStatusChart = (tasks) => {
  if (!taskStatusChart) {
    taskStatusChart = echarts.init(taskStatusChartRef.value)
  }

  const hasData = (tasks.total || 0) > 0
  const chartData = hasData ? [
    { name: '待处理', value: tasks.pending || 0 },
    { name: '处理中', value: tasks.processing || 0 },
    { name: '已完成', value: tasks.completed || 0 },
    { name: '失败', value: tasks.failed || 0 }
  ].filter(item => item.value > 0) : []

  const option = {
    title: hasData ? { text: '' } : { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c}个 ({d}%)' },
    legend: hasData ? { bottom: '0%', left: 'center' } : { show: false },
    color: ['#fac858', '#91cc75', '#5470c6', '#ee6666'],
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
      label: { show: false },
      emphasis: hasData ? { label: { show: true, fontSize: 14, fontWeight: 'bold' } } : { label: { show: false } },
      data: chartData
    }]
  }
  taskStatusChart.setOption(option, true)
}

// 诊断报告状态饼图
const updateReportStatusChart = (reports) => {
  if (!reportStatusChart) {
    reportStatusChart = echarts.init(reportStatusChartRef.value)
  }

  const hasData = (reports.total || 0) > 0
  const chartData = hasData ? [
    { name: '草稿', value: reports.draft || 0 },
    { name: '已确认', value: reports.confirmed || 0 },
    { name: '已归档', value: reports.archived || 0 }
  ].filter(item => item.value > 0) : []

  const option = {
    title: hasData ? { text: '' } : { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c}份 ({d}%)' },
    legend: hasData ? { bottom: '0%', left: 'center' } : { show: false },
    color: ['#ee6666', '#fac858', '#91cc75'],
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
      label: { show: false },
      emphasis: hasData ? { label: { show: true, fontSize: 14, fontWeight: 'bold' } } : { label: { show: false } },
      data: chartData
    }]
  }
  reportStatusChart.setOption(option, true)
}

// 治疗方案状态饼图
const updatePlanStatusChart = (plans) => {
  if (!planStatusChart) {
    planStatusChart = echarts.init(planStatusChartRef.value)
  }

  const hasData = (plans.total || 0) > 0
  const chartData = hasData ? [
    { name: '待执行', value: plans.pending || 0 },
    { name: '进行中', value: plans.in_progress || 0 },
    { name: '已完成', value: plans.completed || 0 },
    { name: '已取消', value: plans.cancelled || 0 }
  ].filter(item => item.value > 0) : []

  const option = {
    title: hasData ? { text: '' } : { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c}个 ({d}%)' },
    legend: hasData ? { bottom: '0%', left: 'center' } : { show: false },
    color: ['#fac858', '#5470c6', '#91cc75', '#ee6666'],
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
      label: { show: false },
      emphasis: hasData ? { label: { show: true, fontSize: 14, fontWeight: 'bold' } } : { label: { show: false } },
      data: chartData
    }]
  }
  planStatusChart.setOption(option, true)
}

// 病灶类型分布
const updateLesionChart = (lesionTypes) => {
  if (!lesionChart) {
    lesionChart = echarts.init(lesionChartRef.value)
  }

  const lesionNames = {
    'microaneurysms': '微动脉瘤',
    'hemorrhages': '出血',
    'neovascularization': '新生血管',
    'hard_exudates': '硬性渗出',
    'soft_exudates': '软性渗出',
    'fibrous_proliferation': '纤维增生',
    'vitreous_hemorrhage': '玻璃体积血',
    'traction_detachment': '牵拉性视网膜脱离'
  }

  const chartData = Object.entries(lesionTypes || {}).map(([key, value]) => ({
    name: lesionNames[key] || key,
    value: value.count || 0
  })).filter(item => item.value > 0)

  const hasData = chartData.length > 0

  const option = {
    title: hasData ? { text: '' } : { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: hasData ? { left: '3%', right: '4%', bottom: '3%', containLabel: true } : { show: false },
    xAxis: hasData ? { type: 'category', data: chartData.map(d => d.name), axisLabel: { rotate: 45 } } : { show: false },
    yAxis: hasData ? { type: 'value' } : { show: false },
    series: [{
      type: 'bar',
      data: chartData.map(d => d.value),
      itemStyle: { color: '#5470c6' },
      label: hasData ? { show: true, position: 'top' } : { show: false }
    }]
  }
  lesionChart.setOption(option, true)
}

// 7天诊断趋势折线图
const updateDiagnosisTrendChart = (dailyTasks, dailyReports) => {
  if (!diagnosisTrendChart) {
    diagnosisTrendChart = echarts.init(diagnosisTrendChartRef.value)
  }

  const hasData = (dailyTasks?.length > 0 && dailyTasks.some(d => d.count > 0)) ||
                  (dailyReports?.length > 0 && dailyReports.some(d => d.count > 0))

  const option = {
    title: hasData ? { text: '' } : { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    tooltip: { trigger: 'axis' },
    legend: hasData ? { data: ['诊断任务', '诊断报告'], bottom: '0%' } : { show: false },
    grid: hasData ? { left: '3%', right: '4%', bottom: '15%', containLabel: true } : { show: false },
    xAxis: hasData ? { type: 'category', data: dailyTasks.map(d => d.date) } : { show: false },
    yAxis: hasData ? { type: 'value' } : { show: false },
    series: [
      {
        name: '诊断任务',
        type: 'line',
        data: dailyTasks?.map(d => d.count) || [],
        smooth: true,
        itemStyle: { color: '#5470c6' },
        areaStyle: hasData ? { color: 'rgba(84, 112, 198, 0.2)' } : undefined
      },
      {
        name: '诊断报告',
        type: 'line',
        data: dailyReports?.map(d => d.count) || [],
        smooth: true,
        itemStyle: { color: '#91cc75' },
        areaStyle: hasData ? { color: 'rgba(145, 204, 119, 0.2)' } : undefined
      }
    ]
  }
  diagnosisTrendChart.setOption(option, true)
}

// 7天治疗趋势折线图
const updateTreatmentTrendChart = (dailyPlans, dailyMessages) => {
  if (!treatmentTrendChart) {
    treatmentTrendChart = echarts.init(treatmentTrendChartRef.value)
  }

  const hasData = (dailyPlans?.length > 0 && dailyPlans.some(d => d.count > 0)) ||
                  (dailyMessages?.length > 0 && dailyMessages.some(d => d.count > 0))

  const option = {
    title: hasData ? { text: '' } : { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    tooltip: { trigger: 'axis' },
    legend: hasData ? { data: ['治疗方案', '消息'], bottom: '0%' } : { show: false },
    grid: hasData ? { left: '3%', right: '4%', bottom: '15%', containLabel: true } : { show: false },
    xAxis: hasData ? { type: 'category', data: dailyPlans.map(d => d.date) } : { show: false },
    yAxis: hasData ? { type: 'value' } : { show: false },
    series: [
      {
        name: '治疗方案',
        type: 'line',
        data: dailyPlans?.map(d => d.count) || [],
        smooth: true,
        itemStyle: { color: '#fac858' }
      },
      {
        name: '消息',
        type: 'line',
        data: dailyMessages?.map(d => d.count) || [],
        smooth: true,
        itemStyle: { color: '#ee6666' }
      }
    ]
  }
  treatmentTrendChart.setOption(option, true)
}

// 用户角色分布饼图
const updateRoleChart = (roles) => {
  if (!roleChart) {
    roleChart = echarts.init(roleChartRef.value)
  }

  const chartData = (roles || []).map(role => ({
    name: getRoleName(role.name),
    value: role.user_count
  })).filter(item => item.value > 0)

  const hasData = chartData.length > 0

  const option = {
    title: hasData ? { text: '' } : { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c}人 ({d}%)' },
    legend: hasData ? { bottom: '0%', left: 'center' } : { show: false },
    color: ['#5470c6', '#91cc75', '#fac858', '#ee6666'],
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
      label: { show: false },
      emphasis: hasData ? { label: { show: true, fontSize: 14, fontWeight: 'bold' } } : { label: { show: false } },
      data: chartData
    }]
  }
  roleChart.setOption(option, true)
}

// 医患沟通统计
const updateChatChart = (data) => {
  if (!chatChart) {
    chatChart = echarts.init(chatChartRef.value)
  }

  const totalMessages = data?.messages?.total || 0
  const unreadPatient = data?.messages?.unread_patient || 0
  const unreadDoctor = data?.messages?.unread_doctor || 0
  const readMessages = totalMessages - unreadPatient - unreadDoctor

  const chartData = [
    { name: '已读消息', value: readMessages > 0 ? readMessages : 0 },
    { name: '患者未读', value: unreadPatient > 0 ? unreadPatient : 0 },
    { name: '医生未读', value: unreadDoctor > 0 ? unreadDoctor : 0 }
  ].filter(item => item.value > 0)

  const hasData = totalMessages > 0

  const option = {
    title: hasData ? { text: '' } : { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c}条 ({d}%)' },
    legend: hasData ? { bottom: '0%', left: 'center' } : { show: false },
    color: ['#91cc75', '#fac858', '#5470c6'],
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
      label: { show: false },
      emphasis: hasData ? { label: { show: true, fontSize: 14, fontWeight: 'bold' } } : { label: { show: false } },
      data: chartData
    }]
  }
  chatChart.setOption(option, true)
}

// 角色名称转换
const getRoleName = (role) => {
  const map = { admin: '管理员', doctor: '医生', patient: '患者' }
  return map[role] || role
}

// 响应式调整
const handleResize = () => {
  taskStatusChart?.resize()
  reportStatusChart?.resize()
  planStatusChart?.resize()
  lesionChart?.resize()
  diagnosisTrendChart?.resize()
  treatmentTrendChart?.resize()
  roleChart?.resize()
  chatChart?.resize()
}

// 生命周期
onMounted(async () => {
  await fetchAllStatistics()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  taskStatusChart?.dispose()
  reportStatusChart?.dispose()
  planStatusChart?.dispose()
  lesionChart?.dispose()
  diagnosisTrendChart?.dispose()
  treatmentTrendChart?.dispose()
  roleChart?.dispose()
  chatChart?.dispose()
})
</script>

<style scoped>
.data-statistics-container {
  padding: 20px;
  background: #f5f7fa;
  min-height: calc(100vh - 50px);
}

.stats-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}

.stat-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s, box-shadow 0.2s;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: white;
  flex-shrink: 0;
}

.users-icon { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.doctors-icon { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.patients-icon { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
.diagnosis-icon { background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%); }
.reports-icon { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); }
.cases-icon { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); }
.treatment-icon { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); }
.chat-icon { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); }
.messages-icon { background: linear-gradient(135deg, #a6c0fe 0%, #f68084 100%); }
.images-icon { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
.upload-icon { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }

.stat-content {
  flex: 1;
  min-width: 0;
}

.stat-label {
  margin: 0;
  color: #909399;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.stat-value {
  margin: 5px 0 0;
  font-size: 22px;
  font-weight: 600;
  color: #303133;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.chart-card {
  min-height: 320px;
}

.chart-card :deep(.el-card__body) {
  padding: 12px;
}

.chart-header {
  font-weight: 600;
  font-size: 14px;
  color: #303133;
}

.chart-container {
  width: 100%;
  height: 260px;
}

.refresh-btn-container {
  margin-top: 20px;
  text-align: center;
}

@media (max-width: 1400px) {
  .stats-cards {
    grid-template-columns: repeat(4, 1fr);
  }
}

@media (max-width: 1200px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .stats-cards {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
