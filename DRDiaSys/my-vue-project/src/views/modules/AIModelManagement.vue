<template>
  <div class="ai-model-management">
    <!-- 顶部统计（可点击筛选） -->
    <div class="stats-row">
      <el-card shadow="hover" class="stat-card" :class="{ 'active-card': activeFilter.type === 'grading' }" @click="filterByType('grading')">
        <div class="stat-icon grading-icon">
          <el-icon><TrendCharts /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">分级模型</p>
          <h3 class="stat-value">{{ modelStats.grading || 0 }}</h3>
        </div>
      </el-card>
      <el-card shadow="hover" class="stat-card" :class="{ 'active-card': activeFilter.type === 'segmentation' }" @click="filterByType('segmentation')">
        <div class="stat-icon segmentation-icon">
          <el-icon><Aim /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">分割模型</p>
          <h3 class="stat-value">{{ modelStats.segmentation || 0 }}</h3>
        </div>
      </el-card>
      <el-card shadow="hover" class="stat-card" :class="{ 'active-card': activeFilter.status === 'production' }" @click="filterByStatus('production')">
        <div class="stat-icon production-icon">
          <el-icon><Monitor /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">生产环境</p>
          <h3 class="stat-value">{{ modelStats.production || 0 }}</h3>
        </div>
      </el-card>
      <el-card shadow="hover" class="stat-card" :class="{ 'active-card': activeFilter.status === 'development' }" @click="filterByStatus('development')">
        <div class="stat-icon active-icon">
          <el-icon><CircleCheck /></el-icon>
        </div>
        <div class="stat-content">
          <p class="stat-label">开发中</p>
          <h3 class="stat-value">{{ modelStats.development || 0 }}</h3>
        </div>
      </el-card>
    </div>

    <!-- 操作栏 -->
    <el-card class="filter-card" shadow="never">
      <div class="filter-bar">
        <div class="filter-tags" v-if="activeFilter.type || activeFilter.status">
          <el-tag v-if="activeFilter.type" closable @close="clearFilter('type')">
            {{ activeFilter.type === 'grading' ? '分级模型' : '分割模型' }}
          </el-tag>
          <el-tag v-if="activeFilter.status" closable @close="clearFilter('status')">
            {{ getStatusLabel(activeFilter.status) }}
          </el-tag>
          <el-button type="primary" link @click="clearAllFilters">清除筛选</el-button>
        </div>
        <div class="actions" style="margin-left: auto;">
          <el-button type="success" @click="openImportDialog">
            <el-icon><Download /></el-icon>
            导入本地模型
          </el-button>
          <el-button @click="fetchModels">
            <el-icon><RefreshRight /></el-icon>
            刷新
          </el-button>
        </div>
      </div>
    </el-card>

    <!-- 模型列表 -->
    <el-card class="table-card" shadow="never">
      <el-table :data="filteredModels" v-loading="loading" stripe>
        <el-table-column prop="name" label="模型名称" min-width="150" />
        <el-table-column prop="model_type_name" label="类型" width="120">
          <template #default="{ row }">
            <el-tag :type="row.model_type === 'grading' ? 'success' : 'warning'">
              {{ row.model_type_name }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="version" label="版本" width="100" />
        <el-table-column prop="status_name" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ row.status_name }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="激活" width="80">
          <template #default="{ row }">
            <el-tag v-if="row.is_active" type="success" effect="dark">已激活</el-tag>
            <el-tag v-else type="info">未激活</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="默认" width="80">
          <template #default="{ row }">
            <el-tag v-if="row.is_default" type="warning" effect="dark">默认</el-tag>
            <el-tag v-else type="info">-</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="accuracy" label="准确率" width="100">
          <template #default="{ row }">
            {{ row.accuracy ? (row.accuracy * 100).toFixed(1) + '%' : '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="model_size_mb" label="大小(MB)" width="100" />
        <el-table-column prop="deployed_at" label="部署时间" width="160">
          <template #default="{ row }">
            {{ row.deployed_at ? formatDate(row.deployed_at) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="220" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link size="small" @click="viewModel(row)">详情</el-button>
            <el-button type="success" link size="small" v-if="!row.is_active" @click="activateModel(row)">激活</el-button>
            <el-button type="warning" link size="small" v-if="!row.is_default" @click="setDefault(row)">设为默认</el-button>
            <el-button type="info" link size="small" @click="viewPerformance(row)">性能</el-button>
            <el-button type="danger" link size="small" @click="deleteModel(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 模型详情对话框 -->
    <el-dialog v-model="detailDialogVisible" title="模型详情" width="800px">
      <div v-if="currentModel" class="model-detail">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="模型名称">{{ currentModel.name }}</el-descriptions-item>
          <el-descriptions-item label="类型">{{ currentModel.model_type_name }}</el-descriptions-item>
          <el-descriptions-item label="版本">{{ currentModel.version }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(currentModel.status)">{{ currentModel.status_name }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="激活状态">
            <el-tag :type="currentModel.is_active ? 'success' : 'info'">{{ currentModel.is_active ? '已激活' : '未激活' }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="默认模型">
            <el-tag :type="currentModel.is_default ? 'warning' : 'info'">{{ currentModel.is_default ? '是' : '否' }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="准确率">{{ currentModel.accuracy ? (currentModel.accuracy * 100).toFixed(1) + '%' : '-' }}</el-descriptions-item>
          <el-descriptions-item label="F1分数">{{ currentModel.f1_score ? (currentModel.f1_score * 100).toFixed(1) + '%' : '-' }}</el-descriptions-item>
          <el-descriptions-item label="模型路径" :span="2">{{ currentModel.model_path || '-' }}</el-descriptions-item>
          <el-descriptions-item label="推理服务" :span="2">{{ currentModel.inference_endpoint || '-' }}</el-descriptions-item>
          <el-descriptions-item label="部署策略">{{ currentModel.deployment_strategy_name }}</el-descriptions-item>
          <el-descriptions-item label="创建人">{{ currentModel.created_by_name || '-' }}</el-descriptions-item>
          <el-descriptions-item label="创建时间">{{ formatDate(currentModel.created_at) }}</el-descriptions-item>
          <el-descriptions-item label="更新时间">{{ formatDate(currentModel.updated_at) }}</el-descriptions-item>
          <el-descriptions-item label="部署时间">{{ currentModel.deployed_at ? formatDate(currentModel.deployed_at) : '-' }}</el-descriptions-item>
          <el-descriptions-item label="模型描述" :span="2">{{ currentModel.description || '-' }}</el-descriptions-item>
          <el-descriptions-item label="更新日志" :span="2">{{ currentModel.changelog || '-' }}</el-descriptions-item>
        </el-descriptions>
      </div>
    </el-dialog>

    <!-- 性能监控对话框 -->
    <el-dialog v-model="performanceDialogVisible" title="性能监控" width="900px">
      <div class="performance-charts">
        <el-card class="performance-chart" shadow="never">
          <template #header>
            <span>推理时间趋势</span>
          </template>
          <div ref="inferenceTimeChartRef" class="chart-container"></div>
        </el-card>
        <el-card class="performance-chart" shadow="never">
          <template #header>
            <span>吞吐量趋势</span>
          </template>
          <div ref="throughputChartRef" class="chart-container"></div>
        </el-card>
      </div>
      <el-card class="performance-table" shadow="never" style="margin-top: 16px;">
        <template #header>
          <span>最近性能日志</span>
        </template>
        <el-table :data="performanceLogs" size="small">
          <el-table-column prop="created_at" label="时间" width="160">
            <template #default="{ row }">
              {{ formatDate(row.created_at) }}
            </template>
          </el-table-column>
          <el-table-column prop="log_type_name" label="类型" width="100" />
          <el-table-column prop="avg_inference_time" label="平均推理时间(ms)" width="150" />
          <el-table-column prop="p95_inference_time" label="P95时间(ms)" width="120" />
          <el-table-column prop="throughput" label="吞吐量" width="100" />
          <el-table-column prop="accuracy" label="准确率" width="100">
            <template #default="{ row }">
              {{ row.accuracy ? (row.accuracy * 100).toFixed(1) + '%' : '-' }}
            </template>
          </el-table-column>
          <el-table-column prop="drift_status" label="漂移状态" width="100">
            <template #default="{ row }">
              <el-tag v-if="row.drift_status" :type="getDriftType(row.drift_status)">{{ row.drift_status }}</el-tag>
              <span v-else>-</span>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </el-dialog>

    <!-- 导入本地模型对话框 -->
    <el-dialog v-model="importDialogVisible" title="导入本地模型" width="600px">
      <el-form :model="importForm" label-width="100px">
        <el-form-item label="模型类型">
          <el-radio-group v-model="importForm.model_type">
            <el-radio value="grading">分级模型</el-radio>
            <el-radio value="segmentation">分割模型</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="模型文件">
          <el-select v-model="importForm.model_path" placeholder="选择模型文件" style="width: 100%;">
            <el-option-group v-for="group in localModelPaths" :key="group.label" :label="group.label">
              <el-option v-for="item in group.options" :key="item.value" :label="item.label" :value="item.value" />
            </el-option-group>
          </el-select>
        </el-form-item>
        <el-form-item label="模型名称">
          <el-input v-model="importForm.name" placeholder="请输入模型名称" />
        </el-form-item>
        <el-form-item label="版本号">
          <el-input v-model="importForm.version" placeholder="如: v1.0.0" />
        </el-form-item>
        <el-form-item label="状态">
          <el-select v-model="importForm.status" style="width: 100%;">
            <el-option label="开发中" value="development" />
            <el-option label="测试中" value="testing" />
            <el-option label="生产环境" value="production" />
          </el-select>
        </el-form-item>
        <el-form-item label="设为激活">
          <el-switch v-model="importForm.is_active" />
        </el-form-item>
        <el-form-item label="设为默认">
          <el-switch v-model="importForm.is_default" />
        </el-form-item>
        <el-form-item label="准确率">
          <el-slider v-model="importForm.accuracy" :min="0" :max="1" :step="0.01" show-stops />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="importForm.description" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="importDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleImport" :loading="saving">确认导入</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { RefreshRight, TrendCharts, Aim, CircleCheck, Monitor, Download } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import api from '../../api'

// 状态
const loading = ref(false)
const saving = ref(false)
const models = ref([])
const activeFilter = ref({ type: '', status: '' })
const modelStats = ref({})

// 对话框
const importDialogVisible = ref(false)
const detailDialogVisible = ref(false)
const performanceDialogVisible = ref(false)
const currentModel = ref(null)

// 导入表单
const importForm = ref({
  model_type: 'grading',
  model_path: '',
  name: '',
  version: 'v1.0.0',
  status: 'production',
  is_active: true,
  is_default: false,
  accuracy: 0.85,
  description: ''
})

// 本地模型路径选项
const localModelPaths = ref([
  { label: '分级模型', options: [
    { value: 'F:\\DRDiaSys\\best_resnet_aptos_enhanced.pth', label: 'best_resnet_aptos_enhanced.pth (增强版ResNet)' },
    { value: 'F:\\DRDiaSys\\best_resnet_aptos_improved1.pth', label: 'best_resnet_aptos_improved1.pth (改进版)' },
    { value: 'F:\\DRDiaSys\\final_model_epoch_60.pth', label: 'final_model_epoch_60.pth' },
    { value: 'F:\\DRDiaSys\\DRDiaSys\\django\\DRDiaSys\\best_resnet_aptos_enhanced.pth', label: 'Django项目内分级模型' }
  ]},
  { label: '分割模型', options: [
    { value: 'F:\\DRDiaSys\\DRDiaSys\\django\\DRDiaSys\\diagnosis\\best_lesion_segmentation_model_v4.pth', label: 'best_lesion_segmentation_model_v4.pth' },
    { value: 'F:\\DRDiaSys\\DRDiaSys\\django\\DRDiaSys\\diagnosis\\best_lesion_segmentation_model_v3.pth', label: 'best_lesion_segmentation_model_v3.pth' },
    { value: 'F:\\DRDiaSys\\DRDiaSys\\django\\DRDiaSys\\diagnosis\\best_lesion_segmentation_model_v2.pth', label: 'best_lesion_segmentation_model_v2.pth' },
    { value: 'F:\\DRDiaSys\\DRDiaSys\\django\\DRDiaSys\\diagnosis\\best_lesion_segmentation_model_v1.pth', label: 'best_lesion_segmentation_model_v1.pth' },
    { value: 'F:\\DRDiaSys\\unet_vessel_segmentation.pth', label: 'unet_vessel_segmentation.pth' },
    { value: 'F:\\DRDiaSys\\unet_epoch30.pth', label: 'unet_epoch30.pth' }
  ]}
])

// 性能监控
const performanceLogs = ref([])
const inferenceTimeChartRef = ref(null)
const throughputChartRef = ref(null)
let inferenceTimeChart = null
let throughputChart = null

// 筛选后的模型
const filteredModels = computed(() => {
  return models.value.filter(m => {
    if (activeFilter.value.type && m.model_type !== activeFilter.value.type) return false
    if (activeFilter.value.status && m.status !== activeFilter.value.status) return false
    return true
  })
})

// 按类型筛选
const filterByType = (type) => {
  if (activeFilter.value.type === type) {
    activeFilter.value.type = ''
  } else {
    activeFilter.value.type = type
  }
}

// 按状态筛选
const filterByStatus = (status) => {
  if (activeFilter.value.status === status) {
    activeFilter.value.status = ''
  } else {
    activeFilter.value.status = status
  }
}

// 清除单个筛选
const clearFilter = (filterType) => {
  activeFilter.value[filterType] = ''
}

// 清除所有筛选
const clearAllFilters = () => {
  activeFilter.value = { type: '', status: '' }
}

// 获取状态标签
const getStatusLabel = (status) => {
  const labels = {
    development: '开发中',
    testing: '测试中',
    production: '生产环境',
    deprecated: '已废弃'
  }
  return labels[status] || status
}

// 获取模型列表
const fetchModels = async () => {
  loading.value = true
  try {
    // 现在使用前端筛选，不需要传筛选参数到后端
    const { data } = await api.get('/api/diagnosis/ai-models/')
    models.value = data
    // 同时更新统计数据
    await fetchStats()
  } catch (error) {
    console.error('获取模型列表失败', error)
    ElMessage.error('获取模型列表失败')
  } finally {
    loading.value = false
  }
}

// 获取模型统计
const fetchStats = async () => {
  try {
    const { data } = await api.get('/api/diagnosis/ai-models/statistics/')
    console.log('统计数据:', data)
    // 兼容不同的返回格式
    const stats = data.models || data || {}
    modelStats.value = {
      grading: stats.grading || 0,
      segmentation: stats.segmentation || 0,
      active: stats.active || 0,
      production: stats.production || 0,
      total: stats.total || 0,
      development: stats.development || 0,
      testing: stats.testing || 0
    }
  } catch (error) {
    console.error('获取统计失败', error)
    // 使用本地模型统计作为备选
    const gradingModels = models.value.filter(m => m.model_type === 'grading').length
    const segmentationModels = models.value.filter(m => m.model_type === 'segmentation').length
    const activeModels = models.value.filter(m => m.is_active).length
    const productionModels = models.value.filter(m => m.status === 'production').length
    modelStats.value = {
      grading: gradingModels,
      segmentation: segmentationModels,
      active: activeModels,
      production: productionModels
    }
  }
}

// 打开导入对话框
const openImportDialog = () => {
  importForm.value = {
    model_type: 'grading',
    model_path: '',
    name: '',
    version: 'v1.0.0',
    status: 'production',
    is_active: true,
    is_default: false,
    accuracy: 0.85,
    description: ''
  }
  importDialogVisible.value = true
}

// 导入模型
const handleImport = async () => {
  if (!importForm.value.model_path) {
    ElMessage.warning('请选择模型文件')
    return
  }
  if (!importForm.value.name) {
    ElMessage.warning('请输入模型名称')
    return
  }

  saving.value = true
  try {
    await api.post('/api/diagnosis/ai-models/', {
      name: importForm.value.name,
      model_type: importForm.value.model_type,
      version: importForm.value.version,
      model_path: importForm.value.model_path,
      status: importForm.value.status,
      is_active: importForm.value.is_active,
      is_default: importForm.value.is_default,
      accuracy: importForm.value.accuracy,
      description: importForm.value.description
    })
    ElMessage.success('模型导入成功')
    importDialogVisible.value = false
    await fetchModels()
    await fetchStats()
  } catch (error) {
    console.error('导入模型失败', error)
    ElMessage.error('导入模型失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    saving.value = false
  }
}

// 查看详情
const viewModel = async (model) => {
  try {
    const { data } = await api.get(`/api/diagnosis/ai-models/${model.id}/`)
    currentModel.value = data
    detailDialogVisible.value = true
  } catch (error) {
    ElMessage.error('获取模型详情失败')
  }
}

// 激活模型
const activateModel = async (model) => {
  try {
    await ElMessageBox.confirm(`确定要激活模型 "${model.name}" 吗？`, '提示', { type: 'warning' })
    await api.post(`/api/diagnosis/ai-models/${model.id}/activate/`)
    ElMessage.success('模型已激活')
    await fetchModels()
    await fetchStats()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('激活模型失败', error)
      ElMessage.error('激活模型失败')
    }
  }
}

// 设为默认
const setDefault = async (model) => {
  try {
    await ElMessageBox.confirm(`确定要将 "${model.name}" 设为默认模型吗？`, '提示', { type: 'warning' })
    await api.post(`/api/diagnosis/ai-models/${model.id}/set-default/`)
    ElMessage.success('已设为默认模型')
    await fetchModels()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('设置默认模型失败', error)
      ElMessage.error('设置默认模型失败')
    }
  }
}

// 查看性能
const viewPerformance = async (model) => {
  currentModel.value = model
  performanceDialogVisible.value = true

  try {
    const { data } = await api.get(`/api/diagnosis/ai-models/${model.id}/performance/`)
    performanceLogs.value = data
    await nextTick()
    updatePerformanceCharts(data)
  } catch (error) {
    console.error('获取性能数据失败', error)
  }
}

// 更新性能图表
const updatePerformanceCharts = (logs) => {
  // 推理时间图表
  if (!inferenceTimeChart) {
    inferenceTimeChart = echarts.init(inferenceTimeChartRef.value)
  }

  const timeLogs = logs.filter(l => l.log_type === 'inference').slice(-20)
  const option1 = {
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: timeLogs.map(l => l.created_at?.slice(0, 16) || '') },
    yAxis: { type: 'value', name: 'ms' },
    series: [
      { name: '平均', type: 'line', data: timeLogs.map(l => l.avg_inference_time), smooth: true },
      { name: 'P95', type: 'line', data: timeLogs.map(l => l.p95_inference_time), smooth: true },
      { name: 'P99', type: 'line', data: timeLogs.map(l => l.p99_inference_time), smooth: true }
    ]
  }
  inferenceTimeChart.setOption(option1)

  // 吞吐量图表
  if (!throughputChart) {
    throughputChart = echarts.init(throughputChartRef.value)
  }

  const option2 = {
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: timeLogs.map(l => l.created_at?.slice(0, 16) || '') },
    yAxis: { type: 'value', name: 'req/s' },
    series: [{
      name: '吞吐量',
      type: 'bar',
      data: timeLogs.map(l => l.throughput),
      itemStyle: { color: '#5470c6' }
    }]
  }
  throughputChart.setOption(option2)
}

// 删除模型
const deleteModel = async (model) => {
  try {
    await ElMessageBox.confirm(`确定要删除模型 "${model.name}" 吗？此操作不可恢复。`, '警告', { type: 'error' })
    await api.delete(`/api/diagnosis/ai-models/${model.id}/`)
    ElMessage.success('模型已删除')
    await fetchModels()
    await fetchStats()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除模型失败', error)
      ElMessage.error('删除模型失败')
    }
  }
}

// 工具函数
const getStatusType = (status) => {
  const map = {
    development: 'info',
    testing: 'warning',
    production: 'success',
    deprecated: 'danger'
  }
  return map[status] || 'info'
}

const getDriftType = (status) => {
  const map = { stable: 'success', warning: 'warning', critical: 'danger' }
  return map[status] || 'info'
}

const formatDate = (dateStr) => {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  return date.toLocaleString('zh-CN')
}

// 生命周期
onMounted(() => {
  fetchModels()
  fetchStats()
})

onUnmounted(() => {
  inferenceTimeChart?.dispose()
  throughputChart?.dispose()
})
</script>

<style scoped>
.ai-model-management {
  padding: 20px;
  background: #f5f7fa;
  min-height: calc(100vh - 50px);
}

.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
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
  cursor: pointer;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.stat-card.active-card {
  border: 2px solid #409eff;
  background: #ecf5ff;
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
}

.grading-icon { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.segmentation-icon { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.active-icon { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); }
.production-icon { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }

.stat-content {
  flex: 1;
}

.stat-label {
  margin: 0;
  color: #909399;
  font-size: 13px;
}

.stat-value {
  margin: 5px 0 0;
  font-size: 24px;
  font-weight: 600;
  color: #303133;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.filter-tags {
  display: flex;
  align-items: center;
  gap: 10px;
}

.actions {
  display: flex;
  gap: 12px;
}

.table-card {
  margin-bottom: 20px;
}

.model-detail {
  padding: 10px;
}

.performance-charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.performance-chart .chart-container {
  height: 250px;
}

@media (max-width: 1200px) {
  .stats-row {
    grid-template-columns: repeat(2, 1fr);
  }

  .performance-charts {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .stats-row {
    grid-template-columns: 1fr;
  }

  .filter-bar {
    flex-direction: column;
    gap: 12px;
  }

  .filters, .actions {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>
