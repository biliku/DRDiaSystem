<template>
  <div class="clinical-image-container">
    <div class="clinical-image-box">
      <!-- 头部 -->
      <div class="header">
        <div class="title-container">
          <h2>临床影像管理</h2>
          <p class="subtitle">管理患者上传的临床影像数据</p>
        </div>
      </div>

      <el-card class="patient-upload-card" shadow="never">
        <div class="patient-upload-stats">
          <div class="stat-item">
            <p>影像数量</p>
            <h3>{{ adminPatientStats.total }}</h3>
          </div>
          <div class="stat-item">
            <p>累计容量</p>
            <h3>{{ adminPatientStats.totalSizeLabel }}</h3>
          </div>
          <div class="stat-item">
            <p>最近上传</p>
            <h3>{{ adminPatientStats.latestTime }}</h3>
          </div>
        </div>

        <div class="patient-toolbar">
          <el-input
            v-model="adminPatientSearch"
            placeholder="搜索患者姓名或文件名"
            clearable
            class="patient-search"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
          <el-select v-model="adminEyeFilter" class="patient-eye-filter">
            <el-option
              v-for="item in patientEyeOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-button type="primary" :loading="adminPatientLoading" @click="handleAdminRefresh">
            <el-icon><RefreshRight /></el-icon>
            刷新
          </el-button>
        </div>

        <div v-loading="adminPatientLoading" class="patient-groups-container">
          <el-collapse v-if="adminFilteredPatientGroups.length > 0" v-model="activePatientGroups" accordion>
            <el-collapse-item
              v-for="patient in adminFilteredPatientGroups"
              :key="patient.patient_id"
              :name="patient.patient_id"
              class="patient-group-item"
            >
              <template #title>
                <div class="patient-group-header">
                  <div class="patient-info">
                    <el-icon class="folder-icon"><Folder /></el-icon>
                    <span class="patient-name">{{ patient.patient_name }}</span>
                    <el-tag size="small" type="info" class="folder-tag">{{ patient.patient_folder }}</el-tag>
                  </div>
                  <div class="patient-stats">
                    <span class="stat-text">{{ patient.total_images }} 张影像</span>
                    <span class="stat-text">{{ formatPatientFileSize(patient.total_size) }}</span>
                    <span class="stat-text" v-if="patient.latest_upload">
                      最近: {{ formatPatientTime(patient.latest_upload) }}
                    </span>
                  </div>
                </div>
              </template>
              <div class="patient-images-table">
                <el-table
                  :data="getFilteredPatientImages(patient.images)"
                  size="default"
                  :header-cell-style="{ background: '#f5f7fa', fontWeight: 600 }"
                  empty-text="该患者暂无影像"
                >
                  <el-table-column prop="original_name" label="文件名" min-width="220" show-overflow-tooltip />
                  <el-table-column label="眼别" width="100">
                    <template #default="scope">
                      <el-tag size="small">{{ formatEyeLabel(scope.row.eye_side) }}</el-tag>
                    </template>
                  </el-table-column>
                  <el-table-column prop="description" label="备注" min-width="200" show-overflow-tooltip />
                  <el-table-column label="大小" width="100">
                    <template #default="scope">
                      {{ formatPatientFileSize(scope.row.file_size) }}
                    </template>
                  </el-table-column>
                  <el-table-column label="上传时间" width="180">
                    <template #default="scope">
                      {{ formatPatientTime(scope.row.created_at) }}
                    </template>
                  </el-table-column>
                  <el-table-column label="操作" width="100" align="center">
                    <template #default="scope">
                      <el-button type="primary" link size="small" @click="handleAdminPreview(scope.row)">
                        <el-icon><View /></el-icon>
                        预览
                      </el-button>
                    </template>
                  </el-table-column>
                </el-table>
              </div>
            </el-collapse-item>
          </el-collapse>
          <el-empty v-else description="暂无患者上传记录" />
        </div>
      </el-card>
    </div>

    <!-- 预览对话框 -->
    <el-dialog
      v-model="patientPreviewDialogVisible"
      width="60%"
      title="影像预览"
      :close-on-click-modal="false"
    >
      <div class="patient-preview-body" v-loading="patientPreviewLoading">
        <div v-if="patientPreviewInfo" class="patient-preview-meta">
          <p><strong>患者：</strong>{{ patientPreviewInfo.owner_name || '未知' }}</p>
          <p><strong>文件名：</strong>{{ patientPreviewInfo.original_name }}</p>
          <p><strong>眼别：</strong>{{ formatEyeLabel(patientPreviewInfo.eye_side) }}</p>
          <p><strong>上传时间：</strong>{{ formatPatientTime(patientPreviewInfo.created_at) }}</p>
          <p v-if="patientPreviewInfo.description"><strong>备注：</strong>{{ patientPreviewInfo.description }}</p>
        </div>
        <div class="patient-preview-image">
          <img v-if="patientPreviewUrl" :src="patientPreviewUrl" alt="预览图像" />
          <div v-else class="no-image">无法加载图像</div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Search, RefreshRight, Folder, View } from '@element-plus/icons-vue'
import api from '../../api'

// 状态变量
const adminPatientGroups = ref([])
const adminPatientLoading = ref(false)
const adminPatientSearch = ref('')
const adminEyeFilter = ref('all')
const patientImagesLoaded = ref(false)
const patientPreviewDialogVisible = ref(false)
const patientPreviewUrl = ref('')
const patientPreviewObjectUrl = ref(null)
const patientPreviewInfo = ref(null)
const patientPreviewLoading = ref(false)
const activePatientGroups = ref([])

const patientEyeOptions = [
  { label: '全部', value: 'all' },
  { label: '左眼', value: 'left' },
  { label: '右眼', value: 'right' },
  { label: '双眼', value: 'both' },
  { label: '未标记', value: 'unknown' }
]

// 工具函数
const formatPatientFileSize = (size) => {
  if (!size) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let index = 0
  let value = size
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index++
  }
  return `${value.toFixed(1)} ${units[index]}`
}

const formatPatientTime = (value) => {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '-'
  return date.toLocaleString()
}

const formatEyeLabel = (value) => {
  const map = {
    left: '左眼',
    right: '右眼',
    both: '双眼',
    unknown: '未标记'
  }
  return map[value] || '未标记'
}

// API 函数
const fetchAdminPatientImages = async () => {
  try {
    adminPatientLoading.value = true
    const { data } = await api.get('/api/datasets/admin/patient-images/')
    adminPatientGroups.value = Array.isArray(data) ? data : []
    patientImagesLoaded.value = true
  } catch (error) {
    console.error('获取患者影像失败', error)
    ElMessage.error(error.response?.data?.message || '获取患者影像失败')
  } finally {
    adminPatientLoading.value = false
  }
}

// 过滤患者组
const adminFilteredPatientGroups = computed(() => {
  const keyword = adminPatientSearch.value.trim().toLowerCase()
  if (!keyword && adminEyeFilter.value === 'all') {
    return adminPatientGroups.value
  }

  return adminPatientGroups.value.filter(patient => {
    const matchPatient = keyword
      ? (patient.patient_name || '').toLowerCase().includes(keyword) ||
        (patient.patient_folder || '').toLowerCase().includes(keyword)
      : true

    if (matchPatient) {
      const hasMatchingImages = getFilteredPatientImages(patient.images).length > 0
      return hasMatchingImages
    }

    return false
  })
})

// 过滤单个患者组内的影像
const getFilteredPatientImages = (images) => {
  const keyword = adminPatientSearch.value.trim().toLowerCase()
  const eyeFilter = adminEyeFilter.value

  return images.filter(image => {
    const matchKeyword = keyword
      ? (image.original_name || '').toLowerCase().includes(keyword) ||
        (image.description || '').toLowerCase().includes(keyword)
      : true
    const matchEye = eyeFilter === 'all' ? true : image.eye_side === eyeFilter
    return matchKeyword && matchEye
  })
}

// 统计数据
const adminPatientStats = computed(() => {
  let total = 0
  let totalSize = 0
  let latestTime = null

  adminPatientGroups.value.forEach(patient => {
    total += patient.total_images || 0
    totalSize += patient.total_size || 0
    if (patient.latest_upload) {
      const uploadTime = new Date(patient.latest_upload)
      if (!latestTime || uploadTime > latestTime) {
        latestTime = uploadTime
      }
    }
  })

  return {
    total,
    totalSizeLabel: formatPatientFileSize(totalSize),
    latestTime: latestTime ? formatPatientTime(latestTime.toISOString()) : '-'
  }
})

// 事件处理
const handleAdminRefresh = () => {
  patientImagesLoaded.value = false
  fetchAdminPatientImages()
}

const loadPreviewBlob = async (imageId) => {
  const { data } = await api.get(`/api/datasets/patient/images/${imageId}/download/`, {
    params: { mode: 'inline' },
    responseType: 'blob'
  })
  return URL.createObjectURL(data)
}

const handleAdminPreview = async (row) => {
  patientPreviewInfo.value = row
  patientPreviewLoading.value = true
  patientPreviewDialogVisible.value = true
  try {
    if (patientPreviewObjectUrl.value) {
      URL.revokeObjectURL(patientPreviewObjectUrl.value)
      patientPreviewObjectUrl.value = null
    }
    const objectUrl = await loadPreviewBlob(row.id)
    patientPreviewObjectUrl.value = objectUrl
    patientPreviewUrl.value = objectUrl
  } catch (error) {
    console.error('预览加载失败', error)
    ElMessage.error(error.response?.data?.message || '预览加载失败，请稍后再试')
    patientPreviewDialogVisible.value = false
  } finally {
    patientPreviewLoading.value = false
  }
}

// 生命周期
onMounted(() => {
  fetchAdminPatientImages()
})
</script>

<style scoped>
.clinical-image-container {
  padding: 20px;
}

.clinical-image-box {
  max-width: 1400px;
  margin: 0 auto;
}

.header {
  margin-bottom: 20px;
}

.title-container h2 {
  margin: 0;
  font-size: 24px;
  color: #303133;
}

.subtitle {
  margin: 5px 0 0;
  color: #909399;
  font-size: 14px;
}

.patient-upload-card {
  margin-top: 20px;
}

.patient-upload-stats {
  display: flex;
  gap: 40px;
  margin-bottom: 20px;
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 8px;
  color: white;
}

.stat-item {
  text-align: center;
}

.stat-item p {
  margin: 0;
  font-size: 14px;
  opacity: 0.9;
}

.stat-item h3 {
  margin: 5px 0 0;
  font-size: 28px;
  font-weight: 600;
}

.patient-toolbar {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.patient-search {
  width: 300px;
}

.patient-eye-filter {
  width: 120px;
}

.patient-group-item {
  margin-bottom: 10px;
}

.patient-group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 5px 0;
}

.patient-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.folder-icon {
  font-size: 20px;
  color: #409eff;
}

.patient-name {
  font-weight: 600;
  font-size: 15px;
}

.folder-tag {
  margin-left: 5px;
}

.patient-stats {
  display: flex;
  gap: 20px;
  color: #909399;
  font-size: 13px;
}

.patient-images-table {
  margin-top: 10px;
}

.patient-preview-body {
  min-height: 400px;
}

.patient-preview-meta {
  margin-bottom: 20px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 6px;
}

.patient-preview-meta p {
  margin: 5px 0;
  color: #606266;
}

.patient-preview-image {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
  background: #f5f7fa;
  border-radius: 6px;
}

.patient-preview-image img {
  max-width: 100%;
  max-height: 500px;
  object-fit: contain;
}

.no-image {
  color: #909399;
  font-size: 14px;
}
</style>
