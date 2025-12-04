<template>
  <div class="report-view-container">
    <div v-if="hasAccess">
      <el-card shadow="never" class="report-card">
      <template #header>
        <div class="card-header">
          <h3>诊断报告</h3>
          <div class="header-actions">
            <el-input
              v-model="searchQuery"
              placeholder="搜索报告编号或患者姓名"
              clearable
              class="search-input"
              @input="handleSearch"
            >
              <template #prefix>
                <el-icon><Search /></el-icon>
              </template>
            </el-input>
            <el-select v-model="statusFilter" class="status-filter" @change="handleFilter">
              <el-option label="全部" value="" />
              <el-option label="草稿" value="draft" />
              <el-option label="待复核" value="pending_review" />
              <el-option label="已复核" value="reviewed" />
              <el-option label="已确认" value="finalized" />
            </el-select>
            <el-button type="primary" :loading="loading" @click="fetchReports">
              <el-icon><RefreshRight /></el-icon>
              刷新
            </el-button>
          </div>
        </div>
      </template>

      <el-table
        :data="filteredReports"
        v-loading="loading"
        empty-text="暂无报告"
        :header-cell-style="{ background: '#fafafa', fontWeight: 600 }"
      >
        <el-table-column prop="report_number" label="报告编号" width="200" />
        <el-table-column label="患者" width="150">
          <template #default="scope">
            {{ scope.row.patient_name || '未知' }}
          </template>
        </el-table-column>
        <el-table-column label="影像文件" min-width="200" show-overflow-tooltip>
          <template #default="scope">
            {{ scope.row.patient_image_info?.original_name || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="状态" width="120">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">
              {{ getStatusLabel(scope.row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="AI诊断摘要" min-width="200" show-overflow-tooltip>
          <template #default="scope">
            {{ scope.row.ai_summary || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="复核医生" width="120">
          <template #default="scope">
            {{ scope.row.reviewed_by_name || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="创建时间" width="180">
          <template #default="scope">
            {{ formatTime(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" align="center" fixed="right">
          <template #default="scope">
            <el-button
              v-if="scope.row.pdf_path"
              type="primary"
              link
              size="small"
              @click="previewPdf(scope.row)"
            >
              <el-icon><View /></el-icon>
              预览PDF
            </el-button>
            <el-button
              v-if="scope.row.pdf_path"
              type="success"
              link
              size="small"
              @click="downloadReport(scope.row)"
            >
              <el-icon><Download /></el-icon>
              下载PDF
            </el-button>
            <el-button
              v-if="isDoctor && scope.row.status !== 'finalized'"
              type="warning"
              link
              size="small"
              @click="reviewReport(scope.row)"
            >
              <el-icon><Edit /></el-icon>
              复核
            </el-button>
            <el-button
              v-if="isAdmin"
              type="danger"
              link
              size="small"
              @click="confirmDeleteReport(scope.row)"
            >
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

      <!-- PDF 预览对话框 -->
      <el-dialog
        v-model="previewDialogVisible"
        title="PDF报告预览"
        width="80%"
        :close-on-click-modal="false"
        @close="handlePreviewClose"
      >
        <div v-if="previewPdfUrl" class="pdf-preview-wrapper">
          <iframe
            :src="previewPdfUrl"
            frameborder="0"
            style="width: 100%; height: 80vh;"
          />
        </div>
      </el-dialog>

      <!-- 医生复核对话框 -->
      <el-dialog
        v-model="reviewDialogVisible"
        title="医生复核"
        width="600px"
        :close-on-click-modal="false"
      >
        <el-form :model="reviewForm" label-width="100px">
          <el-form-item label="医生结论">
            <el-input
              v-model="reviewForm.doctor_conclusion"
              type="textarea"
              :rows="3"
              placeholder="请输入医生结论"
            />
          </el-form-item>
          <el-form-item label="医生备注">
            <el-input
              v-model="reviewForm.doctor_notes"
              type="textarea"
              :rows="4"
              placeholder="请输入医生备注"
            />
          </el-form-item>
          <el-form-item label="报告状态">
            <el-radio-group v-model="reviewForm.status">
              <el-radio label="reviewed">已复核</el-radio>
              <el-radio label="finalized">已确认</el-radio>
            </el-radio-group>
          </el-form-item>
        </el-form>
        <template #footer>
          <el-button @click="reviewDialogVisible = false">取消</el-button>
          <el-button type="primary" :loading="reviewLoading" @click="submitReview">
            提交复核
          </el-button>
        </template>
      </el-dialog>
    </div>

    <el-result
      v-else
      icon="warning"
      title="当前账号无权限查看AI诊断报告"
      sub-title="请使用管理员或医生账号登录后再试"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, RefreshRight, View, Download, Edit, Delete } from '@element-plus/icons-vue'
import api from '../../api'

const loading = ref(false)
const reports = ref([])
const searchQuery = ref('')
const statusFilter = ref('')
const currentReport = ref(null)
const previewDialogVisible = ref(false)
const previewPdfUrl = ref('')
const reviewDialogVisible = ref(false)
const reviewLoading = ref(false)
const reviewForm = ref({
  doctor_conclusion: '',
  doctor_notes: '',
  status: 'reviewed'
})

const userRole = ref(localStorage.getItem('userRole') || 'admin')
const hasAccess = computed(() => ['doctor', 'admin', 'patient'].includes(userRole.value))
const isDoctor = computed(() => userRole.value === 'doctor')
const isAdmin = computed(() => userRole.value === 'admin')

const filteredReports = computed(() => {
  let result = reports.value

  if (searchQuery.value) {
    const keyword = searchQuery.value.toLowerCase()
    result = result.filter(report =>
      report.report_number.toLowerCase().includes(keyword) ||
      (report.patient_name || '').toLowerCase().includes(keyword)
    )
  }

  if (statusFilter.value) {
    result = result.filter(report => report.status === statusFilter.value)
  }

  return result
})

const fetchReports = async () => {
  if (!hasAccess.value) {
    reports.value = []
    return
  }

  try {
    loading.value = true
    const params = {}
    if (statusFilter.value) {
      params.status = statusFilter.value
    }
    const { data } = await api.get('/api/diagnosis/reports/', { params })
    reports.value = data || []
  } catch (error) {
    console.error('获取报告列表失败', error)
    ElMessage.error(error.response?.data?.message || '获取报告列表失败')
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  // 搜索已通过computed处理
}

const handleFilter = () => {
  fetchReports()
}

const downloadReport = async (report) => {
  try {
    const response = await api.get(`/api/diagnosis/reports/${report.id}/download/`, {
      responseType: 'blob'
    })
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', `${report.report_number}.pdf`)
    document.body.appendChild(link)
    link.click()
    link.remove()
    ElMessage.success('下载成功')
  } catch (error) {
    console.error('下载报告失败', error)
    ElMessage.error(error.response?.data?.message || '下载报告失败')
  }
}

const previewPdf = async (report) => {
  try {
    const response = await api.get(`/api/diagnosis/reports/${report.id}/download/`, {
      responseType: 'blob'
    })
    const file = new Blob([response.data], { type: 'application/pdf' })
    if (previewPdfUrl.value) {
      window.URL.revokeObjectURL(previewPdfUrl.value)
    }
    const url = window.URL.createObjectURL(file)
    previewPdfUrl.value = url
    previewDialogVisible.value = true
  } catch (error) {
    console.error('预览报告失败', error)
    ElMessage.error(error.response?.data?.message || '预览报告失败')
  }
}

const handlePreviewClose = () => {
  if (previewPdfUrl.value) {
    window.URL.revokeObjectURL(previewPdfUrl.value)
    previewPdfUrl.value = ''
  }
}

const confirmDeleteReport = (report) => {
  ElMessageBox.confirm('确定删除该诊断报告？删除后不可恢复。', '提示', {
    type: 'warning',
    confirmButtonText: '删除',
    cancelButtonText: '取消'
  })
    .then(() => deleteReport(report))
    .catch(() => {})
}

const deleteReport = async (report) => {
  try {
    await api.delete(`/api/diagnosis/reports/${report.id}/delete/`)
    ElMessage.success('删除成功')
    fetchReports()
  } catch (error) {
    console.error('删除报告失败', error)
    ElMessage.error(error.response?.data?.message || '删除报告失败')
  }
}

const reviewReport = (report) => {
  currentReport.value = report
  reviewForm.value = {
    doctor_conclusion: report.doctor_conclusion || '',
    doctor_notes: report.doctor_notes || '',
    status: report.status === 'finalized' ? 'finalized' : 'reviewed'
  }
  reviewDialogVisible.value = true
}

const submitReview = async () => {
  if (!currentReport.value) return

  try {
    reviewLoading.value = true
    await api.post(`/api/diagnosis/reports/${currentReport.value.id}/review/`, reviewForm.value)
    ElMessage.success('复核成功')
    reviewDialogVisible.value = false
    fetchReports()
  } catch (error) {
    console.error('提交复核失败', error)
    ElMessage.error(error.response?.data?.message || '提交复核失败')
  } finally {
    reviewLoading.value = false
  }
}

const getStatusLabel = (status) => {
  const map = {
    draft: '草稿',
    pending_review: '待复核',
    reviewed: '已复核',
    finalized: '已确认'
  }
  return map[status] || status
}

const getStatusType = (status) => {
  const map = {
    draft: 'info',
    pending_review: 'warning',
    reviewed: 'success',
    finalized: 'success'
  }
  return map[status] || 'info'
}

const formatTime = (time) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

onMounted(() => {
  if (hasAccess.value) {
    fetchReports()
  }
})
</script>

<style scoped>
.report-view-container {
  padding: 20px;
}

.report-card {
  min-height: calc(100vh - 200px);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.header-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.search-input {
  width: 250px;
}

.status-filter {
  width: 150px;
}

.report-detail {
  padding: 20px 0;
}

.ai-result,
.doctor-review {
  margin-top: 20px;
}

.lesion-stats {
  margin-top: 15px;
}

.result-image {
  margin-top: 20px;
}

.result-image h4 {
  margin-bottom: 10px;
  font-size: 14px;
  font-weight: 600;
}

.pdf-preview-wrapper {
  width: 100%;
  height: 100%;
}

:deep(.el-descriptions__label) {
  font-weight: 600;
}
</style>

