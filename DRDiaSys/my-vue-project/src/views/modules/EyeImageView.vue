<template>
  <div class="eye-image-container">
    <div class="eye-image-box">
      <div class="header">
        <div class="title-container">
          <h2>眼部影像管理</h2>
          <p class="subtitle">
            上传由医院设备采集的眼底图像，系统将自动分发给后台用于诊断
          </p>
        </div>
        <el-button type="primary" :loading="loading" @click="fetchImages">
          <el-icon><RefreshRight /></el-icon>
          刷新列表
        </el-button>
      </div>

      <el-card class="upload-card" shadow="never">
        <template #header>
          <div class="card-header">
            <span>上传新的影像</span>
            <span class="sub-text">仅支持医院采集的高质量影像</span>
          </div>
        </template>

        <el-form label-width="100px" class="upload-form">
          <el-form-item label="眼别">
            <el-radio-group v-model="uploadForm.eye_side">
              <el-radio label="left">左眼</el-radio>
              <el-radio label="right">右眼</el-radio>
              <el-radio label="both">双眼</el-radio>
              <el-radio label="unknown">未标记</el-radio>
            </el-radio-group>
          </el-form-item>
          <el-form-item label="备注">
            <el-input
              v-model="uploadForm.description"
              type="textarea"
              :rows="2"
              maxlength="200"
              show-word-limit
              placeholder="可记录拍摄设备、注意事项、模糊程度等（选填）"
            />
          </el-form-item>
          <el-form-item label="影像文件">
            <div class="upload-area">
              <el-upload
                ref="uploadRef"
                drag
                action="#"
                :limit="1"
                accept=".jpg,.jpeg,.png,.bmp,.tiff"
                :auto-upload="false"
                :show-file-list="true"
                :file-list="uploadFileList"
                :on-change="handleFileChange"
                :on-remove="handleFileRemove"
              >
                <el-icon class="upload-icon"><UploadFilled /></el-icon>
                <div class="el-upload__text">
                  将文件拖到此处，或<em>点击选择</em>
                </div>
                <div class="el-upload__tip">
                  单次仅支持上传 1 张图片，大小不超过 50MB。
                </div>
              </el-upload>
              <div class="upload-actions">
                <el-button @click="resetUpload" :disabled="uploadLoading">清空</el-button>
                <el-button
                  type="primary"
                  :loading="uploadLoading"
                  @click="submitUpload"
                >
                  <el-icon><UploadFilled /></el-icon>
                  开始上传
                </el-button>
              </div>
            </div>
          </el-form-item>
        </el-form>
      </el-card>

      <el-card class="table-card" shadow="never">
        <template #header>
          <div class="card-header">
            <span>我的影像记录</span>
            <div class="table-toolbar">
              <el-input
                v-model="searchKeyword"
                placeholder="搜索文件名或备注"
                clearable
                @keyup.enter="handleSearch"
              >
                <template #prefix>
                  <el-icon><Search /></el-icon>
                </template>
              </el-input>
              <el-select v-model="eyeFilter" class="filter-select">
                <el-option label="全部眼别" value="all" />
                <el-option label="左眼" value="left" />
                <el-option label="右眼" value="right" />
                <el-option label="双眼" value="both" />
                <el-option label="未标记" value="unknown" />
              </el-select>
            </div>
          </div>
        </template>

        <el-table
          :data="filteredImages"
          style="width: 100%"
          v-loading="loading"
          element-loading-text="加载中..."
          :header-cell-style="{ background: '#fafafa', fontWeight: 600 }"
        >
          <el-table-column prop="original_name" label="文件名" min-width="220" />
          <el-table-column label="大小" width="120">
            <template #default="scope">
              {{ formatFileSize(scope.row.file_size) }}
            </template>
          </el-table-column>
          <el-table-column label="眼别" width="100">
            <template #default="scope">
              <el-tag size="small">{{ formatEyeSide(scope.row.eye_side) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="description" label="备注" min-width="200" show-overflow-tooltip />
          <el-table-column label="上传时间" width="180">
            <template #default="scope">
              {{ formatTime(scope.row.created_at) }}
            </template>
          </el-table-column>
          <el-table-column label="操作" width="220" align="center" fixed="right">
            <template #default="scope">
              <el-button type="primary" link @click="previewImage(scope.row)">
                <el-icon><ViewIcon /></el-icon>
                预览
              </el-button>
              <el-button link @click="downloadImage(scope.row)">
                <el-icon><DownloadIcon /></el-icon>
                下载
              </el-button>
              <el-button type="danger" link @click="handleDelete(scope.row)">
                <el-icon><DeleteIcon /></el-icon>
                删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-dialog
        v-model="previewDialogVisible"
        width="60%"
        title="影像预览"
        :close-on-click-modal="false"
      >
        <div class="patient-preview-body" v-loading="previewLoading">
          <div v-if="previewInfo" class="patient-preview-meta">
            <p><strong>文件：</strong>{{ previewInfo.original_name }}</p>
            <p><strong>眼别：</strong>{{ formatEyeSide(previewInfo.eye_side) }}</p>
            <p><strong>上传时间：</strong>{{ formatTime(previewInfo.created_at) }}</p>
          </div>
          <div class="patient-preview-img" v-if="!previewLoading && previewUrl">
            <img :src="previewUrl" alt="preview" />
          </div>
        </div>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="previewDialogVisible = false">关闭</el-button>
          </span>
        </template>
      </el-dialog>
    </div>
  </div>
</template>

<script>
import api from '../../api'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  UploadFilled,
  RefreshRight,
  Search,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  View as ViewIcon
} from '@element-plus/icons-vue'

export default {
  name: 'EyeImageView',
  components: {
    UploadFilled,
    RefreshRight,
    Search,
    DownloadIcon,
    DeleteIcon,
    ViewIcon
  },
  data() {
    return {
      imageList: [],
      loading: false,
      uploadLoading: false,
      searchKeyword: '',
      eyeFilter: 'all',
      uploadForm: {
        eye_side: 'both',
        description: ''
      },
      uploadFileList: [],
      selectedFile: null,
      previewDialogVisible: false,
      previewUrl: '',
      previewObjectUrl: null,
      previewInfo: null,
      previewLoading: false
    }
  },
  computed: {
    filteredImages() {
      const keyword = this.searchKeyword.trim().toLowerCase()
      return this.imageList.filter(item => {
        const matchKeyword = keyword
          ? (item.original_name || '').toLowerCase().includes(keyword) ||
            (item.description || '').toLowerCase().includes(keyword)
          : true
        const matchEye = this.eyeFilter === 'all' ? true : item.eye_side === this.eyeFilter
        return matchKeyword && matchEye
      })
    }
  },
  mounted() {
    this.fetchImages()
  },
  watch: {
    previewDialogVisible(val) {
      if (!val) {
        this.previewUrl = ''
        this.previewInfo = null
        this.previewLoading = false
        if (this.previewObjectUrl) {
          URL.revokeObjectURL(this.previewObjectUrl)
          this.previewObjectUrl = null
        }
      }
    }
  },
  beforeUnmount() {
    if (this.previewObjectUrl) {
      URL.revokeObjectURL(this.previewObjectUrl)
      this.previewObjectUrl = null
    }
  },
  methods: {
    async fetchImages() {
      try {
        this.loading = true
        const { data } = await api.get('/api/datasets/patient/images/')
        this.imageList = Array.isArray(data) ? data : []
      } catch (error) {
        console.error('获取影像失败', error)
        ElMessage.error(error.response?.data?.message || '获取影像列表失败')
      } finally {
        this.loading = false
      }
    },
    formatFileSize(size) {
      if (!size) return '0 B'
      const units = ['B', 'KB', 'MB', 'GB']
      let index = 0
      let value = size
      while (value >= 1024 && index < units.length - 1) {
        value /= 1024
        index++
      }
      return `${value.toFixed(1)} ${units[index]}`
    },
    formatEyeSide(value) {
      const map = {
        left: '左眼',
        right: '右眼',
        both: '双眼',
        unknown: '未标记'
      }
      return map[value] || '未标记'
    },
    formatTime(value) {
      if (!value) return '-'
      const date = new Date(value)
      if (Number.isNaN(date.getTime())) return '-'
      return date.toLocaleString()
    },
    handleFileChange(file, fileList) {
      this.uploadFileList = fileList.slice(-1)
      this.selectedFile = this.uploadFileList[0]?.raw || null
    },
    handleFileRemove(file, fileList) {
      this.uploadFileList = fileList
      if (!fileList.length) {
        this.selectedFile = null
      }
    },
    resetUpload() {
      this.uploadFileList = []
      this.selectedFile = null
      this.uploadForm.description = ''
      this.uploadForm.eye_side = 'both'
    },
    async submitUpload() {
      if (!this.selectedFile) {
        ElMessage.warning('请先选择要上传的影像文件')
        return
      }
      const formData = new FormData()
      formData.append('image', this.selectedFile)
      formData.append('eye_side', this.uploadForm.eye_side)
      if (this.uploadForm.description) {
        formData.append('description', this.uploadForm.description)
      }
      try {
        this.uploadLoading = true
        await api.post('/api/datasets/patient/images/', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        ElMessage.success('影像上传成功')
        this.resetUpload()
        this.fetchImages()
      } catch (error) {
        console.error('上传失败', error)
        ElMessage.error(error.response?.data?.message || '上传影像失败')
      } finally {
        this.uploadLoading = false
      }
    },
    async handleDelete(row) {
      try {
        await ElMessageBox.confirm('确认删除该影像？删除后将无法恢复。', '提示', {
          type: 'warning',
          confirmButtonText: '删除',
          cancelButtonText: '取消'
        })
        await api.delete(`/api/datasets/patient/images/${row.id}/`)
        ElMessage.success('删除成功')
        this.fetchImages()
      } catch (error) {
        if (error !== 'cancel') {
          console.error('删除影像失败', error)
          ElMessage.error(error.response?.data?.message || '删除失败')
        }
      }
    },
    async downloadImage(row) {
      try {
        const { data } = await api.get(`/api/datasets/patient/images/${row.id}/download/`, {
          responseType: 'blob'
        })
        const objectUrl = URL.createObjectURL(data)
        const link = document.createElement('a')
        link.href = objectUrl
        link.download = row.original_name || 'eye_image.png'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(objectUrl)
      } catch (error) {
        console.error('下载失败', error)
        ElMessage.error(error.response?.data?.message || '下载失败，请稍后再试')
      }
    },
    async previewImage(row) {
      this.previewInfo = row
      this.previewLoading = true
      this.previewDialogVisible = true
      try {
        if (this.previewObjectUrl) {
          URL.revokeObjectURL(this.previewObjectUrl)
          this.previewObjectUrl = null
        }
        const { data } = await api.get(`/api/datasets/patient/images/${row.id}/download/`, {
          params: { mode: 'inline' },
          responseType: 'blob'
        })
        const objectUrl = URL.createObjectURL(data)
        this.previewObjectUrl = objectUrl
        this.previewUrl = objectUrl
      } catch (error) {
        console.error('预览失败', error)
        ElMessage.error(error.response?.data?.message || '预览加载失败，请稍后再试')
        this.previewDialogVisible = false
      } finally {
        this.previewLoading = false
      }
    },
    handleSearch() {
      // 仅用于触发计算属性
    }
  }
}
</script>

<style scoped>
.eye-image-container {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: calc(100vh - 60px);
}

.eye-image-box {
  max-width: 1100px;
  margin: 0 auto;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.title-container h2 {
  margin: 0;
}

.subtitle {
  margin: 6px 0 0;
  color: #909399;
  font-size: 14px;
}

.upload-card,
.table-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

.sub-text {
  color: #a8abb2;
  font-size: 13px;
}

.upload-area {
  width: 100%;
}

.upload-icon {
  font-size: 48px;
  color: #409eff;
}

.upload-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 10px;
}

.table-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
}

.filter-select {
  width: 150px;
}

.patient-preview-body {
  min-height: 300px;
}

.patient-preview-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
  color: #606266;
}

.patient-preview-img {
  text-align: center;
}

.patient-preview-img img {
  max-width: 100%;
  max-height: 500px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
</style>

