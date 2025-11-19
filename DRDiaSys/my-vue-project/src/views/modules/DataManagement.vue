<template>
    <div class="data-management-container">
      <div class="data-management-box">
        <!-- 头部 -->
        <div class="header">
          <div class="title-container">
            <h2>数据管理</h2>
            <p class="subtitle">管理系统中的图像与诊断数据</p>
          </div>
          <el-button type="primary" class="add-button" @click="handleUpload">
            <el-icon><Upload /></el-icon>
            上传数据
          </el-button>
        </div>
  
        <!-- 统计卡片 -->
        <el-row :gutter="20" class="stat-cards">
          <el-col :span="6">
            <el-card shadow="hover" class="stat-card">
              <div class="stat-icon original-icon">
                <el-icon><Picture /></el-icon>
              </div>
              <div class="stat-info">
                <div class="stat-name">原始图像</div>
                <div class="stat-value">{{ dataStats.original || 0 }}</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover" class="stat-card">
              <div class="stat-icon processed-icon">
                <el-icon><PictureFilled /></el-icon>
              </div>
              <div class="stat-info">
                <div class="stat-name">处理后图像</div>
                <div class="stat-value">{{ dataStats.processed || 0 }}</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover" class="stat-card">
              <div class="stat-icon report-icon">
                <el-icon><Document /></el-icon>
              </div>
              <div class="stat-info">
                <div class="stat-name">诊断报告</div>
                <div class="stat-value">{{ dataStats.reports || 0 }}</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover" class="stat-card">
              <div class="stat-icon total-icon">
                <el-icon><Files /></el-icon>
              </div>
              <div class="stat-info">
                <div class="stat-name">数据总量</div>
                <div class="stat-value">{{ dataStats.total || 0 }}</div>
              </div>
            </el-card>
          </el-col>
        </el-row>
  
        <!-- 数据列表 -->
        <el-card class="data-list">
          <template #header>
            <div class="card-header">
              <span>数据列表</span>
              <div class="header-controls">
                <el-select v-model="dataType" placeholder="数据类型" class="filter-select">
                  <el-option label="全部" value="all" />
                  <el-option label="原始图像" value="original" />
                  <el-option label="处理后图像" value="processed" />
                  <el-option label="诊断报告" value="report" />
                </el-select>
                <el-input
                  v-model="searchQuery"
                  placeholder="搜索文件名/标签"
                  class="search-input"
                >
                  <template #prefix>
                    <el-icon><Search /></el-icon>
                  </template>
                </el-input>
              </div>
            </div>
          </template>
  
          <el-table 
            :data="filteredDataList" 
            style="width: 100%"
            :header-cell-style="{
              background: 'rgba(33, 150, 243, 0.1)',
              color: '#333',
              fontWeight: '600'
            }"
          >
            <el-table-column prop="filename" label="文件名" />
            <el-table-column prop="type" label="类型">
              <template #default="scope">
                <el-tag :type="getTypeTag(scope.row.type)" class="type-tag">
                  {{ getTypeLabel(scope.row.type) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="size" label="大小" />
            <el-table-column prop="uploadTime" label="上传时间" />
            <el-table-column prop="owner" label="所有者" />
            <el-table-column label="操作" width="250">
              <template #default="scope">
                <el-button
                  type="primary"
                  link
                  @click="handlePreview(scope.row)"
                >
                  预览
                </el-button>
                <el-button
                  type="success"
                  link
                  @click="handleDownload(scope.row)"
                >
                  下载
                </el-button>
                <el-button
                  type="danger"
                  link
                  @click="handleDelete(scope.row)"
                >
                  删除
                </el-button>
              </template>
            </el-table-column>
          </el-table>
  
          <div class="pagination">
            <el-pagination
              v-model:current-page="currentPage"
              v-model:page-size="pageSize"
              :total="total"
              :page-sizes="[10, 20, 50, 100]"
              layout="total, sizes, prev, pager, next"
              @size-change="handleSizeChange"
              @current-change="handleCurrentChange"
            />
          </div>
        </el-card>
  
        <!-- 上传数据对话框 -->
        <el-dialog
          v-model="uploadDialogVisible"
          title="上传数据"
          width="500px"
          class="upload-dialog"
        >
          <el-form
            ref="uploadForm"
            :model="uploadForm"
            :rules="uploadRules"
            label-width="100px"
            class="upload-form"
          >
            <el-form-item label="数据类型" prop="type">
              <el-select v-model="uploadForm.type" placeholder="请选择数据类型" class="type-select">
                <el-option label="原始图像" value="original" />
                <el-option label="处理后图像" value="processed" />
                <el-option label="诊断报告" value="report" />
              </el-select>
            </el-form-item>
            <el-form-item label="文件" prop="file">
              <el-upload
                class="upload-field"
                action="#"
                :auto-upload="false"
                :limit="5"
                multiple
                :on-change="handleFileChange"
              >
                <template #trigger>
                  <el-button type="primary">选择文件</el-button>
                </template>
                <template #tip>
                  <div class="el-upload__tip">
                    支持上传jpg/png/pdf格式文件，单个文件不超过10MB
                  </div>
                </template>
              </el-upload>
            </el-form-item>
            <el-form-item label="标签" prop="tags">
              <el-select
                v-model="uploadForm.tags"
                multiple
                filterable
                allow-create
                default-first-option
                placeholder="请选择或创建标签"
                class="tag-select"
              >
                <el-option
                  v-for="tag in availableTags"
                  :key="tag"
                  :label="tag"
                  :value="tag"
                />
              </el-select>
            </el-form-item>
            <el-form-item label="描述" prop="description">
              <el-input
                v-model="uploadForm.description"
                type="textarea"
                :rows="3"
                placeholder="请输入数据描述"
              />
            </el-form-item>
          </el-form>
          <template #footer>
            <span class="dialog-footer">
              <el-button @click="uploadDialogVisible = false">取消</el-button>
              <el-button type="primary" @click="submitUpload">
                上传
              </el-button>
            </span>
          </template>
        </el-dialog>
  
        <!-- 预览对话框 -->
        <el-dialog
          v-model="previewDialogVisible"
          title="数据预览"
          width="800px"
          class="preview-dialog"
        >
          <div class="preview-content" v-if="currentPreviewData">
            <div class="preview-image" v-if="currentPreviewData.type !== 'report'">
              <img :src="currentPreviewData.url" :alt="currentPreviewData.filename" />
            </div>
            <div class="preview-report" v-else>
              <div class="report-content" v-html="currentPreviewData.content"></div>
            </div>
            <div class="preview-metadata">
              <h3>文件信息</h3>
              <p><strong>文件名:</strong> {{ currentPreviewData.filename }}</p>
              <p><strong>类型:</strong> {{ getTypeLabel(currentPreviewData.type) }}</p>
              <p><strong>大小:</strong> {{ currentPreviewData.size }}</p>
              <p><strong>上传时间:</strong> {{ currentPreviewData.uploadTime }}</p>
              <p><strong>所有者:</strong> {{ currentPreviewData.owner }}</p>
              <p v-if="currentPreviewData.tags && currentPreviewData.tags.length">
                <strong>标签:</strong>
                <el-tag
                  v-for="tag in currentPreviewData.tags"
                  :key="tag"
                  size="small"
                  class="preview-tag"
                >{{ tag }}</el-tag>
              </p>
              <p v-if="currentPreviewData.description">
                <strong>描述:</strong><br>
                {{ currentPreviewData.description }}
              </p>
            </div>
          </div>
        </el-dialog>
      </div>
  
      <!-- 装饰效果 -->
      <div class="tech-decoration">
        <div class="circle circle-1"></div>
        <div class="circle circle-2"></div>
        <div class="circle circle-3"></div>
      </div>
    </div>
  </template>
  
  <script>
  import { 
    Upload, 
    Picture, 
    PictureFilled, 
    Document, 
    Files, 
    Search 
  } from '@element-plus/icons-vue'
  
  export default {
    name: 'DataManagement',
    components: {
      Upload,
      Picture,
      PictureFilled,
      Document,
      Files,
      Search
    },
    data() {
      return {
        searchQuery: '',
        dataType: 'all',
        currentPage: 1,
        pageSize: 10,
        total: 0,
        dataList: [],
        uploadDialogVisible: false,
        previewDialogVisible: false,
        currentPreviewData: null,
        dataStats: {
          original: 0,
          processed: 0,
          reports: 0,
          total: 0
        },
        uploadForm: {
          type: '',
          file: null,
          tags: [],
          description: ''
        },
        uploadRules: {
          type: [
            { required: true, message: '请选择数据类型', trigger: 'change' }
          ],
          file: [
            { required: true, message: '请选择上传文件', trigger: 'change' }
          ]
        },
        availableTags: ['糖尿病视网膜病变', '健康', '轻度', '中度', '重度', '增殖性']
      }
    },
    computed: {
      filteredDataList() {
        let filtered = this.dataList
        
        // 按类型筛选
        if (this.dataType !== 'all') {
          filtered = filtered.filter(item => item.type === this.dataType)
        }
        
        // 搜索筛选
        if (this.searchQuery) {
          const query = this.searchQuery.toLowerCase()
          filtered = filtered.filter(item => 
            item.filename.toLowerCase().includes(query) || 
            (item.tags && item.tags.some(tag => tag.toLowerCase().includes(query)))
          )
        }
        
        return filtered
      }
    },
    methods: {
      // 获取数据统计
      async fetchDataStatistics() {
        try {
          const token = localStorage.getItem('token')
          const response = await fetch('http://localhost:8000/api/data/statistics/', {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          })
          const data = await response.json()
          if (response.ok) {
            this.dataStats = data.statistics
          } else {
            throw new Error(data.message || '获取数据统计失败')
          }
        } catch (error) {
          this.$message.error(error.message)
        }
      },
      // 获取数据列表
      async fetchDataList() {
        try {
          const token = localStorage.getItem('token')
          const response = await fetch('http://localhost:8000/api/data/', {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          })
          const data = await response.json()
          if (response.ok) {
            this.dataList = data
            this.total = data.length
            this.fetchDataStatistics()
          } else {
            throw new Error(data.message || '获取数据列表失败')
          }
        } catch (error) {
          this.$message.error(error.message)
        }
      },
      // 上传数据
      handleUpload() {
        this.uploadForm = {
          type: '',
          file: null,
          tags: [],
          description: ''
        }
        this.uploadDialogVisible = true
      },
      // 处理文件变更
      handleFileChange(file) {
        this.uploadForm.file = file.raw
      },
      // 提交上传
      submitUpload() {
        this.$refs.uploadForm.validate(async (valid) => {
          if (valid) {
            try {
              // 模拟上传请求
              setTimeout(() => {
                this.$message.success('数据上传成功')
                this.uploadDialogVisible = false
                this.fetchDataList() // 刷新数据列表
              }, 1000)
            } catch (error) {
              this.$message.error(error.message || '上传失败')
            }
          }
        })
      },
      // 预览数据
      handlePreview(data) {
        this.currentPreviewData = data
        this.previewDialogVisible = true
      },
      // 下载数据
      handleDownload(data) {
        this.$message.success(`开始下载: ${data.filename}`)
        // 模拟下载
      },
      // 删除数据
      handleDelete(data) {
        this.$confirm(`确认删除文件 "${data.filename}"?`, '提示', {
          type: 'warning'
        }).then(async () => {
          try {
            // 模拟删除请求
            setTimeout(() => {
              const index = this.dataList.findIndex(item => item.id === data.id)
              if (index !== -1) {
                this.dataList.splice(index, 1)
              }
              this.$message.success('删除成功')
              this.fetchDataStatistics() // 更新数据统计
            }, 500)
          } catch (error) {
            this.$message.error(error.message || '删除失败')
          }
        })
      },
      // 获取类型标签样式
      getTypeTag(type) {
        const typeMap = {
          original: 'info',
          processed: 'success',
          report: 'warning'
        }
        return typeMap[type] || 'info'
      },
      // 获取类型标签文本
      getTypeLabel(type) {
        const labelMap = {
          original: '原始图像',
          processed: '处理后图像',
          report: '诊断报告'
        }
        return labelMap[type] || type
      },
      // 分页处理
      handleSizeChange(val) {
        this.pageSize = val
      },
      handleCurrentChange(val) {
        this.currentPage = val
      }
    },
    created() {
      // 模拟数据
      this.dataList = [
        {
          id: 1,
          filename: 'patient_001_left.jpg',
          type: 'original',
          size: '2.4MB',
          uploadTime: '2023-06-10 09:32',
          owner: '张医生',
          url: 'https://via.placeholder.com/500',
          tags: ['糖尿病视网膜病变', '轻度'],
          description: '左眼原始图像，轻度病变'
        },
        {
          id: 2,
          filename: 'patient_001_right.jpg',
          type: 'original',
          size: '2.3MB',
          uploadTime: '2023-06-10 09:32',
          owner: '张医生',
          url: 'https://via.placeholder.com/500',
          tags: ['健康'],
          description: '右眼原始图像，无明显病变'
        },
        {
          id: 3,
          filename: 'patient_001_left_processed.jpg',
          type: 'processed',
          size: '1.8MB',
          uploadTime: '2023-06-10 09:45',
          owner: '系统',
          url: 'https://via.placeholder.com/500',
          tags: ['糖尿病视网膜病变', '轻度', '增强对比度'],
          description: '左眼增强对比度处理后图像'
        },
        {
          id: 4,
          filename: 'patient_001_report.pdf',
          type: 'report',
          size: '0.5MB',
          uploadTime: '2023-06-10 10:15',
          owner: '系统',
          content: '<div class="report"><h2>患者诊断报告</h2><p>患者ID: 001</p><p>诊断结果: 轻度糖尿病视网膜病变</p><p>建议: 三个月后复查</p></div>',
          tags: ['糖尿病视网膜病变', '轻度', '报告'],
          description: '患者001的诊断报告'
        },
        {
          id: 5,
          filename: 'patient_002_left.jpg',
          type: 'original',
          size: '2.5MB',
          uploadTime: '2023-06-12 14:22',
          owner: '李医生',
          url: 'https://via.placeholder.com/500',
          tags: ['糖尿病视网膜病变', '中度'],
          description: '左眼原始图像，中度病变'
        }
      ]
      this.total = this.dataList.length
      
      // 计算统计数据
      this.dataStats = {
        original: this.dataList.filter(item => item.type === 'original').length,
        processed: this.dataList.filter(item => item.type === 'processed').length,
        reports: this.dataList.filter(item => item.type === 'report').length,
        total: this.dataList.length
      }
    }
  }
  </script>
  
  <style scoped>
  .data-management-container {
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #1a1a1a 0%, #2c3e50 100%);
    position: relative;
    overflow: hidden;
    padding: 20px;
  }
  
  .data-management-box {
    width: 1500px;
    padding: 150px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    position: relative;
    z-index: 1;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  
  .title-container {
    color: #fff;
  }
  
  .title-container h2 {
    font-size: 24px;
    margin: 0;
    font-weight: 600;
  }
  
  .subtitle {
    color: #a8b2d1;
    font-size: 14px;
    margin-top: 8px;
  }
  
  .add-button {
    background: linear-gradient(45deg, #2196F3, #00BCD4);
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 500;
    transition: all 0.3s ease;
  }
  
  .add-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
  }
  
  .data-list {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 16px;
    border: none;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  }
  
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 20px;
  }
  
  .card-header span {
    font-size: 18px;
    font-weight: 600;
    color: #333;
  }
  
  .header-controls {
    display: flex;
    gap: 10px;
  }
  
  .filter-select {
    width: 120px;
  }
  
  .search-input {
    width: 250px;
  }
  
  .type-tag {
    border-radius: 4px;
    padding: 4px 8px;
  }
  
  .pagination {
    margin-top: 20px;
    display: flex;
    justify-content: flex-end;
    padding: 0 20px;
  }
  
  /* 上传对话框 */
  .upload-field {
    width: 100%;
  }
  
  .tag-select {
    width: 100%;
  }
  
  /* 预览对话框 */
  .preview-content {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .preview-image img {
    max-width: 100%;
    max-height: 400px;
    border-radius: 8px;
    display: block;
    margin: 0 auto;
  }
  
  .preview-metadata {
    background: #f5f7fa;
    padding: 15px;
    border-radius: 8px;
  }
  
  .preview-tag {
    margin-right: 5px;
    margin-bottom: 5px;
  }
  
  .report-content {
    background: white;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #ebeef5;
    max-height: 400px;
    overflow-y: auto;
  }
  
  /* 统计卡片样式 */
  .stat-cards {
    margin-bottom: 20px;
  }
  
  .stat-card {
    height: 100px;
    display: flex;
    align-items: center;
    padding: 20px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.9);
    transition: all 0.3s;
  }
  
  .stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
  }
  
  .stat-icon {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-right: 15px;
  }
  
  .stat-icon .el-icon {
    font-size: 24px;
    color: white;
  }
  
  .original-icon {
    background: linear-gradient(45deg, #3f51b5, #7986cb);
  }
  
  .processed-icon {
    background: linear-gradient(45deg, #4caf50, #81c784);
  }
  
  .report-icon {
    background: linear-gradient(45deg, #ff9800, #ffb74d);
  }
  
  .total-icon {
    background: linear-gradient(45deg, #673ab7, #9575cd);
  }
  
  .stat-info {
    flex: 1;
  }
  
  .stat-name {
    font-size: 14px;
    color: #666;
    margin-bottom: 5px;
  }
  
  .stat-value {
    font-size: 24px;
    font-weight: 600;
    color: #333;
  }
  
  /* 装饰效果 */
  .tech-decoration {
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    pointer-events: none;
  }
  
  .circle {
    position: absolute;
    border-radius: 50%;
    background: linear-gradient(45deg, rgba(33, 150, 243, 0.1), rgba(0, 188, 212, 0.1));
    animation: float 6s ease-in-out infinite;
  }
  
  .circle-1 {
    width: 300px;
    height: 300px;
    top: -150px;
    right: -150px;
    animation-delay: 0s;
  }
  
  .circle-2 {
    width: 200px;
    height: 200px;
    bottom: -100px;
    left: -100px;
    animation-delay: 2s;
  }
  
  .circle-3 {
    width: 150px;
    height: 150px;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    animation-delay: 4s;
  }
  
  @keyframes float {
    0% {
      transform: translateY(0) rotate(0deg);
    }
    50% {
      transform: translateY(-20px) rotate(180deg);
    }
    100% {
      transform: translateY(0) rotate(360deg);
    }
  }
  
  /* 弹窗样式 */
  :deep(.upload-dialog), :deep(.preview-dialog) {
    border-radius: 16px;
    overflow: hidden;
  }
  
  :deep(.upload-dialog .el-dialog__header), :deep(.preview-dialog .el-dialog__header) {
    background: linear-gradient(45deg, #2196F3, #00BCD4);
    margin: 0;
    padding: 20px;
  }
  
  :deep(.upload-dialog .el-dialog__title), :deep(.preview-dialog .el-dialog__title) {
    color: #fff;
    font-weight: 600;
  }
  
  :deep(.upload-dialog .el-dialog__body), :deep(.preview-dialog .el-dialog__body) {
    padding: 30px 20px;
  }
  
  :deep(.el-dialog__footer) {
    padding: 20px;
    border-top: 1px solid #eee;
  }
  </style>