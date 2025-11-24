<template>
  <div class="image-processing-container">
    <div class="image-processing-box">
      <!-- 头部 -->
      <div class="header">
        <div class="title-container">
          <h2>图像预处理</h2>
          <p class="subtitle">管理数据集和图像预处理任务</p>
        </div>
      </div>

      <!-- 功能区切换 -->
      <el-tabs v-model="activeTab" class="function-tabs">
        <!-- 数据集管理 -->
        <el-tab-pane label="数据集管理" name="dataset">
          <el-row :gutter="20">
            <!-- Left Sidebar -->
            <el-col :span="5">
              <el-card class="sidebar-filters" shadow="never">
                <div class="filter-group">
                  <h4 class="filter-title">类型</h4>
                  <el-radio-group v-model="selectedType" @change="applyFiltersAndResetPage">
                    <el-radio label="all" size="large">全部</el-radio>
                    <el-radio label="public" size="large">公开</el-radio>
                    <el-radio label="clinical" size="large">临床</el-radio>
                  </el-radio-group>
                </div>
                <el-divider />
                <div class="filter-group">
                  <h4 class="filter-title">状态</h4>
                  <el-radio-group v-model="selectedStatus" @change="applyFiltersAndResetPage">
                    <el-radio label="all" size="large">全部</el-radio>
                    <el-radio label="unprocessed" size="large">原始</el-radio>
                    <el-radio label="processed" size="large">已处理</el-radio>
                  </el-radio-group>
                </div>
              </el-card>
            </el-col>

            <!-- Right Content Area -->
            <el-col :span="19">
              <div class="main-content-header">
                <div class="search-add-container">
                  <el-input
                    v-model="searchQuery"
                    placeholder="搜索数据集名称或描述"
                    class="search-input"
                    clearable
                    @keyup.enter="performSearch"
                    @clear="clearSearchAndFilter"
                  >
                    <template #prefix>
                      <el-icon><Search /></el-icon>
                    </template>
                  </el-input>
                  <el-button @click="performSearch" style="margin-left: 10px;">搜索</el-button>
                </div>
                <el-button type="primary" class="add-button-main" @click="handleAddDataset()">
                  <el-icon><Plus /></el-icon>
                  添加新数据集
                </el-button>
              </div>
               <div v-if="searchQuery && isSearching" class="search-active-tag">
                  <el-tag type="info" closable @close="clearSearchAndFilter">
                    当前搜索: "{{ searchQuery }}" ({{ totalFilteredDatasets }} 条结果)
                  </el-tag>
              </div>

              <el-card class="dataset-table-card" shadow="never">
                <el-table
                  :data="paginatedFilteredDatasetList"
                  style="width: 100%"
                  :header-cell-style="{
                    background: '#fafafa',
                    color: '#333',
                    fontWeight: '600'
                  }"
                  v-loading="tableLoading"
                  element-loading-text="加载中..."
                >
                  <el-table-column type="selection" width="50" align="center" />
                  <el-table-column prop="name" label="名称" min-width="180" sortable />
                  <el-table-column prop="type" label="类型" width="100" align="center">
                    <template #default="scope">
                      {{ getDatasetTypeLabel(scope.row.type) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="status" label="状态" width="160" align="center">
                    <template #default="scope">
                      <el-tag :type="getStatusType(scope.row.status)" size="small">
                        {{ getTableStatusLabel(scope.row.status, scope.row.version) }}
                      </el-tag>
                    </template>
                  </el-table-column>
                  <el-table-column prop="imageCount" label="数量" width="100" align="center" sortable />
                  <el-table-column prop="description" label="描述" min-width="200" show-overflow-tooltip />
                  <el-table-column prop="createTime" label="创建时间" width="170" align="center" sortable />
                  <el-table-column label="操作" width="220" align="center" fixed="right">
                    <template #default="scope">
                      <el-tooltip content="查看" placement="top">
                        <el-button :icon="ViewIcon" type="primary" link @click="handleView(scope.row)" />
                      </el-tooltip>
                      <el-tooltip content="以此为模板创建" placement="top">
                        <el-button :icon="DocumentCopyIcon" type="primary" link @click="handleAddDataset(scope.row)" />
                      </el-tooltip>
                      <el-tooltip :content="scope.row.status === 'unprocessed' ? '提交预处理' : (scope.row.status === 'processed' ? '重新处理/配置' : '处理中')" placement="top">
                        <el-button
                          :icon="SettingIcon"
                          type="primary"
                          link
                          :disabled="scope.row.status === 'processing'"
                          @click="handleSubmitToProcess(scope.row)"
                        />
                      </el-tooltip>
                      <el-tooltip content="编辑/标签" placement="top">
                        <el-button :icon="PriceTagIcon" type="primary" link @click="handleEditTags(scope.row)" />
                      </el-tooltip>
                      <el-tooltip content="删除" placement="top">
                        <el-button :icon="DeleteIcon" type="danger" link @click="handleDelete(scope.row)" />
                      </el-tooltip>
                    </template>
                  </el-table-column>
                </el-table>

                <div class="pagination-container">
                  <el-pagination
                    v-model:current-page="currentPage"
                    v-model:page-size="pageSize"
                    :total="totalFilteredDatasets"
                    :page-sizes="[10, 20, 50, 100]"
                    layout="total, sizes, prev, pager, next, jumper"
                    @size-change="handleSizeChange"
                    @current-change="handleCurrentChange"
                  />
                </div>
              </el-card>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 预处理任务 -->
        <el-tab-pane label="预处理任务" name="process">
          <div class="process-container">
            <el-card class="process-method">
              <template #header>
                <div class="card-header">
                  <span>处理方式</span>
                </div>
              </template>
              <el-form :model="processForm" label-position="top">
                <el-form-item label="选择处理方式">
                  <el-radio-group v-model="processForm.method">
                    <el-radio label="normalize">图像归一化</el-radio>
                    <el-radio label="enhance">图像增强</el-radio>
                    <el-radio label="segment">血管分割</el-radio>
                    <el-radio label="custom">自定义处理</el-radio>
                  </el-radio-group>
                </el-form-item>
                <el-form-item v-if="processForm.method === 'custom'" label="自定义处理参数">
                  <el-input
                    v-model="processForm.customParams"
                    type="textarea"
                    :rows="4"
                    placeholder="请输入自定义处理参数，每行一个参数"
                  />
                </el-form-item>
              </el-form>
            </el-card>

            <el-card class="task-list">
              <template #header>
                <div class="card-header">
                  <span>任务列表</span>
                  <el-button type="primary" @click="startProcess" :disabled="!selectedDatasets.length">
                    开始处理
                  </el-button>
                </div>
              </template>

              <el-table
                :data="selectedDatasets"
                style="width: 100%"
                :header-cell-style="{
                  background: 'rgba(33, 150, 243, 0.1)',
                  color: '#333',
                  fontWeight: '600'
                }"
              >
                <el-table-column type="selection" width="55" />
                <el-table-column prop="name" label="数据集名称" min-width="200" />
                <el-table-column prop="imageCount" label="图像数量" width="120" align="center" />
                <el-table-column prop="createTime" label="创建时间" width="180" align="center" />
                <el-table-column label="操作" width="120" align="center">
                  <template #default="scope">
                    <el-button
                      type="danger"
                      link
                      @click="removeFromTask(scope.row)"
                    >
                      移除
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>
          </div>
        </el-tab-pane>
      </el-tabs>

      <!-- 添加/编辑 数据集对话框 -->
      <el-dialog
        v-model="datasetDialogVisible"
        :title="editingDatasetId ? '编辑数据集信息' : '添加新数据集'"
        width="600px"
        class="dataset-dialog"
        :close-on-click-modal="false"
        :show-close="true"
        @close="resetDatasetForm"
      >
        <el-form
          ref="datasetFormRef"
          :model="datasetForm"
          :rules="rules"
          label-position="top"
          class="dataset-form"
          @submit.prevent
        >
          <el-form-item label="数据集名称" prop="name">
            <el-input v-model="datasetForm.name" placeholder="请输入数据集名称" />
          </el-form-item>

          <el-form-item label="数据集类型" prop="type">
            <el-select v-model="datasetForm.type" placeholder="请选择数据集类型" style="width: 100%;">
              <el-option
                v-for="item in datasetTypeOptions"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="描述信息" prop="description">
            <el-input
              v-model="datasetForm.description"
              type="textarea"
              :rows="3"
              placeholder="请输入数据集的简要描述（可选）"
            />
          </el-form-item>

          <el-divider content-position="left">图像来源</el-divider>

          <el-form-item label="选择图片来源方式" prop="sourceMethod">
            <el-radio-group v-model="datasetForm.sourceMethod">
              <el-radio label="upload">上传本地文件</el-radio>
              <el-radio label="server_path">从服务器路径导入</el-radio>
            </el-radio-group>
          </el-form-item>

          <el-form-item v-if="datasetForm.sourceMethod === 'upload'" label="数据集文件 (图像)" prop="files">
            <el-upload
              class="upload-demo"
              drag
              action="#"
              :auto-upload="false"
              :on-change="handleFileChange"
              :file-list="fileList"
              :on-remove="handleFileRemove"
              multiple
            >
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">
                拖拽文件到此处或 <em>点击上传</em>
              </div>
              <template #tip>
                <div class="el-upload__tip">
                  支持jpg/png/gif等格式的眼底图像文件
                </div>
              </template>
            </el-upload>
          </el-form-item>

          <el-form-item v-if="datasetForm.sourceMethod === 'server_path'" label="服务器图片文件夹路径" prop="serverPath">
            <el-input v-model="datasetForm.serverPath" placeholder="请输入服务器上的图片文件夹绝对路径" />
            <div class="el-upload__tip">
              例如: /data/my_images 或 D:\\datasets\\retina_images
            </div>
          </el-form-item>
        </el-form>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="datasetDialogVisible = false">取消</el-button>
            <el-button type="primary" @click="submitDataset">
              {{ editingDatasetId && !isCopyCreateMode ? '保存更改' : '确定添加' }}
            </el-button>
          </span>
        </template>
      </el-dialog>

      <!-- 预处理任务进度对话框 -->
      <el-dialog
        v-model="processDialogVisible"
        title="预处理任务进度"
        width="600px"
        class="process-dialog"
        :close-on-click-modal="false"
        :close-on-press-escape="false"
        :show-close="true"
      >
        <div class="process-progress">
          <el-progress
            :percentage="processProgress"
            :status="processStatus"
            :stroke-width="20"
            :show-text="false"
          />
          <div class="process-info">
            <p>已处理: {{ processedCount }} / {{ totalCount }}</p>
            <p>预计剩余时间: {{ remainingTime }}</p>
          </div>
        </div>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="processDialogVisible = false" :disabled="processStatus !== 'success'">关闭</el-button>
          </span>
        </template>
      </el-dialog>

      <!-- 数据集预览对话框 -->
      <el-dialog
        v-model="previewDialogVisible"
        title="数据集预览"
        width="90%"
        :close-on-click-modal="false"
        class="preview-dialog"
        :show-close="true"
      >
        <div v-if="currentDataset" class="preview-header">
          <h3>{{ currentDataset.name }}</h3>
          <div class="preview-tools">
            <div class="breadcrumb">
              <el-breadcrumb separator="/">
                <el-breadcrumb-item @click="navigateToDir('')" class="breadcrumb-item">
                  根目录
                </el-breadcrumb-item>
                <el-breadcrumb-item
                  v-for="crumb in breadcrumbs"
                  :key="crumb.path"
                  @click="navigateToDir(crumb.path)"
                  class="breadcrumb-item"
                >
                  {{ crumb.name }}
                </el-breadcrumb-item>
              </el-breadcrumb>
            </div>
            <div class="search-box">
              <el-input
                v-model="previewSearchQuery"
                placeholder="搜索图片名称"
                class="preview-search-input"
                clearable
                @input="handlePreviewSearch"
              >
                <template #prefix>
                  <el-icon><Search /></el-icon>
                </template>
              </el-input>
            </div>
          </div>
        </div>

        <div class="preview-content">
          <div v-if="directories.length === 0 && previewImages.length === 0" class="no-content">
            当前目录为空
          </div>
          <div v-else>
            <div v-if="directories.length > 0" class="directory-list">
              <div
                v-for="dir in directories"
                :key="dir.path"
                class="directory-item"
                @click="navigateToDir(dir.path)"
              >
                <el-icon><Folder /></el-icon>
                <span>{{ dir.name }}</span>
              </div>
            </div>

            <div v-if="previewImages.length > 0" class="image-grid">
              <div v-for="image in previewImages" :key="image.path" class="image-item">
                <img :src="getImageUrl(currentDataset, image.path)" :alt="image.name" />
                <div class="image-name">{{ image.name }}</div>
              </div>
            </div>

            <div v-if="previewImages.length > 0" class="preview-pagination">
              <el-pagination
                v-model:current-page="previewPage"
                v-model:page-size="previewPageSize"
                :page-sizes="[20, 40, 60, 100]"
                :total="previewTotalPages * previewPageSize"
                layout="total, sizes, prev, pager, next, jumper"
                @size-change="handlePreviewSizeChange"
                @current-change="handlePreviewPageChange"
              />
            </div>
          </div>
        </div>
      </el-dialog>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Plus, Search, UploadFilled, Folder, DocumentCopy as DocumentCopyIcon,
  View as ViewIcon, Delete as DeleteIcon, Setting as SettingIcon, PriceTag as PriceTagIcon
} from '@element-plus/icons-vue'
import axios from 'axios'

const activeTab = ref('dataset')
const searchQuery = ref('')
const isSearching = ref(false)
const tableLoading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)

const processDialogVisible = ref(false)
const datasetDialogVisible = ref(false)
const processProgress = ref(0)
const processStatus = ref('')
const processedCount = ref(0)
const totalCount = ref(0)
const remainingTime = ref('')

const datasetFormRef = ref(null)
const editingDatasetId = ref(null); // null for add, ID for edit/template
const isCopyCreateMode = ref(false); // True when '以此为模板创建' is used

const datasetForm = ref({
  name: '',
  type: 'public',
  description: '',
  files: [],
  sourceMethod: 'upload',
  serverPath: ''
})
const fileList = ref([])

const datasetTypeOptions = ref([
  { value: 'public', label: '公开数据集' },
  { value: 'clinical', label: '临床研究' },
  { value: 'private', label: '私有项目' },
])

const rules = ref({
  name: [
    { required: true, message: '请输入数据集名称', trigger: 'blur' },
    { min: 3, max: 50, message: '长度在 3 到 50 个字符', trigger: 'blur' }
  ],
  type: [
    { required: true, message: '请选择数据集类型', trigger: 'change' }
  ],
  sourceMethod: [
    { required: true, message: '请选择图片来源', trigger: 'change' }
  ],
  files: [
    {
      validator: (rule, value, callback) => {
        // Only validate if sourceMethod is 'upload' and not in 'copy create' mode where files are not pre-filled
        if (datasetForm.value.sourceMethod === 'upload' && !isCopyCreateMode.value && fileList.value.length === 0) {
          callback(new Error('请选择要上传的图片文件'))
        } else {
          callback()
        }
      },
      trigger: 'change'
    }
  ],
  serverPath: [
    {
      validator: (rule, value, callback) => {
        if (datasetForm.value.sourceMethod === 'server_path' && !isCopyCreateMode.value && !value) {
          callback(new Error('请输入服务器图片路径'))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ]
})

const processForm = ref({
  method: 'normalize',
  customParams: ''
})
const selectedDatasets = ref([]) // For "预处理任务" tab
const datasetList = ref([]) // For "数据集管理" table

const selectedType = ref('all') // Filter for table
const selectedStatus = ref('all') // Filter for table

// Preview Dialog State
const previewDialogVisible = ref(false)
const previewImages = ref([])
const currentDataset = ref(null)
const previewPage = ref(1)
const previewPageSize = ref(20)
const previewTotalPages = ref(1)
const currentDir = ref('')
const breadcrumbs = ref([])
const directories = ref([])
const previewSearchQuery = ref('')
const originalPreviewImages = ref([])


const fetchDatasetList = async () => {
  try {
    tableLoading.value = true
    const response = await axios.get('http://localhost:8000/api/datasets/list_local/', {
      params: {
        base_path: 'F:/DRDiaSys/DRDiaSys/django/DRDiaSys/datasets/dataset'
      }
    })

    if (response.status === 200) {
      datasetList.value = response.data.datasets.map((dataset, index) => ({
        id: dataset.id,
        name: dataset.name,
        imageCount: dataset.image_count,
        createTime: new Date(dataset.created_at).toLocaleString(),
        status: dataset.status,
        path: dataset.path,
        type: dataset.type || (index % 2 === 0 ? 'public' : 'clinical'),
        description: dataset.description || `这是 ${dataset.name} 的描述信息。`,
        version: dataset.status === 'processed' ? (dataset.version || `v1.${index % 3}`) : null
      }))
    }
  } catch (error) {
    console.error('获取数据集列表失败:', error)
    ElMessage.error('获取数据集列表失败')
  } finally {
    tableLoading.value = false
  }
}

const filteredDatasetList = computed(() => {
  let filtered = datasetList.value
  if (selectedType.value !== 'all') {
    filtered = filtered.filter(item => item.type === selectedType.value)
  }
  if (selectedStatus.value !== 'all') {
    filtered = filtered.filter(item => item.status === selectedStatus.value)
  }
  if (searchQuery.value && isSearching.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(item =>
      item.name.toLowerCase().includes(query) ||
      (item.description && item.description.toLowerCase().includes(query))
    )
  }
  return filtered
})

const totalFilteredDatasets = computed(() => filteredDatasetList.value.length)

const paginatedFilteredDatasetList = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filteredDatasetList.value.slice(start, end)
})

const getStatusType = (status) => {
  const types = { unprocessed: 'info', processing: 'warning', processed: 'success' }
  return types[status] || 'info'
}

const getTableStatusLabel = (status, version) => {
  const labels = { unprocessed: '原始', processing: '处理中', processed: `已处理${version ? ` (${version})` : ''}` }
  return labels[status] || '未知'
}

const getDatasetTypeLabel = (type) => {
  const labels = { public: '公开', clinical: '临床', private: '私有项目' }
  return labels[type] || type
}

const applyFiltersAndResetPage = () => {
  currentPage.value = 1
}

const performSearch = () => {
  isSearching.value = !!searchQuery.value.trim();
  currentPage.value = 1;
};

const clearSearchAndFilter = () => {
  searchQuery.value = ''
  isSearching.value = false
  currentPage.value = 1
}

const handleSizeChange = (val) => {
  pageSize.value = val
  currentPage.value = 1
}

const handleCurrentChange = (val) => {
  currentPage.value = val
}

const handleAddDataset = (templateData = null) => {
  if (templateData && typeof templateData === 'object') {
    editingDatasetId.value = Date.now(); // Indicate it's based on a template, but will be a new entry
    isCopyCreateMode.value = true;
    datasetForm.value = {
      name: `${templateData.name || '新数据集'}_副本`,
      type: templateData.type || 'public',
      description: templateData.description || '',
      sourceMethod: 'upload', // Default to upload, user must re-select files
      files: [],
      serverPath: '',
    };
    fileList.value = [];
  } else {
    editingDatasetId.value = null;
    isCopyCreateMode.value = false;
    resetDatasetFormInternal();
  }
  datasetDialogVisible.value = true;
}

const resetDatasetFormInternal = () => {
  datasetForm.value = {
    name: '',
    type: 'public',
    description: '',
    files: [],
    sourceMethod: 'upload',
    serverPath: ''
  };
  fileList.value = [];
  if (datasetFormRef.value) {
    datasetFormRef.value.clearValidate(); // Use clearValidate for better reset
    datasetFormRef.value.resetFields();
  }
}

const resetDatasetForm = () => { // Dialog close callback
  editingDatasetId.value = null;
  isCopyCreateMode.value = false;
  resetDatasetFormInternal();
}


const handleFileChange = (file, uploadedFileList) => { // El-upload onChange provides uploadedFileList
  fileList.value = uploadedFileList; // Keep fileList in sync with el-upload's internal list
}

const handleFileRemove = (file, uploadedFileList) => {
 fileList.value = uploadedFileList;
}

const submitDataset = async () => {
  if (!datasetFormRef.value) return;
  try {
    await datasetFormRef.value.validate();
    const basePathForAPI = 'F:/DRDiaSys/DRDiaSys/django/DRDiaSys/datasets/dataset';
    let response;

    const payload = {
      name: datasetForm.value.name,
      type: datasetForm.value.type,
      description: datasetForm.value.description,
      base_path: basePathForAPI,
    };

    // For both new and "copy create", it's a POST to create a new dataset
    if (datasetForm.value.sourceMethod === 'upload') {
      const formData = new FormData();
      Object.keys(payload).forEach(key => formData.append(key, payload[key]));
      fileList.value.forEach(f => { // Use the synced fileList
        if (f.raw) formData.append('files', f.raw);
      });
      response = await axios.post('http://localhost:8000/api/datasets/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
    } else if (datasetForm.value.sourceMethod === 'server_path') {
      payload.server_path = datasetForm.value.serverPath;
      response = await axios.post('http://localhost:8000/api/datasets/import_from_path/', payload);
    }

    if (response && (response.status === 201 || response.status === 200)) {
      ElMessage.success('数据集添加成功'); // Always "added" in this simplified flow
      datasetDialogVisible.value = false;
      await fetchDatasetList();
    }
  } catch (error) {
    console.error('操作数据集失败:', error);
    ElMessage.error(error.response?.data?.error || '操作数据集失败');
  }
}


const handleView = async (row) => {
  try {
    currentDir.value = ''
    previewPage.value = 1
    previewSearchQuery.value = ''
    currentDataset.value = row
    await fetchPreviewData()
    previewDialogVisible.value = true
  } catch (error) {
    console.error('获取数据集预览失败:', error)
    ElMessage.error(error.response?.data?.error || '获取数据集预览失败')
  }
}

const fetchPreviewData = async () => {
  if (!currentDataset.value) return
  try {
    const response = await axios.get(`http://localhost:8000/api/datasets/${currentDataset.value.id}/preview/`, {
      params: {
        base_path: 'F:/DRDiaSys/DRDiaSys/django/DRDiaSys/datasets/dataset',
        dir: currentDir.value,
        page: previewPage.value,
        page_size: previewPageSize.value
      }
    })
    if (response.status === 200) {
      originalPreviewImages.value = response.data.files
      previewImages.value = response.data.files
      previewTotalPages.value = response.data.total_pages
      directories.value = response.data.directories
      breadcrumbs.value = response.data.breadcrumbs
      currentDir.value = response.data.current_dir
    }
  } catch (error) {
    console.error('获取预览数据失败:', error)
    throw error
  }
}

const handleSubmitToProcess = async (row) => {
  try {
    // Assuming this API changes the dataset's status or adds it to a processing queue
    const response = await axios.post(`http://localhost:8000/api/datasets/${row.id}/submit_process/`);
    if (response.status === 200) {
      activeTab.value = 'process';
      // Add to selectedDatasets for the "预处理任务" tab if not already there
      if (!selectedDatasets.value.find(item => item.id === row.id)) {
        selectedDatasets.value.push({ ...row, status: 'processing' }); // Reflect status change locally
      }
      ElMessage.success(`已将数据集 "${row.name}" 提交处理`);
      await fetchDatasetList(); // Refresh main list to show status changes
    }
  } catch (error) {
    console.error('提交预处理失败:', error);
    ElMessage.error(error.response?.data?.error || '提交预处理失败');
  }
};


const handleEditTags = (row) => {
  ElMessage.info(`“编辑/标签”功能开发中：${row.name}`)
}

const removeFromTask = (row) => {
  const index = selectedDatasets.value.findIndex(item => item.id === row.id)
  if (index !== -1) {
    selectedDatasets.value.splice(index, 1)
    ElMessage.success(`已从处理任务中移除数据集"${row.name}"`)
  }
}

const startProcess = async () => {
  if (!selectedDatasets.value.length) {
    ElMessage.warning('请先选择要处理的数据集')
    return
  }
  if (!processForm.value.method) {
    ElMessage.warning('请选择处理方式')
    return
  }
  processDialogVisible.value = true
  processProgress.value = 0
  processStatus.value = ''
  processedCount.value = 0
  totalCount.value = selectedDatasets.value.reduce((sum, ds) => sum + (ds.imageCount || 0), 0)
  remainingTime.value = '计算中...'

  try {
    const response = await axios.post('http://localhost:8000/api/process/start/', {
      dataset_ids: selectedDatasets.value.map(ds => ds.id),
      method: processForm.value.method,
      custom_params: processForm.value.customParams
    })

    if (response.status === 200) {
      const taskId = response.data.task_id
      const pollInterval = setInterval(async () => {
        try {
          const progressResponse = await axios.get(`http://localhost:8000/api/process/${taskId}/progress/`)
          if (progressResponse.status === 200) {
            const progress = progressResponse.data
            processProgress.value = progress.percentage
            processedCount.value = progress.processed_count
            totalCount.value = progress.total_count
            remainingTime.value = progress.remaining_time

            if (progress.status === 'completed') {
              clearInterval(pollInterval)
              processStatus.value = 'success'
              ElMessage.success('预处理完成')
              selectedDatasets.value = []
              await fetchDatasetList()
            } else if (progress.status === 'failed') {
              clearInterval(pollInterval)
              processStatus.value = 'exception'
              ElMessage.error(progress.error_message || '预处理失败')
            } else {
              processStatus.value = ''
            }
          }
        } catch (error) {
          console.error('获取处理进度失败:', error)
          clearInterval(pollInterval)
          processStatus.value = 'exception'
          ElMessage.error('获取处理进度失败')
        }
      }, 2000)
    }
  } catch (error) {
    console.error('开始处理失败:', error)
    ElMessage.error(error.response?.data?.error || '开始处理失败')
    processDialogVisible.value = false
  }
}

const handleDelete = async (row) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除数据集"${row.name}"吗？此操作不可恢复。`,
      '警告',
      { confirmButtonText: '确定删除', cancelButtonText: '取消', type: 'warning' }
    )
    const response = await axios.delete(`http://localhost:8000/api/datasets/${row.id}/`, {
      params: { base_path: 'F:/DRDiaSys/DRDiaSys/django/DRDiaSys/datasets/dataset' }
    })
    if (response.status === 204) {
      ElMessage.success('删除成功')
      await fetchDatasetList()
    }
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除数据集失败:', error)
      ElMessage.error(error.response?.data?.error || '删除数据集失败')
    }
  }
}

const handlePreviewPageChange = async (page) => {
  previewPage.value = page
  await fetchPreviewData()
}
const handlePreviewSizeChange = async (size) => {
  previewPageSize.value = size
  previewPage.value = 1
  await fetchPreviewData()
}
const navigateToDir = async (dir) => {
  if (!currentDataset.value) return
  try {
    const targetDir = dir === '' ? '' : dir.replace(/^\/+|\/+$/g, '')
    currentDir.value = targetDir
    previewPage.value = 1
    previewSearchQuery.value = ''
    await fetchPreviewData()
  } catch (error) {
    console.error('导航到目录失败:', error)
    ElMessage.error(error.response?.data?.error || '导航到目录失败')
  }
}
const handlePreviewSearch = () => {
  if (!previewSearchQuery.value) {
    previewImages.value = originalPreviewImages.value
    return
  }
  const query = previewSearchQuery.value.toLowerCase()
  previewImages.value = originalPreviewImages.value.filter(image =>
    image.name.toLowerCase().includes(query)
  )
}
const getImageUrl = (dataset, imagePath) => {
  const encodedImagePath = encodeURIComponent(imagePath)
  return `http://localhost:8000/api/datasets/${dataset.id}/image/${encodedImagePath}`
}

onMounted(async () => {
  await fetchDatasetList()
})
</script>

<style scoped>
.image-processing-container {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  padding-top: 20px;
  background: #f0f2f5;
  overflow-y: auto;
}

.image-processing-box {
  width: 100%;
  max-width: 1800px;
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1);
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid #e8e8e8;
}

.title-container h2 {
  font-size: 22px;
  color: #303133;
  margin: 0;
  font-weight: 600;
}

.subtitle {
  color: #909399;
  font-size: 14px;
  margin-top: 4px;
}

.sidebar-filters {
  background-color: #fff;
  padding: 15px;
  border-radius: 4px;
}

.filter-group {
  margin-bottom: 20px;
}
.filter-group:last-child {
  margin-bottom: 0;
}

.filter-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin-bottom: 12px;
}

.sidebar-filters .el-radio-group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.sidebar-filters .el-radio {
  margin-bottom: 10px;
}
.sidebar-filters .el-radio:last-child {
  margin-bottom: 0;
}

.main-content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}
.search-add-container {
  display: flex;
  align-items: center;
}
.search-input {
  width: 300px;
}
.search-active-tag {
  margin-bottom: 15px;
}

.dataset-table-card {
  border: none;
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

:deep(.el-tabs__item) {
  font-size: 15px;
  color: #606266;
}
:deep(.el-tabs__item.is-active) {
  color: var(--el-color-primary);
  font-weight: 600;
}
:deep(.el-tabs__nav-wrap::after) {
  background-color: #e4e7ed;
}

:deep(.dataset-dialog .el-dialog__header),
:deep(.process-dialog .el-dialog__header) {
  background: var(--el-color-primary);
  color: #fff;
}
:deep(.dataset-dialog .el-dialog__title),
:deep(.process-dialog .el-dialog__title) {
  color: #fff;
}

:deep(.el-dialog__headerbtn .el-dialog__close) {
  color: #909399;
}
:deep(.dataset-dialog .el-dialog__headerbtn .el-dialog__close),
:deep(.process-dialog .el-dialog__headerbtn .el-dialog__close) {
  color: #fff;
}
:deep(.el-dialog__headerbtn .el-dialog__close:hover) {
    color: var(--el-color-primary);
}

.preview-dialog :deep(.el-dialog) {
  width: 90% !important;
  max-width: 1800px;
  height: 90vh;
  display: flex;
  flex-direction: column;
  margin: 5vh auto !important;
}

.preview-dialog :deep(.el-dialog__body) {
  flex: 1;
  overflow-y: auto;
  padding: 20px 30px;
}

.preview-header {
  margin-bottom: 20px;
}
.preview-header h3 {
  margin: 0 0 15px;
  color: #303133;
  font-size: 20px;
}
.preview-header .preview-tools {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 30px;
}
.preview-header .preview-tools .breadcrumb {
  flex: 1;
  margin-right: 20px;
}
.preview-header .preview-tools .search-box {
  width: 300px;
  flex-shrink: 0;
}
.preview-header .preview-tools .search-box .preview-search-input :deep(.el-input__wrapper) {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
.preview-header .preview-tools .search-box .preview-search-input :deep(.el-input__wrapper:hover),
.preview-header .preview-tools .search-box .preview-search-input :deep(.el-input__wrapper.is-focus) {
  box-shadow: 0 2px 12px rgba(33, 150, 243, 0.2);
}
.preview-header .preview-tools .search-box .preview-search-input :deep(.el-input__inner) {
  height: 40px;
}

.preview-content .no-content {
  text-align: center;
  color: #909399;
  padding: 60px 0;
  font-size: 16px;
}

.breadcrumb {
  margin: 10px 0;
  padding: 10px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
:deep(.breadcrumb-item) {
  cursor: pointer;
  transition: all 0.3s ease;
  padding: 4px 8px;
  border-radius: 4px;
}
:deep(.breadcrumb-item:hover) {
  background-color: rgba(33, 150, 243, 0.1);
  color: #2196F3;
  transform: translateY(-1px);
}
:deep(.el-breadcrumb__separator) {
  margin: 0 8px;
  color: #909399;
}

.directory-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}
.directory-item {
  display: flex;
  align-items: center;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
  border: 1px solid transparent;
}
.directory-item:hover {
  background: #e4e7ed;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  border-color: #2196F3;
}
.directory-item .el-icon {
  margin-right: 10px;
  font-size: 20px;
  color: #909399;
  transition: all 0.3s ease;
}
.directory-item:hover .el-icon {
  color: #2196F3;
  transform: scale(1.1);
}
.directory-item span {
  color: #606266;
  font-size: 14px;
  transition: all 0.3s ease;
}
.directory-item:hover span {
  color: #2196F3;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 25px;
  padding: 20px 0 30px;
}
.image-grid .image-item {
  background: #fff;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 3px 15px 0 rgba(0,0,0,0.1);
  transition: all 0.3s;
}
.image-grid .image-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 20px 0 rgba(0,0,0,0.2);
}
.image-grid .image-item img {
  width: 100%;
  height: 220px;
  object-fit: cover;
}
.image-grid .image-item .image-name {
  padding: 12px;
  font-size: 14px;
  color: #606266;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.preview-content .preview-pagination {
  margin-top: 30px;
  display: flex;
  justify-content: center;
}

:deep(.dataset-dialog),
:deep(.process-dialog) {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

:deep(.el-dialog__header) {
  margin: 0;
  padding: 16px 20px;
  position: relative;
}

:deep(.el-dialog__title) {
  font-weight: 600;
  font-size: 16px;
}

:deep(.el-dialog__headerbtn) {
  top: 18px;
  right: 20px;
}

:deep(.el-dialog__body) {
  padding: 20px;
}

:deep(.el-dialog__footer) {
  padding: 10px 20px;
  border-top: 1px solid #ebeef5;
  text-align: right;
}

:deep(.el-form-item__label) {
  font-weight: 500;
  color: #333;
  font-size: 14px;
  padding-bottom: 6px;
}

:deep(.dialog-footer .el-button) {
  padding: 8px 15px;
  font-size: 14px;
  border-radius: 4px;
}

:deep(.el-table) {
  border-radius: 4px;
  overflow: hidden;
}

:deep(.el-progress-bar__outer) {
  background-color: #EBEEF5;
}
:deep(.el-progress-bar__inner) {
  background-color: var(--el-color-primary);
}
:deep(.el-progress-bar__inner.is-success) {
  background-color: var(--el-color-success);
}
:deep(.el-progress-bar__inner.is-exception) {
  background-color: var(--el-color-danger);
}

.process-container {
  display: flex;
  gap: 20px;
}
.process-method {
  flex: 1;
  min-width: 300px;
}
.task-list {
  flex: 2;
}
.process-method :deep(.el-form-item) {
  margin-bottom: 20px;
}
.process-method :deep(.el-radio-group) {
  display: flex;
  flex-direction: column;
  gap: 15px;
}
.task-list :deep(.card-header) {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.process-progress {
  text-align: center;
}
.process-progress .el-progress {
  margin-bottom: 15px;
}
.process-info p {
  margin: 5px 0;
  color: #606266;
}

.el-upload__tip {
  color: #909399;
  font-size: 12px;
  margin-top: 5px;
}

:deep(.dataset-form.el-form--label-top .el-form-item__label) {
  display: block !important;
  text-align: left !important;
  float: none !important;
  width: auto !important;
  line-height: normal !important;
  padding-bottom: 8px !important;
}

:deep(.dataset-form.el-form--label-top .el-form-item__content) {
  margin-left: 0 !important;
}
:deep(.dataset-form .el-form-item) {
  margin-bottom: 22px;
}
</style>
