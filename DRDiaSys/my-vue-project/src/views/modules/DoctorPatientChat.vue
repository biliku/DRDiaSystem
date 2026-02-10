<template>
  <div class="chat-page">
    <el-row :gutter="20" style="height: calc(100vh - 100px)">
      <!-- 左侧：会话列表 -->
      <el-col :span="6">
        <el-card shadow="never" class="conversation-list-card">
          <template #header>
            <div class="conversation-header">
              <h3>{{ isDoctor ? '患者列表' : '我的咨询' }}</h3>
              <el-button
                v-if="isDoctor"
                type="primary"
                size="small"
                @click="openCreateConversationDialog"
              >
                <el-icon><Plus /></el-icon>
                新建会话
              </el-button>
            </div>
          </template>
          <el-input
            v-model="conversationSearchQuery"
            placeholder="搜索"
            clearable
            class="search-input"
            @input="filterConversations"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
          <div class="conversation-list" v-loading="conversationLoading">
            <div
              v-for="conv in filteredConversations"
              :key="conv.id"
              class="conversation-item"
              :class="{ active: selectedConversation?.id === conv.id }"
              @click="selectConversation(conv)"
            >
              <div class="conversation-avatar">
                <el-avatar :size="40">
                  {{ (isDoctor ? conv.patient_name : conv.doctor_name)?.charAt(0) }}
                </el-avatar>
                <el-badge
                  :value="isDoctor ? conv.doctor_unread_count : conv.patient_unread_count"
                  :hidden="(isDoctor ? conv.doctor_unread_count : conv.patient_unread_count) === 0"
                  class="unread-badge"
                />
              </div>
              <div class="conversation-info">
                <div class="conversation-name">
                  {{ isDoctor ? conv.patient_name : conv.doctor_name }}
                </div>
                <div class="conversation-last-message">
                  {{ conv.last_message || '暂无消息' }}
                </div>
                <div class="conversation-time">
                  {{ formatTime(conv.last_message_at) }}
                </div>
              </div>
            </div>
            <el-empty v-if="filteredConversations.length === 0" description="暂无会话" />
          </div>
        </el-card>
      </el-col>

      <!-- 右侧：消息区域 -->
      <el-col :span="18">
        <el-card shadow="never" class="message-card" v-if="selectedConversation">
          <template #header>
            <div class="message-header">
              <div class="message-header-info">
                <el-avatar :size="32">
                  {{ (isDoctor ? selectedConversation.patient_name : selectedConversation.doctor_name)?.charAt(0) }}
                </el-avatar>
                <div style="margin-left: 10px">
                  <div class="message-header-name">
                    {{ isDoctor ? selectedConversation.patient_name : selectedConversation.doctor_name }}
                  </div>
                  <div class="message-header-meta">
                    {{ selectedConversation.related_case_info?.title || '未关联病例' }}
                  </div>
                </div>
              </div>
              <div class="message-header-actions">
                <el-button
                  v-if="isDoctor"
                  type="primary"
                  size="small"
                  @click="openTemplateDialog"
                >
                  <el-icon><Document /></el-icon>
                  快捷回复
                </el-button>
              </div>
            </div>
          </template>

          <!-- 消息列表 -->
          <div class="message-list" ref="messageListRef" v-loading="messageLoading">
            <div
            v-for="msg in messages"
            :key="msg.id"
            class="message-item"
            :class="{ 'message-sent': isSent(msg), 'message-received': !isSent(msg) }"
            >
              <el-avatar :size="32" class="message-avatar">
                {{ msg.sender_name?.charAt(0) }}
              </el-avatar>
              <div class="message-content">
                <div class="message-header-small">
                  <span class="message-sender">{{ msg.sender_name }}</span>
                  <span class="message-time">{{ formatTime(msg.created_at) }}</span>
                </div>
                <div class="message-body">
                  <!-- 文字消息 -->
                  <div v-if="msg.message_type === 'text'" class="message-text">
                    {{ msg.content }}
                  </div>
                  <!-- 图片消息 -->
                  <div v-else-if="msg.message_type === 'image'" class="message-image">
                    <el-image
                      :src="getFileUrl(msg.file_url)"
                      :preview-src-list="[getFileUrl(msg.file_url)]"
                      fit="cover"
                      style="max-width: 300px; max-height: 300px"
                    />
                  </div>
                  <!-- 文件消息 -->
                  <div v-else-if="msg.message_type === 'file'" class="message-file">
                    <el-icon><Document /></el-icon>
                    <span>{{ msg.file_name }}</span>
                    <el-button
                      type="primary"
                      link
                      size="small"
                      @click="downloadFile(msg.file_url, msg.file_name)"
                    >
                      下载
                    </el-button>
                  </div>
                  <!-- 关联报告 -->
                  <div v-if="msg.related_report_info" class="message-related">
                    <el-tag type="info" size="small">关联报告：{{ msg.related_report_info.report_number }}</el-tag>
                  </div>
                  <!-- 关联治疗方案 -->
                  <div v-if="msg.related_treatment_plan_info" class="message-related">
                    <el-tag type="success" size="small">关联方案：{{ msg.related_treatment_plan_info.title }}</el-tag>
                  </div>
                </div>
              </div>
            </div>
            <el-empty v-if="messages.length === 0" description="暂无消息，开始对话吧" />
          </div>

          <!-- 消息输入区 -->
          <div class="message-input-area">
            <div class="input-toolbar">
              <el-button
                type="primary"
                link
                size="small"
                @click="openFileUpload"
              >
                <el-icon><Picture /></el-icon>
                图片
              </el-button>
              <el-button
                type="primary"
                link
                size="small"
                @click="openFileUpload('file')"
              >
                <el-icon><Document /></el-icon>
                文件
              </el-button>
              <input
                ref="fileInputRef"
                type="file"
                style="display: none"
                :accept="currentFileType === 'image' ? 'image/*' : '.pdf,.doc,.docx,.xls,.xlsx,.txt,.zip,.rar'"
                @change="handleFileSelect"
              />
            </div>
            <el-input
              v-model="newMessage"
              type="textarea"
              :rows="3"
              placeholder="输入消息..."
              @keydown.ctrl.enter="sendMessage"
            />
            <div class="input-actions">
              <el-button @click="clearInput">清空</el-button>
              <el-button type="primary" @click="sendMessage" :loading="sending">
                发送 (Ctrl+Enter)
              </el-button>
            </div>
          </div>
        </el-card>
        <el-card v-else shadow="never" class="message-card">
          <el-empty description="请选择一个会话或创建新会话" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 创建会话对话框 -->
    <el-dialog
      v-model="createConversationDialogVisible"
      title="新建会话"
      width="500px"
    >
      <el-form :model="createConversationForm" label-width="80px">
        <el-form-item label="患者" required>
          <el-select
            v-model="createConversationForm.patient_id"
            filterable
            placeholder="选择患者"
            style="width: 100%"
            @change="onPatientChange"
          >
            <el-option
              v-for="patient in patients"
              :key="patient.id"
              :label="patient.username"
              :value="patient.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="关联病例">
          <el-select
            v-model="createConversationForm.case_id"
            filterable
            placeholder="选择病例（可选）"
            style="width: 100%"
          >
            <el-option
              v-for="caseItem in cases"
              :key="caseItem.id"
              :label="caseItem.title"
              :value="caseItem.id"
            />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createConversationDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="createConversation" :loading="creating">创建</el-button>
      </template>
    </el-dialog>

    <!-- 快捷回复模板对话框 -->
    <el-dialog
      v-model="templateDialogVisible"
      title="快捷回复"
      width="600px"
    >
      <el-table
        :data="templates"
        v-loading="templateLoading"
        @row-click="useTemplate"
      >
        <el-table-column prop="title" label="模板标题" />
        <el-table-column prop="category" label="分类" width="120" />
        <el-table-column label="操作" width="100">
          <template #default="scope">
            <el-button type="primary" link size="small" @click.stop="useTemplate(scope.row)">
              使用
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-dialog>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage } from 'element-plus'
import {
  Plus,
  Search,
  Document,
  Picture
} from '@element-plus/icons-vue'

export default {
  name: 'DoctorPatientChat',
  components: {
    Plus,
    Search,
    Document,
    Picture
  },
  data() {
    return {
      isDoctor: localStorage.getItem('userRole') === 'doctor',
      currentUserId: parseInt(localStorage.getItem('userId') || '0'),
      conversationLoading: false,
      conversations: [],
      filteredConversations: [],
      conversationSearchQuery: '',
      selectedConversation: null,
      messageLoading: false,
      messages: [],
      newMessage: '',
      sending: false,
      messageListRef: null,
      fileInputRef: null,
      currentFileType: 'image',
      createConversationDialogVisible: false,
      createConversationForm: {
        patient_id: null,
        case_id: null
      },
      patients: [],
      cases: [],
      allCases: [],
      creating: false,
      templateDialogVisible: false,
      templates: [],
      templateLoading: false,
      pollTimer: null
    }
  },
  created() {
    this.fetchConversations()
    // 开始轮询未读消息
    this.startPolling()
  },
  beforeUnmount() {
    this.stopPolling()
  },
  methods: {
    async fetchConversations() {
      this.conversationLoading = true
      try {
        const res = await api.get('/api/treatment/conversations/')
        this.conversations = res.data
        this.filteredConversations = res.data
        // 如果有未读消息，自动选择第一个
        if (this.conversations.length > 0 && !this.selectedConversation) {
          const unreadConv = this.conversations.find(c =>
            (this.isDoctor ? c.doctor_unread_count : c.patient_unread_count) > 0
          )
          if (unreadConv) {
            this.selectConversation(unreadConv)
          } else {
            this.selectConversation(this.conversations[0])
          }
        }
      } catch (error) {
        ElMessage.error('获取会话列表失败')
      } finally {
        this.conversationLoading = false
      }
    },
    isSent(msg) {
      try {
        const uid = this.currentUserId || Number(localStorage.getItem('userId') || '0')
        // msg.sender may be id, string id, or nested object; also compare sender_name as fallback
        if (!msg) return false
        if (msg.sender && typeof msg.sender === 'number') {
          if (Number(msg.sender) === uid) return true
        }
        if (msg.sender && typeof msg.sender === 'string') {
          if (Number(msg.sender) === uid || msg.sender === String(uid)) return true
        }
        if (msg.sender && typeof msg.sender === 'object' && (msg.sender.id || msg.sender.pk)) {
          const sid = Number(msg.sender.id || msg.sender.pk)
          if (sid === uid) return true
        }
        const userName = localStorage.getItem('userName')
        if (msg.sender_name && userName && msg.sender_name === userName) return true
      } catch (e) {
        // ignore
      }
      return false
    },
    filterConversations() {
      if (!this.conversationSearchQuery) {
        this.filteredConversations = this.conversations
        return
      }
      const query = this.conversationSearchQuery.toLowerCase()
      this.filteredConversations = this.conversations.filter(conv =>
        (this.isDoctor ? conv.patient_name : conv.doctor_name)?.toLowerCase().includes(query) ||
        conv.last_message?.toLowerCase().includes(query)
      )
    },
    async selectConversation(conversation) {
      this.selectedConversation = conversation
      await this.fetchMessages()
      // 刷新会话列表以更新未读数
      await this.fetchConversations()
      // 滚动到底部
      this.$nextTick(() => {
        this.scrollToBottom()
      })
    },
    async fetchMessages() {
      if (!this.selectedConversation) return
      this.messageLoading = true
      try {
        const res = await api.get(`/api/treatment/conversations/${this.selectedConversation.id}/messages/`)
        this.messages = res.data
      } catch (error) {
        ElMessage.error('获取消息失败')
      } finally {
        this.messageLoading = false
      }
    },
    async sendMessage() {
      if (!this.newMessage.trim() || !this.selectedConversation) {
        return
      }
      this.sending = true
      try {
        await api.post(`/api/treatment/conversations/${this.selectedConversation.id}/messages/`, {
          message_type: 'text',
          content: this.newMessage
        })
        this.newMessage = ''
        await this.fetchMessages()
        await this.fetchConversations()
        this.scrollToBottom()
      } catch (error) {
        ElMessage.error('发送消息失败')
      } finally {
        this.sending = false
      }
    },
    clearInput() {
      this.newMessage = ''
    },
    openFileUpload(type = 'image') {
      this.currentFileType = type
      // clear previous selection
      if (this.$refs.fileInputRef) {
        this.$refs.fileInputRef.value = null
        this.$refs.fileInputRef.click()
      }
    },
    async handleFileSelect(event) {
      const file = event.target.files[0]
      if (!file) return
      // frontend-level validation for common types/sizes
      const maxSize = 10 * 1024 * 1024 // 10MB
      if (file.size > maxSize) {
        ElMessage.warning('文件过大，最大支持 10MB')
        event.target.value = ''
        return
      }
      if (this.currentFileType === 'image') {
        if (!file.type.startsWith('image/')) {
          ElMessage.warning('请选择图片文件')
          event.target.value = ''
          return
        }
      } else {
        const allowedExt = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.zip', '.rar']
        const ext = (file.name.match(/(\.[^.]+)$/) || [''])[0].toLowerCase()
        if (!allowedExt.includes(ext)) {
          ElMessage.warning('不支持的文件类型')
          event.target.value = ''
          return
        }
      }

      const formData = new FormData()
      formData.append('file', file)
      formData.append('file_type', this.currentFileType)

      try {
        await api.post(`/api/treatment/conversations/${this.selectedConversation.id}/upload/`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        ElMessage.success('文件上传成功')
        await this.fetchMessages()
        await this.fetchConversations()
        this.scrollToBottom()
      } catch (error) {
        ElMessage.error('文件上传失败')
      } finally {
        event.target.value = ''
      }
    },
    downloadFile(fileUrl, fileName) {
      const url = this.getFileUrl(fileUrl)
      const link = document.createElement('a')
      link.href = encodeURI(url)
      link.download = fileName
      // ensure new tab if browser blocks direct download
      link.target = '_blank'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    },
    getFileUrl(url) {
      if (!url) return ''
      if (url.startsWith('http')) return url
      // ensure leading slash
      const path = url.startsWith('/') ? url : `/${url}`
      const origin = window.location.origin || 'http://localhost:8000'
      return `${origin}${path}`
    },
    scrollToBottom() {
      if (this.messageListRef) {
        this.messageListRef.scrollTop = this.messageListRef.scrollHeight
      }
    },
    async openCreateConversationDialog() {
      // 获取患者列表
      try {
        const res = await api.get('/api/users/patients/')
        this.patients = res.data
      } catch (error) {
        ElMessage.error('获取患者列表失败')
      }
      // 获取病例列表
      try {
        const res = await api.get('/api/diagnosis/cases/')
        this.allCases = res.data
        this.cases = res.data
      } catch (error) {
        // 忽略错误
      }
      this.createConversationDialogVisible = true
    },
    onPatientChange() {
      if (!this.createConversationForm.patient_id) {
        this.cases = this.allCases
        return
      }
      const pid = this.createConversationForm.patient_id
      this.cases = this.allCases.filter(
        item => item.patient === pid || item.patient_id === pid
      )
      // 如果当前选择的病例不属于该患者，清空选择
      const selectedCase = this.cases.find(
        c => c.id === this.createConversationForm.case_id
      )
      if (!selectedCase) {
        this.createConversationForm.case_id = null
      }
    },
    async createConversation() {
      if (!this.createConversationForm.patient_id) {
        ElMessage.warning('请选择患者')
        return
      }
      this.creating = true
      try {
        const res = await api.post('/api/treatment/conversations/', this.createConversationForm)
        this.createConversationDialogVisible = false
        await this.fetchConversations()
        this.selectConversation(res.data)
        ElMessage.success('会话创建成功')
      } catch (error) {
        ElMessage.error('创建会话失败')
      } finally {
        this.creating = false
      }
    },
    async openTemplateDialog() {
      this.templateDialogVisible = true
      this.templateLoading = true
      try {
        const res = await api.get('/api/treatment/message-templates/')
        this.templates = res.data
      } catch (error) {
        ElMessage.error('获取模板失败')
      } finally {
        this.templateLoading = false
      }
    },
    useTemplate(template) {
      this.newMessage = template.content
      this.templateDialogVisible = false
      // 更新使用次数
      api.patch(`/api/treatment/message-templates/${template.id}/`, {
        usage_count: template.usage_count + 1
      }).catch(() => {})
    },
    formatTime(time) {
      if (!time) return ''
      const date = new Date(time)
      const now = new Date()
      const diff = now - date
      const minutes = Math.floor(diff / 60000)
      if (minutes < 1) return '刚刚'
      if (minutes < 60) return `${minutes}分钟前`
      const hours = Math.floor(minutes / 60)
      if (hours < 24) return `${hours}小时前`
      return date.toLocaleDateString('zh-CN') + ' ' + date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    },
    startPolling() {
      // 每30秒轮询一次未读消息
      this.pollTimer = setInterval(() => {
        this.fetchConversations()
        if (this.selectedConversation) {
          this.fetchMessages()
        }
      }, 30000)
    },
    stopPolling() {
      if (this.pollTimer) {
        clearInterval(this.pollTimer)
        this.pollTimer = null
      }
    }
  }
}
</script>

<style scoped>
.chat-page {
  padding: 20px;
  height: calc(100vh - 100px);
}

.conversation-list-card {
  height: 100%;
}

.conversation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.conversation-header h3 {
  margin: 0;
  font-size: 16px;
}

.search-input {
  margin-bottom: 15px;
}

.conversation-list {
  height: calc(100vh - 200px);
  overflow-y: auto;
}

.conversation-item {
  display: flex;
  padding: 12px;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
  transition: background-color 0.2s;
}

.conversation-item:hover {
  background-color: #f5f5f5;
}

.conversation-item.active {
  background-color: #ecf5ff;
}

.conversation-avatar {
  position: relative;
  margin-right: 12px;
}

.unread-badge {
  position: absolute;
  top: -5px;
  right: -5px;
}

.conversation-info {
  flex: 1;
  min-width: 0;
}

.conversation-name {
  font-weight: 500;
  font-size: 14px;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversation-last-message {
  font-size: 12px;
  color: #909399;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 4px;
}

.conversation-time {
  font-size: 11px;
  color: #c0c4cc;
}

.message-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

/* 让卡片内容区域变为可滚动布局，消息区固定高度内滚动 */
:deep(.message-card .el-card__body) {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 220px);
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.message-header-info {
  display: flex;
  align-items: center;
}

.message-header-name {
  font-weight: 500;
  font-size: 14px;
}

.message-header-meta {
  font-size: 12px;
  color: #909399;
}

.message-list {
  flex: 1;
  min-height: 0; /* 关键：允许在flex容器内收缩，避免撑开整体 */
  overflow-y: auto;
  padding: 20px;
  background-color: #f5f5f5;
}

.message-item {
  display: flex;
  margin-bottom: 20px;
}

.message-item.message-sent {
  flex-direction: row-reverse;
}

.message-avatar {
  flex-shrink: 0;
  margin: 0 10px;
}

.message-content {
  max-width: 60%;
}

.message-header-small {
  display: flex;
  justify-content: space-between;
  margin-bottom: 5px;
  font-size: 12px;
  color: #909399;
}

.message-sent .message-header-small {
  flex-direction: row-reverse;
}

.message-body {
  background-color: #fff;
  padding: 10px 15px;
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.message-sent .message-body {
  background-color: #409eff;
  color: #fff;
}

/* 对齐文本：己方消息右对齐，对方消息左对齐 */
.message-item.message-sent .message-content {
  text-align: right;
}
.message-item.message-received .message-content {
  text-align: left;
}

.message-text {
  word-wrap: break-word;
}

.message-image {
  max-width: 300px;
}

.message-file {
  display: flex;
  align-items: center;
  gap: 10px;
}

.message-related {
  margin-top: 8px;
}

.message-input-area {
  border-top: 1px solid #e6e6e6;
  padding: 15px;
  background-color: #fff;
  flex-shrink: 0; /* 保持输入区域和发送按钮位置固定 */
}

.input-toolbar {
  margin-bottom: 10px;
}

.input-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 10px;
}
</style>

