<template>
  <div class="treatment-plan-page" v-if="isDoctor">
    <el-row :gutter="20">
      <!-- 左侧：病例列表 -->
      <el-col :span="6">
        <el-card shadow="never" class="sidebar-card">
          <div class="sidebar-header">
            <h3>病例列表</h3>
            <el-button type="primary" size="small" @click="fetchCases">
              <el-icon><RefreshRight /></el-icon>
              刷新
            </el-button>
          </div>
          <el-input
            v-model="caseSearchQuery"
            placeholder="搜索患者"
            clearable
            class="search-input"
            @input="filterCases"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
          <el-table
            :data="filteredCases"
            height="calc(100vh - 200px)"
            v-loading="caseLoading"
            @row-click="handleCaseSelect"
            :row-class-name="rowClassName"
          >
            <el-table-column prop="title" label="病例" min-width="120" />
            <el-table-column prop="patient_name" label="患者" width="100" />
          </el-table>
        </el-card>
      </el-col>

      <!-- 右侧：治疗方案管理 -->
      <el-col :span="18">
        <el-card shadow="never" v-if="selectedCase">
          <template #header>
            <div class="card-header">
              <div>
                <h3>{{ selectedCase.title }}</h3>
                <p class="patient-info">患者：{{ selectedCase.patient_name }}</p>
              </div>
              <div class="header-actions">
                <el-button type="primary" @click="openRecommendDialog">
                  <el-icon><MagicStick /></el-icon>
                  AI推荐方案
                </el-button>
                <el-button type="success" @click="openCreateDialog">
                  <el-icon><Plus /></el-icon>
                  新建方案
                </el-button>
              </div>
            </div>
          </template>

          <!-- 方案列表 -->
          <el-tabs v-model="activeTab">
            <el-tab-pane label="进行中" name="active">
              <TreatmentPlanList
                :case-id="selectedCase.id"
                status="active"
                @edit="handleEdit"
                @confirm="handleConfirm"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
            <el-tab-pane label="已确认" name="confirmed">
              <TreatmentPlanList
                :case-id="selectedCase.id"
                status="confirmed"
                @edit="handleEdit"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
            <el-tab-pane label="已完成" name="completed">
              <TreatmentPlanList
                :case-id="selectedCase.id"
                status="completed"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
            <el-tab-pane label="全部" name="all">
              <TreatmentPlanList
                :case-id="selectedCase.id"
                @edit="handleEdit"
                @confirm="handleConfirm"
                @view-executions="handleViewExecutions"
              />
            </el-tab-pane>
          </el-tabs>
        </el-card>
        <el-card v-else shadow="never">
          <el-empty description="请选择一个病例" />
        </el-card>
      </el-col>
    </el-row>

    <!-- AI推荐对话框 -->
    <el-dialog
      v-model="recommendDialogVisible"
      title="AI治疗方案推荐"
      width="900px"
      :close-on-click-modal="false"
    >
      <div v-loading="recommendLoading">
        <div v-if="recommendations.length > 0">
          <el-alert
            type="info"
            :closable="false"
            style="margin-bottom: 20px"
          >
            系统基于诊断报告和患者信息推荐了以下治疗方案，请选择或自定义
          </el-alert>
          <el-row :gutter="20">
            <el-col
              v-for="(rec, index) in recommendations"
              :key="index"
              :span="12"
              style="margin-bottom: 20px"
            >
              <el-card shadow="hover" class="recommendation-card">
                <div class="recommendation-header">
                  <el-tag type="success">推荐度: {{ rec.score.toFixed(0) }}</el-tag>
                  <el-button
                    type="primary"
                    size="small"
                    @click="applyRecommendation(rec)"
                  >
                    应用此方案
                  </el-button>
                </div>
                <p class="match-reason">{{ rec.match_reason }}</p>
                <div class="recommendation-content">
                  <h4>药物治疗</h4>
                  <ul>
                    <li v-for="(med, i) in rec.medications" :key="i">
                      {{ med.name }} - {{ med.dosage }}
                    </li>
                  </ul>
                  <h4>复查计划</h4>
                  <p>下次复查：{{ rec.follow_up_plan.next_date || '待定' }}</p>
                  <p>检查项目：{{ rec.follow_up_plan.check_items?.join('、') || '待定' }}</p>
                  <h4>生活方式建议</h4>
                  <p>{{ rec.lifestyle_advice }}</p>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </div>
        <el-empty v-else description="暂无推荐方案" />
      </div>
    </el-dialog>

    <!-- 创建/编辑方案对话框 -->
    <el-dialog
      v-model="planDialogVisible"
      :title="editingPlan ? '编辑治疗方案' : '新建治疗方案'"
      width="800px"
      :close-on-click-modal="false"
    >
      <el-form :model="planForm" label-width="120px" ref="planFormRef">
        <el-form-item label="方案标题" required>
          <el-input v-model="planForm.title" placeholder="例如：DR1级治疗方案" />
        </el-form-item>
        <el-form-item label="药物治疗">
          <div v-for="(med, index) in planForm.medications" :key="index" class="medication-item">
            <el-input v-model="med.name" placeholder="药物名称" style="width: 200px" />
            <el-input v-model="med.dosage" placeholder="用法用量" style="width: 200px; margin-left: 10px" />
            <el-input v-model="med.duration" placeholder="疗程" style="width: 150px; margin-left: 10px" />
            <el-button
              type="danger"
              link
              @click="removeMedication(index)"
              style="margin-left: 10px"
            >
              删除
            </el-button>
          </div>
          <el-button type="primary" link @click="addMedication">
            <el-icon><Plus /></el-icon>
            添加药物
          </el-button>
        </el-form-item>
        <el-form-item label="复查计划">
          <el-date-picker
            v-model="planForm.follow_up_plan.next_date"
            type="date"
            placeholder="选择复查日期"
            style="width: 100%"
          />
          <el-input
            v-model="planForm.follow_up_plan.check_items_text"
            placeholder="检查项目（用逗号分隔）"
            style="margin-top: 10px"
          />
          <el-input-number
            v-model="planForm.follow_up_plan.interval_days"
            :min="30"
            :max="365"
            placeholder="复查间隔（天）"
            style="width: 100%; margin-top: 10px"
          />
        </el-form-item>
        <el-form-item label="生活方式建议">
          <el-input
            v-model="planForm.lifestyle_advice"
            type="textarea"
            :rows="4"
            placeholder="饮食、运动、血糖监测等建议"
          />
        </el-form-item>
        <el-form-item label="注意事项">
          <el-input
            v-model="planForm.precautions"
            type="textarea"
            :rows="4"
            placeholder="用药注意事项、风险提示等"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="planDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="savePlan" :loading="saving">保存</el-button>
      </template>
    </el-dialog>

    <!-- 执行记录对话框 -->
    <el-dialog
      v-model="executionDialogVisible"
      title="方案执行记录"
      width="700px"
    >
      <ExecutionList :plan-id="selectedPlanId" />
    </el-dialog>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  RefreshRight,
  Search,
  MagicStick,
  Plus
} from '@element-plus/icons-vue'
import TreatmentPlanList from './components/TreatmentPlanList.vue'
import ExecutionList from './components/ExecutionList.vue'

export default {
  name: 'TreatmentPlan',
  components: {
    RefreshRight,
    Search,
    MagicStick,
    Plus,
    TreatmentPlanList,
    ExecutionList
  },
  data() {
    return {
      isDoctor: localStorage.getItem('userRole') === 'doctor',
      caseLoading: false,
      cases: [],
      filteredCases: [],
      caseSearchQuery: '',
      selectedCase: null,
      activeTab: 'active',
      recommendDialogVisible: false,
      recommendLoading: false,
      recommendations: [],
      planDialogVisible: false,
      editingPlan: null,
      planForm: {
        title: '',
        medications: [],
        follow_up_plan: {
          next_date: null,
          check_items_text: '',
          interval_days: 90
        },
        lifestyle_advice: '',
        precautions: ''
      },
      planFormRef: null,
      saving: false,
      executionDialogVisible: false,
      selectedPlanId: null
    }
  },
  created() {
    this.fetchCases()
  },
  methods: {
    async fetchCases() {
      this.caseLoading = true
      try {
        const res = await api.get('/api/diagnosis/cases/')
        this.cases = res.data
        this.filteredCases = res.data
      } catch (error) {
        ElMessage.error('获取病例列表失败')
      } finally {
        this.caseLoading = false
      }
    },
    filterCases() {
      if (!this.caseSearchQuery) {
        this.filteredCases = this.cases
        return
      }
      this.filteredCases = this.cases.filter(caseItem =>
        caseItem.title.includes(this.caseSearchQuery) ||
        caseItem.patient_name.includes(this.caseSearchQuery)
      )
    },
    handleCaseSelect(row) {
      this.selectedCase = row
    },
    rowClassName({ row }) {
      return row.id === this.selectedCase?.id ? 'selected-row' : ''
    },
    async openRecommendDialog() {
      if (!this.selectedCase) {
        ElMessage.warning('请先选择病例')
        return
      }
      this.recommendDialogVisible = true
      this.recommendLoading = true
      try {
        const res = await api.get('/api/treatment/recommend/', {
          params: {
            case_id: this.selectedCase.id,
            report_id: this.selectedCase.primary_report_info?.id
          }
        })
        this.recommendations = res.data.recommendations || []
      } catch (error) {
        ElMessage.error('获取推荐方案失败')
      } finally {
        this.recommendLoading = false
      }
    },
    async applyRecommendation(rec) {
      try {
        await api.post('/api/treatment/recommend/', {
          case_id: this.selectedCase.id,
          template_id: rec.template_id,
          report_id: this.selectedCase.primary_report_info?.id,
          title: `DR${rec.dr_grade}级治疗方案`
        })
        ElMessage.success('方案创建成功')
        this.recommendDialogVisible = false
        // 刷新方案列表
        this.$forceUpdate()
      } catch (error) {
        ElMessage.error('应用方案失败')
      }
    },
    openCreateDialog() {
      if (!this.selectedCase) {
        ElMessage.warning('请先选择病例')
        return
      }
      this.editingPlan = null
      this.planForm = {
        title: '',
        medications: [],
        follow_up_plan: {
          next_date: null,
          check_items_text: '',
          interval_days: 90
        },
        lifestyle_advice: '',
        precautions: ''
      }
      this.planDialogVisible = true
    },
    addMedication() {
      this.planForm.medications.push({
        name: '',
        dosage: '',
        duration: '',
        notes: ''
      })
    },
    removeMedication(index) {
      this.planForm.medications.splice(index, 1)
    },
    async savePlan() {
      if (!this.planForm.title) {
        ElMessage.warning('请输入方案标题')
        return
      }
      this.saving = true
      try {
        const data = {
          case: this.selectedCase.id,
          title: this.planForm.title,
          medications: this.planForm.medications,
          follow_up_plan: {
            next_date: this.planForm.follow_up_plan.next_date,
            check_items: this.planForm.follow_up_plan.check_items_text
              .split(',').map(s => s.trim()).filter(s => s),
            interval_days: this.planForm.follow_up_plan.interval_days
          },
          lifestyle_advice: this.planForm.lifestyle_advice,
          precautions: this.planForm.precautions
        }
        if (this.editingPlan) {
          await api.patch(`/api/treatment/plans/${this.editingPlan.id}/`, data)
          ElMessage.success('方案更新成功')
        } else {
          await api.post('/api/treatment/plans/', data)
          ElMessage.success('方案创建成功')
        }
        this.planDialogVisible = false
        this.$forceUpdate()
      } catch (error) {
        ElMessage.error('保存方案失败')
      } finally {
        this.saving = false
      }
    },
    handleEdit(plan) {
      this.editingPlan = plan
      this.planForm = {
        title: plan.title,
        medications: plan.medications || [],
        follow_up_plan: {
          next_date: plan.follow_up_plan?.next_date || null,
          check_items_text: plan.follow_up_plan?.check_items?.join(',') || '',
          interval_days: plan.follow_up_plan?.interval_days || 90
        },
        lifestyle_advice: plan.lifestyle_advice || '',
        precautions: plan.precautions || ''
      }
      this.planDialogVisible = true
    },
    async handleConfirm(plan) {
      try {
        await ElMessageBox.confirm('确认后方案将通知患者，是否继续？', '确认方案', {
          type: 'warning'
        })
        await api.post(`/api/treatment/plans/${plan.id}/confirm/`)
        ElMessage.success('方案已确认')
        this.$forceUpdate()
      } catch (error) {
        if (error !== 'cancel') {
          ElMessage.error('确认方案失败')
        }
      }
    },
    handleViewExecutions(plan) {
      this.selectedPlanId = plan.id
      this.executionDialogVisible = true
    }
  }
}
</script>

<style scoped>
.treatment-plan-page {
  padding: 20px;
}

.sidebar-card {
  height: calc(100vh - 100px);
}

.sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.sidebar-header h3 {
  margin: 0;
  font-size: 16px;
}

.search-input {
  margin-bottom: 15px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h3 {
  margin: 0;
}

.patient-info {
  margin: 5px 0 0 0;
  color: #909399;
  font-size: 14px;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.recommendation-card {
  height: 100%;
}

.recommendation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.match-reason {
  color: #909399;
  font-size: 12px;
  margin-bottom: 15px;
}

.recommendation-content h4 {
  margin: 15px 0 10px 0;
  font-size: 14px;
  color: #303133;
}

.recommendation-content ul {
  margin: 0;
  padding-left: 20px;
}

.recommendation-content p {
  margin: 5px 0;
  color: #606266;
  font-size: 13px;
}

.medication-item {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

:deep(.selected-row) {
  background-color: #ecf5ff;
  cursor: pointer;
}
</style>

