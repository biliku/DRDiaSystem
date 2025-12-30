<template>
  <div class="treatment-plan-list">
    <el-table
      :data="plans"
      v-loading="loading"
      empty-text="暂无治疗方案"
      @row-click="handleRowClick"
    >
      <el-table-column prop="plan_number" label="方案编号" width="180" />
      <el-table-column prop="title" label="方案标题" min-width="150" />
      <el-table-column label="状态" width="100">
        <template #default="scope">
          <el-tag :type="getStatusType(scope.row.status)">
            {{ getStatusLabel(scope.row.status) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="AI推荐" width="100">
        <template #default="scope">
          <el-tag v-if="scope.row.is_ai_recommended" type="success" size="small">是</el-tag>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="创建时间" width="180">
        <template #default="scope">
          {{ formatTime(scope.row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="250" fixed="right">
        <template #default="scope">
          <el-button
            v-if="scope.row.status === 'draft'"
            type="primary"
            link
            size="small"
            @click.stop="$emit('edit', scope.row)"
          >
            编辑
          </el-button>
          <el-button
            v-if="scope.row.status === 'draft'"
            type="success"
            link
            size="small"
            @click.stop="$emit('confirm', scope.row)"
          >
            确认
          </el-button>
          <el-button
            type="info"
            link
            size="small"
            @click.stop="$emit('view-executions', scope.row)"
          >
            执行记录
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 方案详情对话框 -->
    <el-dialog
      v-model="detailDialogVisible"
      title="治疗方案详情"
      width="800px"
    >
      <div v-if="selectedPlan">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="方案编号">{{ selectedPlan.plan_number }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(selectedPlan.status)">
              {{ getStatusLabel(selectedPlan.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="方案标题" :span="2">{{ selectedPlan.title }}</el-descriptions-item>
        </el-descriptions>

        <h4 style="margin-top: 20px">药物治疗</h4>
        <el-table :data="selectedPlan.medications" border>
          <el-table-column prop="name" label="药物名称" />
          <el-table-column prop="dosage" label="用法用量" />
          <el-table-column prop="duration" label="疗程" />
          <el-table-column prop="notes" label="备注" />
        </el-table>

        <h4 style="margin-top: 20px">复查计划</h4>
        <el-descriptions :column="1" border>
          <el-descriptions-item label="下次复查日期">
            {{ selectedPlan.follow_up_plan?.next_date || '待定' }}
          </el-descriptions-item>
          <el-descriptions-item label="检查项目">
            {{ selectedPlan.follow_up_plan?.check_items?.join('、') || '待定' }}
          </el-descriptions-item>
          <el-descriptions-item label="复查间隔">
            {{ selectedPlan.follow_up_plan?.interval_days || '-' }} 天
          </el-descriptions-item>
        </el-descriptions>

        <h4 style="margin-top: 20px">生活方式建议</h4>
        <p>{{ selectedPlan.lifestyle_advice || '无' }}</p>

        <h4 style="margin-top: 20px">注意事项</h4>
        <p>{{ selectedPlan.precautions || '无' }}</p>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage } from 'element-plus'

export default {
  name: 'TreatmentPlanList',
  props: {
    caseId: {
      type: Number,
      required: true
    },
    status: {
      type: String,
      default: ''
    }
  },
  data() {
    return {
      loading: false,
      plans: [],
      detailDialogVisible: false,
      selectedPlan: null
    }
  },
  created() {
    this.fetchPlans()
  },
  watch: {
    caseId() {
      this.fetchPlans()
    },
    status() {
      this.fetchPlans()
    }
  },
  methods: {
    async fetchPlans() {
      this.loading = true
      try {
        const params = { case_id: this.caseId }
        if (this.status && this.status !== 'all') {
          params.status = this.status
        }
        const res = await api.get('/api/treatment/plans/', { params })
        this.plans = res.data
      } catch (error) {
        ElMessage.error('获取方案列表失败')
      } finally {
        this.loading = false
      }
    },
    handleRowClick(row) {
      this.selectedPlan = row
      this.detailDialogVisible = true
    },
    getStatusType(status) {
      const map = {
        draft: 'info',
        confirmed: 'warning',
        active: 'success',
        completed: '',
        cancelled: 'danger'
      }
      return map[status] || ''
    },
    getStatusLabel(status) {
      const map = {
        draft: '草稿',
        confirmed: '已确认',
        active: '执行中',
        completed: '已完成',
        cancelled: '已取消'
      }
      return map[status] || status
    },
    formatTime(time) {
      if (!time) return '-'
      return new Date(time).toLocaleString('zh-CN')
    }
  }
}
</script>

<style scoped>
.treatment-plan-list {
  margin-top: 20px;
}
</style>

