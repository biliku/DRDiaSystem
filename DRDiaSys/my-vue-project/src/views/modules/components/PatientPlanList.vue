<template>
  <div class="patient-plan-list">
    <el-table
      :data="plans"
      v-loading="loading"
      empty-text="暂无治疗方案"
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
      <el-table-column label="创建时间" width="180">
        <template #default="scope">
          {{ formatTime(scope.row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="scope">
          <el-button
            type="primary"
            link
            size="small"
            @click="$emit('view-detail', scope.row)"
          >
            查看详情
          </el-button>
          <el-button
            v-if="scope.row.status === 'active' || scope.row.status === 'confirmed'"
            type="success"
            link
            size="small"
            @click="$emit('add-execution', scope.row)"
          >
            记录执行
          </el-button>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script>
export default {
  name: 'PatientPlanList',
  props: {
    plans: {
      type: Array,
      default: () => []
    },
    loading: {
      type: Boolean,
      default: false
    }
  },
  methods: {
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
.patient-plan-list {
  margin-top: 20px;
}
</style>

