<template>
  <div class="execution-list">
    <el-button
      type="primary"
      @click="openAddDialog"
      style="margin-bottom: 15px"
    >
      <el-icon><Plus /></el-icon>
      添加执行记录
    </el-button>
    <el-table
      :data="executions"
      v-loading="loading"
      empty-text="暂无执行记录"
    >
      <el-table-column prop="execution_date" label="执行日期" width="150" />
      <el-table-column label="用药情况" min-width="200">
        <template #default="scope">
          <div v-if="Object.keys(scope.row.medication_taken || {}).length > 0">
            <el-tag
              v-for="(value, key) in scope.row.medication_taken"
              :key="key"
              style="margin-right: 5px"
            >
              {{ key }}: {{ value }}
            </el-tag>
          </div>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="复查完成" width="100">
        <template #default="scope">
          <el-tag :type="scope.row.follow_up_completed ? 'success' : 'info'">
            {{ scope.row.follow_up_completed ? '是' : '否' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="patient_feedback" label="患者反馈" min-width="200" show-overflow-tooltip />
      <el-table-column prop="doctor_notes" label="医生备注" min-width="200" show-overflow-tooltip />
      <el-table-column prop="created_by_name" label="记录人" width="100" />
      <el-table-column prop="created_at" label="记录时间" width="180">
        <template #default="scope">
          {{ formatTime(scope.row.created_at) }}
        </template>
      </el-table-column>
    </el-table>

    <!-- 添加执行记录对话框 -->
    <el-dialog
      v-model="addDialogVisible"
      title="添加执行记录"
      width="600px"
    >
      <el-form :model="executionForm" label-width="120px">
        <el-form-item label="执行日期" required>
          <el-date-picker
            v-model="executionForm.execution_date"
            type="date"
            placeholder="选择日期"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="用药情况">
          <div v-for="(item, index) in medicationList" :key="index" class="medication-taken-item">
            <el-input v-model="item.name" placeholder="药物名称" style="width: 150px" />
            <el-input v-model="item.value" placeholder="用药情况" style="width: 200px; margin-left: 10px" />
            <el-button
              type="danger"
              link
              @click="removeMedicationTaken(index)"
              style="margin-left: 10px"
            >
              删除
            </el-button>
          </div>
          <el-button type="primary" link @click="addMedicationTaken">
            <el-icon><Plus /></el-icon>
            添加用药记录
          </el-button>
        </el-form-item>
        <el-form-item label="复查完成">
          <el-switch v-model="executionForm.follow_up_completed" />
        </el-form-item>
        <el-form-item label="患者反馈">
          <el-input
            v-model="executionForm.patient_feedback"
            type="textarea"
            :rows="3"
            placeholder="患者反馈信息"
          />
        </el-form-item>
        <el-form-item label="医生备注">
          <el-input
            v-model="executionForm.doctor_notes"
            type="textarea"
            :rows="3"
            placeholder="医生备注"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="addDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveExecution" :loading="saving">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import api from '@/api'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'

export default {
  name: 'ExecutionList',
  components: {
    Plus
  },
  props: {
    planId: {
      type: Number,
      required: true
    }
  },
  data() {
    return {
      loading: false,
      executions: [],
      addDialogVisible: false,
      executionForm: {
        execution_date: null,
        medication_taken: {},
        follow_up_completed: false,
        patient_feedback: '',
        doctor_notes: ''
      },
      medicationList: [],
      saving: false
    }
  },
  created() {
    this.fetchExecutions()
  },
  watch: {
    planId() {
      this.fetchExecutions()
    }
  },
  methods: {
    async fetchExecutions() {
      this.loading = true
      try {
        const res = await api.get(`/api/treatment/plans/${this.planId}/executions/`)
        this.executions = res.data
      } catch (error) {
        ElMessage.error('获取执行记录失败')
      } finally {
        this.loading = false
      }
    },
    openAddDialog() {
      this.executionForm = {
        execution_date: new Date(),
        medication_taken: {},
        follow_up_completed: false,
        patient_feedback: '',
        doctor_notes: ''
      }
      this.medicationList = []
      this.addDialogVisible = true
    },
    addMedicationTaken() {
      this.medicationList.push({
        name: '',
        value: ''
      })
    },
    removeMedicationTaken(index) {
      this.medicationList.splice(index, 1)
    },
    async saveExecution() {
      if (!this.executionForm.execution_date) {
        ElMessage.warning('请选择执行日期')
        return
      }
      // 将medicationList转换为medication_taken对象
      const medicationTaken = {}
      this.medicationList.forEach(item => {
        if (item.name && item.name.trim()) {
          medicationTaken[item.name] = item.value || ''
        }
      })
      this.executionForm.medication_taken = medicationTaken
      
      this.saving = true
      try {
        await api.post(`/api/treatment/plans/${this.planId}/executions/`, this.executionForm)
        ElMessage.success('执行记录添加成功')
        this.addDialogVisible = false
        this.fetchExecutions()
      } catch (error) {
        ElMessage.error('添加执行记录失败')
      } finally {
        this.saving = false
      }
    },
    formatTime(time) {
      if (!time) return '-'
      return new Date(time).toLocaleString('zh-CN')
    }
  }
}
</script>

<style scoped>
.medication-taken-item {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}
</style>

