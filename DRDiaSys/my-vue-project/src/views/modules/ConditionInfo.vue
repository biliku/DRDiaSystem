<template>
  <div class="condition-info-container">
    <div class="condition-info-box">
      <!-- 头部 -->
      <div class="header">
        <div class="title-container">
          <h2>病情信息录入</h2>
          <p class="subtitle">请填写您的病情相关信息</p>
        </div>
        <el-button type="primary" @click="handleAdd">
          <el-icon><Plus /></el-icon>
          新增病情记录
        </el-button>
      </div>

      <!-- 病情记录列表 -->
      <el-card class="list-card" v-if="!showForm">
        <template #header>
          <div class="card-header">
            <span>病情记录列表</span>
            <el-tag type="info">共 {{ conditionList.length }} 条记录</el-tag>
          </div>
        </template>

        <el-table :data="conditionList" style="width: 100%" v-loading="loading">
          <el-table-column prop="id" label="ID" width="80" />
          <el-table-column label="糖尿病类型" width="120">
            <template #default="{ row }">
              <el-tag v-if="row.has_diabetes" type="warning">
                {{ getDiabetesTypeLabel(row.diabetes_type) }}
              </el-tag>
              <el-tag v-else type="success">无糖尿病</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="主要症状" min-width="200">
            <template #default="{ row }">
              <el-tag
                v-for="symptom in row.symptoms"
                :key="symptom"
                size="small"
                style="margin-right: 5px"
              >
                {{ getSymptomLabel(symptom) }}
              </el-tag>
              <span v-if="!row.symptoms || row.symptoms.length === 0">无</span>
            </template>
          </el-table-column>
          <el-table-column prop="symptom_duration" label="症状持续时间" width="150" />
          <el-table-column prop="created_at" label="创建时间" width="180">
            <template #default="{ row }">
              {{ formatDate(row.created_at) }}
            </template>
          </el-table-column>
          <el-table-column label="操作" width="180" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" size="small" @click="handleEdit(row)">编辑</el-button>
              <el-button type="danger" size="small" @click="handleDelete(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <!-- 表单卡片 -->
      <el-card class="form-card" v-if="showForm">
        <template #header>
          <div class="card-header">
            <span>{{ isEdit ? '编辑病情信息' : '新增病情信息' }}</span>
            <el-button text @click="handleCancel">返回列表</el-button>
          </div>
        </template>

        <el-form
          ref="formRef"
          :model="formData"
          :rules="rules"
          label-width="150px"
          label-position="left"
        >
          <el-divider content-position="left">糖尿病相关信息</el-divider>

          <el-form-item label="是否患有糖尿病" prop="has_diabetes">
            <el-switch v-model="formData.has_diabetes" />
          </el-form-item>

          <template v-if="formData.has_diabetes">
            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="糖尿病类型" prop="diabetes_type">
                  <el-select v-model="formData.diabetes_type" placeholder="请选择糖尿病类型" style="width: 100%">
                    <el-option label="1型糖尿病" value="TYPE1" />
                    <el-option label="2型糖尿病" value="TYPE2" />
                    <el-option label="妊娠期糖尿病" value="GESTATIONAL" />
                    <el-option label="其他类型" value="OTHER" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="糖尿病病程（年）" prop="diabetes_duration">
                  <el-input-number
                    v-model="formData.diabetes_duration"
                    :min="0"
                    :max="100"
                    placeholder="病程"
                    style="width: 100%"
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <el-row :gutter="20">
              <el-col :span="12">
                <el-form-item label="血糖水平（mmol/L）" prop="blood_sugar_level">
                  <el-input-number
                    v-model="formData.blood_sugar_level"
                    :min="0"
                    :max="50"
                    :precision="2"
                    placeholder="血糖水平"
                    style="width: 100%"
                  />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="糖化血红蛋白（%）" prop="hba1c">
                  <el-input-number
                    v-model="formData.hba1c"
                    :min="0"
                    :max="20"
                    :precision="2"
                    placeholder="HbA1c"
                    style="width: 100%"
                  />
                </el-form-item>
              </el-col>
            </el-row>
          </template>

          <el-divider content-position="left">症状信息</el-divider>

          <el-form-item label="主要症状" prop="symptoms">
            <el-checkbox-group v-model="formData.symptoms">
              <el-checkbox label="BLURRED_VISION">视力模糊</el-checkbox>
              <el-checkbox label="FLOATERS">飞蚊症</el-checkbox>
              <el-checkbox label="DARK_SPOTS">暗点</el-checkbox>
              <el-checkbox label="POOR_NIGHT_VISION">夜视能力差</el-checkbox>
              <el-checkbox label="COLOR_VISION_LOSS">色觉减退</el-checkbox>
              <el-checkbox label="NONE">无明显症状</el-checkbox>
            </el-checkbox-group>
          </el-form-item>

          <el-form-item label="症状详细描述" prop="symptom_description">
            <el-input
              v-model="formData.symptom_description"
              type="textarea"
              :rows="4"
              placeholder="请详细描述您的症状"
            />
          </el-form-item>

          <el-form-item label="症状持续时间" prop="symptom_duration">
            <el-input v-model="formData.symptom_duration" placeholder="例如：3个月、1年等" />
          </el-form-item>

          <el-divider content-position="left">病史信息</el-divider>

          <el-form-item label="既往病史" prop="medical_history">
            <el-input
              v-model="formData.medical_history"
              type="textarea"
              :rows="3"
              placeholder="请填写您的既往病史"
            />
          </el-form-item>

          <el-form-item label="家族病史" prop="family_history">
            <el-input
              v-model="formData.family_history"
              type="textarea"
              :rows="3"
              placeholder="请填写您的家族病史"
            />
          </el-form-item>

          <el-form-item label="用药史" prop="medication_history">
            <el-input
              v-model="formData.medication_history"
              type="textarea"
              :rows="3"
              placeholder="请填写您正在使用或曾经使用的药物"
            />
          </el-form-item>

          <el-form-item label="过敏史" prop="allergy_history">
            <el-input
              v-model="formData.allergy_history"
              type="textarea"
              :rows="3"
              placeholder="请填写您的过敏史"
            />
          </el-form-item>

          <el-form-item label="其他疾病" prop="other_conditions">
            <el-input
              v-model="formData.other_conditions"
              type="textarea"
              :rows="3"
              placeholder="请填写其他相关疾病"
            />
          </el-form-item>

          <el-form-item label="备注" prop="notes">
            <el-input
              v-model="formData.notes"
              type="textarea"
              :rows="3"
              placeholder="其他需要说明的信息"
            />
          </el-form-item>

          <el-form-item>
            <el-button type="primary" @click="handleSubmit" :loading="submitting">
              {{ isEdit ? '更新信息' : '保存信息' }}
            </el-button>
            <el-button @click="handleCancel">取消</el-button>
          </el-form-item>
        </el-form>
      </el-card>
    </div>
  </div>
</template>

<script>
import api from '../../api'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'

export default {
  name: 'ConditionInfo',
  components: {
    Plus
  },
  data() {
    return {
      loading: false,
      submitting: false,
      showForm: false,
      isEdit: false,
      conditionList: [],
      currentId: null,
      formData: {
        has_diabetes: false,
        diabetes_type: '',
        diabetes_duration: null,
        blood_sugar_level: null,
        hba1c: null,
        symptoms: [],
        symptom_description: '',
        symptom_duration: '',
        medical_history: '',
        family_history: '',
        medication_history: '',
        allergy_history: '',
        other_conditions: '',
        notes: ''
      },
      rules: {
        diabetes_type: [
          { required: true, message: '请选择糖尿病类型', trigger: 'change', validator: (rule, value, callback) => {
            if (this.formData.has_diabetes && !value) {
              callback(new Error('请选择糖尿病类型'))
            } else {
              callback()
            }
          }}
        ]
      }
    }
  },
  mounted() {
    this.fetchConditionList()
  },
  methods: {
    async fetchConditionList() {
      try {
        this.loading = true
        const response = await api.get('/api/patient/conditions/')
        this.conditionList = response.data || []
      } catch (error) {
        ElMessage.error('获取病情信息失败：' + (error.response?.data?.message || error.message))
      } finally {
        this.loading = false
      }
    },
    handleAdd() {
      this.isEdit = false
      this.currentId = null
      this.resetForm()
      this.showForm = true
    },
    handleEdit(row) {
      this.isEdit = true
      this.currentId = row.id
      this.formData = { ...row }
      this.showForm = true
    },
    handleCancel() {
      this.showForm = false
      this.resetForm()
    },
    resetForm() {
      this.formData = {
        has_diabetes: false,
        diabetes_type: '',
        diabetes_duration: null,
        blood_sugar_level: null,
        hba1c: null,
        symptoms: [],
        symptom_description: '',
        symptom_duration: '',
        medical_history: '',
        family_history: '',
        medication_history: '',
        allergy_history: '',
        other_conditions: '',
        notes: ''
      }
      if (this.$refs.formRef) {
        this.$refs.formRef.resetFields()
      }
    },
    async handleSubmit() {
      try {
        await this.$refs.formRef.validate()
        this.submitting = true

        if (this.isEdit) {
          await api.put(`/api/patient/conditions/${this.currentId}/`, this.formData)
          ElMessage.success('病情信息更新成功')
        } else {
          await api.post('/api/patient/conditions/', this.formData)
          ElMessage.success('病情信息保存成功')
        }

        this.showForm = false
        this.resetForm()
        this.fetchConditionList()
      } catch (error) {
        if (error.response && error.response.data) {
          const errors = error.response.data
          if (typeof errors === 'object') {
            const errorMessages = Object.values(errors).flat().join(', ')
            ElMessage.error('保存失败：' + errorMessages)
          } else {
            ElMessage.error('保存失败：' + (errors.message || error.message))
          }
        } else {
          ElMessage.error('保存失败：' + error.message)
        }
      } finally {
        this.submitting = false
      }
    },
    async handleDelete(row) {
      try {
        await ElMessageBox.confirm(
          `确定要删除这条病情记录吗？`,
          '确认删除',
          {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
          }
        )

        await api.delete(`/api/patient/conditions/${row.id}/`)
        ElMessage.success('删除成功')
        this.fetchConditionList()
      } catch (error) {
        if (error !== 'cancel') {
          ElMessage.error('删除失败：' + (error.response?.data?.message || error.message))
        }
      }
    },
    getDiabetesTypeLabel(type) {
      const map = {
        'TYPE1': '1型糖尿病',
        'TYPE2': '2型糖尿病',
        'GESTATIONAL': '妊娠期糖尿病',
        'OTHER': '其他类型',
        'NONE': '无糖尿病'
      }
      return map[type] || type
    },
    getSymptomLabel(symptom) {
      const map = {
        'BLURRED_VISION': '视力模糊',
        'FLOATERS': '飞蚊症',
        'DARK_SPOTS': '暗点',
        'POOR_NIGHT_VISION': '夜视能力差',
        'COLOR_VISION_LOSS': '色觉减退',
        'NONE': '无明显症状'
      }
      return map[symptom] || symptom
    },
    formatDate(dateString) {
      if (!dateString) return ''
      const date = new Date(dateString)
      return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      })
    }
  }
}
</script>

<style scoped>
.condition-info-container {
  padding: 20px;
  background-color: #f0f2f5;
  min-height: calc(100vh - 60px);
}

.condition-info-box {
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.title-container h2 {
  margin: 0;
  font-size: 24px;
  color: #303133;
  font-weight: 600;
}

.subtitle {
  margin: 5px 0 0 0;
  color: #909399;
  font-size: 14px;
}

.list-card,
.form-card {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

:deep(.el-divider__text) {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

:deep(.el-form-item__label) {
  font-weight: 500;
  color: #606266;
}

:deep(.el-checkbox-group) {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
}
</style>

