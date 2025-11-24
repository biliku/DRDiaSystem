<template>
  <div class="personal-info-container">
    <div class="personal-info-box">
      <!-- 头部 -->
      <div class="header">
        <div class="title-container">
          <h2>个人信息录入</h2>
          <p class="subtitle">请填写您的个人基本信息</p>
        </div>
      </div>

      <!-- 表单卡片 -->
      <el-card class="form-card">
        <el-form
          ref="formRef"
          :model="formData"
          :rules="rules"
          label-width="120px"
          label-position="left"
        >
          <el-divider content-position="left">基本信息</el-divider>
          
          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="真实姓名" prop="real_name">
                <el-input v-model="formData.real_name" placeholder="请输入真实姓名" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="性别" prop="gender">
                <el-radio-group v-model="formData.gender">
                  <el-radio label="M">男</el-radio>
                  <el-radio label="F">女</el-radio>
                  <el-radio label="O">其他</el-radio>
                </el-radio-group>
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="出生日期" prop="birth_date">
                <el-date-picker
                  v-model="formData.birth_date"
                  type="date"
                  placeholder="选择出生日期"
                  style="width: 100%"
                  format="YYYY-MM-DD"
                  value-format="YYYY-MM-DD"
                />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="年龄" prop="age">
                <el-input-number
                  v-model="formData.age"
                  :min="0"
                  :max="150"
                  placeholder="年龄"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="身份证号" prop="id_card">
                <el-input v-model="formData.id_card" placeholder="请输入身份证号" maxlength="18" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="联系电话" prop="phone">
                <el-input v-model="formData.phone" placeholder="请输入联系电话" />
              </el-form-item>
            </el-col>
          </el-row>

          <el-form-item label="邮箱" prop="email">
            <el-input v-model="formData.email" placeholder="请输入邮箱地址" type="email" />
          </el-form-item>

          <el-divider content-position="left">地址信息</el-divider>

          <el-row :gutter="20">
            <el-col :span="8">
              <el-form-item label="省份" prop="province">
                <el-input v-model="formData.province" placeholder="请输入省份" />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="城市" prop="city">
                <el-input v-model="formData.city" placeholder="请输入城市" />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="区县" prop="district">
                <el-input v-model="formData.district" placeholder="请输入区县" />
              </el-form-item>
            </el-col>
          </el-row>

          <el-form-item label="详细地址" prop="address">
            <el-input
              v-model="formData.address"
              type="textarea"
              :rows="3"
              placeholder="请输入详细地址"
            />
          </el-form-item>

          <el-divider content-position="left">医疗信息</el-divider>

          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="血型" prop="blood_type">
                <el-select v-model="formData.blood_type" placeholder="请选择血型" style="width: 100%">
                  <el-option label="A型" value="A" />
                  <el-option label="B型" value="B" />
                  <el-option label="AB型" value="AB" />
                  <el-option label="O型" value="O" />
                  <el-option label="未知" value="UNKNOWN" />
                </el-select>
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="紧急联系人" prop="emergency_contact">
                <el-input v-model="formData.emergency_contact" placeholder="请输入紧急联系人姓名" />
              </el-form-item>
            </el-col>
          </el-row>

          <el-form-item label="紧急联系电话" prop="emergency_phone">
            <el-input v-model="formData.emergency_phone" placeholder="请输入紧急联系电话" />
          </el-form-item>

          <el-form-item>
            <el-button type="primary" @click="handleSubmit" :loading="loading">
              {{ isEdit ? '更新信息' : '保存信息' }}
            </el-button>
            <el-button @click="handleReset">重置</el-button>
          </el-form-item>
        </el-form>
      </el-card>
    </div>
  </div>
</template>

<script>
import api from '../../api'
import { ElMessage } from 'element-plus'

export default {
  name: 'PersonalInfo',
  data() {
    return {
      loading: false,
      isEdit: false,
      formData: {
        real_name: '',
        gender: '',
        birth_date: '',
        age: null,
        id_card: '',
        phone: '',
        email: '',
        address: '',
        province: '',
        city: '',
        district: '',
        blood_type: '',
        emergency_contact: '',
        emergency_phone: ''
      },
      rules: {
        real_name: [
          { required: true, message: '请输入真实姓名', trigger: 'blur' }
        ],
        phone: [
          { required: true, message: '请输入联系电话', trigger: 'blur' },
          { pattern: /^1[3-9]\d{9}$/, message: '请输入正确的手机号码', trigger: 'blur' }
        ],
        email: [
          { type: 'email', message: '请输入正确的邮箱地址', trigger: 'blur' }
        ],
        id_card: [
          { pattern: /^[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]$/, message: '请输入正确的身份证号', trigger: 'blur' }
        ]
      }
    }
  },
  mounted() {
    this.fetchPatientInfo()
  },
  methods: {
    async fetchPatientInfo() {
      try {
        this.loading = true
        const response = await api.get('/api/patient/info/')
        if (response.data) {
          this.formData = { ...response.data }
          this.isEdit = true
        }
      } catch (error) {
        if (error.response && error.response.status === 404) {
          // 信息不存在，这是正常的，用户需要填写
          this.isEdit = false
        } else {
          ElMessage.error('获取个人信息失败：' + (error.response?.data?.message || error.message))
        }
      } finally {
        this.loading = false
      }
    },
    async handleSubmit() {
      try {
        await this.$refs.formRef.validate()
        this.loading = true

        const method = this.isEdit ? 'put' : 'post'
        const response = await api[method]('/api/patient/info/', this.formData)

        ElMessage.success(this.isEdit ? '个人信息更新成功' : '个人信息保存成功')
        this.isEdit = true
        this.formData = { ...response.data }
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
        this.loading = false
      }
    },
    handleReset() {
      this.$refs.formRef.resetFields()
      if (this.isEdit) {
        this.fetchPatientInfo()
      }
    }
  }
}
</script>

<style scoped>
.personal-info-container {
  padding: 20px;
  background-color: #f0f2f5;
  min-height: calc(100vh - 60px);
}

.personal-info-box {
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

.form-card {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
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

:deep(.el-input),
:deep(.el-select),
:deep(.el-date-picker) {
  width: 100%;
}
</style>

