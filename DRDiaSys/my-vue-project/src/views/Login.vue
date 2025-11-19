<template>
  <div class="login-container">
    <div class="login-box">
      <div class="login-header">
        <div class="logo-container">
          <img src="../assets/logo.svg" alt="Logo" class="logo">
          <h1>DRDiaSystem</h1>
        </div>
        <p class="subtitle">糖尿病视网膜病变智能检测系统</p>
      </div>
      
      <!-- 登录表单 -->
      <el-form v-if="!isRegister" :model="loginForm" :rules="loginRules" ref="loginForm" class="login-form">
        <el-form-item prop="username">
          <el-input 
            v-model="loginForm.username"
            placeholder="用户名">
            <template #prefix>
              <el-icon><User /></el-icon>
            </template>
          </el-input>
        </el-form-item>
        
        <el-form-item prop="password">
          <el-input 
            v-model="loginForm.password"
            type="password"
            placeholder="密码"
            show-password>
            <template #prefix>
              <el-icon><Lock /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" class="submit-button" @click="handleLogin">
            登录
          </el-button>
        </el-form-item>
      </el-form>

      <!-- 注册表单 -->
      <el-form v-else :model="registerForm" :rules="registerRules" ref="registerForm" class="login-form">
        <el-form-item prop="username">
          <el-input 
            v-model="registerForm.username"
            placeholder="用户名">
            <template #prefix>
              <el-icon><User /></el-icon>
            </template>
          </el-input>
        </el-form-item>
        
        <el-form-item prop="password">
          <el-input 
            v-model="registerForm.password"
            type="password"
            placeholder="密码"
            show-password>
            <template #prefix>
              <el-icon><Lock /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item prop="confirmPassword">
          <el-input 
            v-model="registerForm.confirmPassword"
            type="password"
            placeholder="确认密码"
            show-password>
            <template #prefix>
              <el-icon><Lock /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item prop="email">
          <el-input 
            v-model="registerForm.email"
            placeholder="电子邮箱">
            <template #prefix>
              <el-icon><Message /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" class="submit-button" @click="handleRegister">
            注册
          </el-button>
        </el-form-item>
      </el-form>

      <!-- 切换按钮 -->
      <div class="switch-form">
        <span 
          :class="{ active: !isRegister }" 
          @click="isRegister = false"
        >登录</span>
        <span class="divider">|</span>
        <span 
          :class="{ active: isRegister }" 
          @click="isRegister = true"
        >注册</span>
      </div>

      <div class="tech-decoration">
        <div class="circle circle-1"></div>
        <div class="circle circle-2"></div>
        <div class="circle circle-3"></div>
      </div>
    </div>
  </div>
</template>

<script>
import { User, Lock, Message } from '@element-plus/icons-vue'


export default {
  name: 'LoginPage',
  data() {
    // 确认密码验证
    const validateConfirmPassword = (rule, value, callback) => {
      if (value === '') {
        callback(new Error('请再次输入密码'))
      } else if (value !== this.registerForm.password) {
        callback(new Error('两次输入密码不一致'))
      } else {
        callback()
      }
    }

    return {
      isRegister: false,
      loginForm: {
        username: '',
        password: ''
      },
      registerForm: {
        username: '',
        password: '',
        confirmPassword: '',
        email: ''
      },
      loginRules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' }
        ]
      },
      registerRules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' },
          { min: 2, max: 20, message: '长度在 2 到 20 个字符', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' },
          { min: 6, message: '密码长度不能小于6位', trigger: 'blur' }
        ],
        confirmPassword: [
          { required: true, message: '请再次输入密码', trigger: 'blur' },
          { validator: validateConfirmPassword, trigger: 'blur' }
        ],
        email: [
          { required: true, message: '请输入电子邮箱', trigger: 'blur' },
          { type: 'email', message: '请输入正确的电子邮箱地址', trigger: 'blur' }
        ]
      }
    }
  },
  methods: {
    handleLogin() {
      this.$refs.loginForm.validate(async (valid) => {
        if (valid) {
          try {
            const { data } = await this.axios.post('/api/login/', {
              username: this.loginForm.username,
              password: this.loginForm.password
            })

            // 存储登录信息
            localStorage.setItem('token', data.token)
            if (data.refresh) localStorage.setItem('refresh', data.refresh)
            localStorage.setItem('userName', data.username)
            localStorage.setItem('userRole', data.userRole)
            
            this.$message.success('登录成功')
            this.$router.push('/layout')
          } catch (error) {
            this.$message.error(error?.response?.data?.message || error.message || '登录失败')
          }
        }
      })
    },
    handleRegister() {
      this.$refs.registerForm.validate(async (valid) => {
        if (valid) {
          try {
            const response = await fetch('http://localhost:8000/api/register/', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                username: this.registerForm.username,
                password: this.registerForm.password,
                email: this.registerForm.email
              })
            })

            const data = await response.json()
            
            if (response.ok) {
              this.$message.success('注册成功')
              this.isRegister = false
            } else {
              throw new Error(data.message || '注册失败')
            }
          } catch (error) {
            this.$message.error(error.message)
          }
        }
      })
    }
  },

  // 在这里添加图标组件
  components: {
    User,
    Lock,
    Message
  }
}
</script>

<style scoped>
.login-container {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #1a1a1a 0%, #2c3e50 100%);
  position: relative;
  overflow: hidden;
}

.login-box {
  width: 400px;
  padding: 40px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
  position: relative;
  z-index: 1;
}

.login-header {
  text-align: center;
  margin-bottom: 40px;
}

.logo-container {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20px;
}

.logo {
  width: 50px;
  height: 50px;
  margin-right: 10px;
}

h1 {
  color: #fff;
  font-size: 24px;
  margin: 0;
  font-weight: 600;
}

.subtitle {
  color: #a8b2d1;
  font-size: 14px;
  margin-top: 10px;
}

.login-form {
  margin-top: 30px;
}

.submit-button {
  width: 100%;
  height: 45px;
  background: linear-gradient(45deg, #2196F3, #00BCD4);
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 500;
  transition: all 0.3s ease;
}

.submit-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
}

.switch-form {
  margin-top: 20px;
  text-align: center;
  color: #a8b2d1;
}

.switch-form span {
  cursor: pointer;
  transition: all 0.3s ease;
}

.switch-form span:hover {
  color: #2196F3;
}

.switch-form .active {
  color: #2196F3;
  font-weight: 500;
}

.switch-form .divider {
  margin: 0 10px;
  cursor: default;
}

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


:deep(.el-input__inner) {
  color: #333 !important;
  background: rgba(255, 255, 255, 0.9) !important;  
  border: 1px solid rgba(0, 0, 0, 0.1) !important; 
}


:deep(.el-input__inner::placeholder) {
  color: #666 !important;
}


:deep(.el-input--password .el-input__inner) {
  color: #333 !important;  
}


:deep(.el-input__prefix) {
  color: #666 !important;  
}


:deep(.el-input.is-active .el-input__inner),
:deep(.el-input__inner:focus) {
  border-color: #2196F3 !important;
  box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.2) !important;
}
</style> 