import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Layout from '../views/Layout.vue'
import UserManagement from '../views/modules/UserManagement.vue'
import ImageProcessing from '../views/modules/ImageProcessing.vue'
import Dashboard from '../views/modules/Dashboard.vue'
import PersonalInfo from '../views/modules/PersonalInfo.vue'
import ConditionInfo from '../views/modules/ConditionInfo.vue'
import EyeImageView from '../views/modules/EyeImageView.vue'
import ReportView from '../views/modules/ReportView.vue'
import MedicalRecord from '../views/modules/MedicalRecord.vue'
import DoctorPlan from '../views/modules/treatment/DoctorPlan.vue'
import PatientPlan from '../views/modules/treatment/PatientPlan.vue'
import DoctorPatientChat from '../views/modules/DoctorPatientChat.vue'

const routes = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/login',
    name: 'Login',
    component: Login
  },
  {
    path: '/layout',
    name: 'Layout',
    component: Layout,
    children: [
      {
        path: '/user-management',
        name: 'UserManagement',
        component: UserManagement
      },
      {
        path: '/image-processing',
        name: 'ImageProcessing',
        component: ImageProcessing
      },
      {
        path: '/dashboard',
        name: 'Dashboard',
        component: Dashboard
      },
      {
        path: '/personal-info',
        name: 'PersonalInfo',
        component: PersonalInfo
      },
      {
        path: '/condition-info',
        name: 'ConditionInfo',
        component: ConditionInfo
      },
      {
        path: '/eye-image-view',
        name: 'EyeImageView',
        component: EyeImageView
      },
      {
        path: '/report-view',
        name: 'ReportView',
        component: ReportView
      },
      {
        path: '/medical-record',
        name: 'MedicalRecord',
        component: MedicalRecord
      },
      {
        path: '/treatment-plan',
        name: 'TreatmentPlan',
        component: DoctorPlan
      },
      {
        path: '/patient-treatment-plan',
        name: 'PatientTreatmentPlan',
        component: PatientPlan
      },
      {
        path: '/doctor-patient-chat',
        name: 'DoctorPatientChat',
        component: DoctorPatientChat
      }
      // 其他子路由可以在这里添加
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 全局前置守卫
router.beforeEach((to, from, next) => {
  // 如果是登录页面，直接通过（不检查token）
  if (to.path === '/login') {
    // 如果已经登录，跳转到首页
    const token = localStorage.getItem('token')
    if (token) {
      next('/dashboard')
    } else {
    next()
    }
    return
  }
  
  // 检查是否有token
  const token = localStorage.getItem('token')
  if (!token) {
    next('/login')
    return
  }
  
  next()
})

export default router 